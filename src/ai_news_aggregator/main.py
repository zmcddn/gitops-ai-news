#!/usr/bin/env python3
"""Daily AI news aggregator designed for GitHub Actions/GitOps.

Fetches RSS/Atom feeds, scores and deduplicates items, asks GitHub Models for a
single Markdown briefing, and writes the result back into the repository.

The script is intentionally serverless: the repository is the database, the
workflow scheduler is the cron, and GitHub Pages can serve the generated docs.
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import os
import re
import sys
import textwrap
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests
import yaml
from dateutil import parser as date_parser
from zoneinfo import ZoneInfo

USER_AGENT = (
    "gitops-ai-news/0.1 (+https://github.com/your-user/gitops-ai-news; "
    "RSS fetcher for a personal daily digest)"
)

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "ref",
    "ref_src",
    "spm",
}

DEFAULT_MODEL_ENDPOINT = "https://models.github.ai/inference/chat/completions"
GITHUB_API_VERSION = "2022-11-28"


@dataclass
class NewsItem:
    title: str
    url: str
    canonical_url: str
    source: str
    published: str
    summary: str = ""
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def published_dt(self) -> datetime:
        return parse_datetime(self.published) or datetime.now(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GitOps AI news digest.")
    parser.add_argument("--config", default="config/sources.yml", help="Path to YAML config")
    parser.add_argument("--no-ai", action="store_true", help="Skip LLM call and use deterministic Markdown")
    parser.add_argument("--dry-run", action="store_true", help="Print digest but do not write files")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if "feeds" not in config or not config["feeds"]:
        raise ValueError("config must include a non-empty 'feeds' list")
    return config


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = date_parser.parse(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def entry_datetime(entry: Any, fallback: datetime) -> datetime:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            except Exception:
                pass

    for key in ("published", "updated", "created"):
        parsed = parse_datetime(entry.get(key))
        if parsed:
            return parsed

    return fallback


def clean_text(value: str | None, max_chars: int = 500) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower().startswith("utm_") or key.lower() in TRACKING_QUERY_KEYS:
            continue
        query.append((key, value))

    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")

    normalized = parts._replace(
        scheme=parts.scheme.lower() or "https",
        netloc=parts.netloc.lower(),
        path=path,
        query=urlencode(query, doseq=True),
        fragment="",
    )
    return urlunsplit(normalized)


def title_fingerprint(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    stopwords = {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with", "via", "new"}
    tokens = [token for token in title.split() if token not in stopwords]
    return " ".join(tokens[:18])


def entry_summary_value(entry: Any) -> str:
    content = entry.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if hasattr(first, "get"):
            return str(first.get("value", ""))
    return ""


def fetch_feed(feed: dict[str, Any], now: datetime) -> list[NewsItem]:
    name = str(feed.get("name") or feed.get("url") or "unknown source")
    url = str(feed.get("url") or "")
    if not url:
        return []

    timeout = int(feed.get("timeout_seconds", 25))
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        print(f"WARN: feed fetch failed for {name}: {exc}", file=sys.stderr)
        return []

    parsed = feedparser.parse(response.content)
    if parsed.bozo:
        print(f"WARN: feed parsed with warnings for {name}: {parsed.bozo_exception}", file=sys.stderr)

    limit = int(feed.get("entry_limit", 60))
    items: list[NewsItem] = []
    for entry in parsed.entries[:limit]:
        title = clean_text(entry.get("title"), max_chars=180)
        link = entry.get("link") or entry.get("id") or ""
        if not title or not link:
            continue

        published_dt = entry_datetime(entry, fallback=now)
        summary = clean_text(
            entry.get("summary") or entry.get("description") or entry.get("subtitle") or entry_summary_value(entry),
            max_chars=650,
        )
        items.append(
            NewsItem(
                title=title,
                url=link,
                canonical_url=canonicalize_url(link),
                source=name,
                published=published_dt.isoformat(),
                summary=summary,
            )
        )
    return items


def load_seen(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(key): str(value) for key, value in data.get("urls", data).items()}
    except Exception as exc:
        print(f"WARN: could not read seen URL cache {path}: {exc}", file=sys.stderr)
        return {}


def save_seen(path: Path, seen: dict[str, str], now: datetime, keep_days: int) -> None:
    cutoff = now - timedelta(days=keep_days)
    compact: dict[str, str] = {}
    for url, value in seen.items():
        parsed = parse_datetime(value)
        if parsed is None or parsed >= cutoff:
            compact[url] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"updated_at": now.isoformat(), "urls": compact}, f, indent=2, sort_keys=True)
        f.write("\n")


def score_item(item: NewsItem, config: dict[str, Any], feed_weights: dict[str, float], now: datetime) -> NewsItem:
    keywords = config.get("keywords", {}) or {}
    high = [str(keyword).lower() for keyword in keywords.get("high", [])]
    medium = [str(keyword).lower() for keyword in keywords.get("medium", [])]
    penalties = [str(keyword).lower() for keyword in keywords.get("penalty", [])]

    text = f"{item.title}\n{item.summary}".lower()
    score = float(feed_weights.get(item.source, 1.0))
    reasons: list[str] = []

    age_hours = max(0.0, (now - item.published_dt).total_seconds() / 3600)
    if age_hours <= 12:
        score += 2.0
        reasons.append("fresh <12h")
    elif age_hours <= 30:
        score += 1.2
        reasons.append("fresh <30h")
    elif age_hours <= 72:
        score += 0.4
        reasons.append("recent <72h")

    for keyword in high:
        if keyword and keyword in text:
            score += 1.7
            reasons.append(keyword)
    for keyword in medium:
        if keyword and keyword in text:
            score += 0.8
            reasons.append(keyword)
    for keyword in penalties:
        if keyword and keyword in text:
            score -= 2.0
            reasons.append(f"penalty:{keyword}")

    if item.summary:
        score += 0.3

    item.score = round(score, 2)
    item.reasons = reasons[:8]
    return item


def collect_items(config: dict[str, Any], now: datetime, today_local: str) -> list[NewsItem]:
    site = config.get("site", {}) or {}
    hours_back = float(site.get("hours_back", 36))
    min_dt = now - timedelta(hours=hours_back)
    max_per_source = int(site.get("max_per_source", 4))
    max_items = int(site.get("max_items", 18))
    min_score = float(site.get("min_score", 0.0))
    suppress_seen_days = int(site.get("suppress_seen_days", 14))
    seen_path = Path(site.get("seen_path", "data/seen_urls.json"))
    seen = load_seen(seen_path)
    feed_weights = {str(feed.get("name")): float(feed.get("weight", 1.0)) for feed in config.get("feeds", [])}

    fetched: list[NewsItem] = []
    for feed in config.get("feeds", []):
        fetched.extend(fetch_feed(feed, now=now))
        time.sleep(float(feed.get("polite_delay_seconds", site.get("polite_delay_seconds", 0.2))))

    by_url: set[str] = set()
    by_title: set[str] = set()
    deduped: list[NewsItem] = []
    for item in fetched:
        if item.published_dt < min_dt:
            continue

        seen_date = seen.get(item.canonical_url)
        # Allow same-day reruns to reproduce the same digest; suppress older repeats.
        if seen_date and not seen_date.startswith(today_local):
            seen_dt = parse_datetime(seen_date)
            if seen_dt and seen_dt > now - timedelta(days=suppress_seen_days):
                continue

        title_key = title_fingerprint(item.title)
        if item.canonical_url in by_url or title_key in by_title:
            continue
        by_url.add(item.canonical_url)
        by_title.add(title_key)
        deduped.append(score_item(item, config, feed_weights=feed_weights, now=now))

    deduped.sort(key=lambda item: (item.score, item.published_dt), reverse=True)

    selected: list[NewsItem] = []
    counts: Counter[str] = Counter()
    for item in deduped:
        if len(selected) >= max_items:
            break
        if item.score < min_score and len(selected) >= max(5, max_items // 2):
            continue
        if counts[item.source] >= max_per_source:
            continue
        selected.append(item)
        counts[item.source] += 1

    if len(selected) < min(8, max_items):
        already = {item.canonical_url for item in selected}
        for item in deduped:
            if len(selected) >= min(8, max_items):
                break
            if item.canonical_url not in already:
                selected.append(item)
                already.add(item.canonical_url)

    for item in selected:
        seen[item.canonical_url] = now.isoformat()
    save_seen(seen_path, seen, now=now, keep_days=max(30, suppress_seen_days + 7))
    return selected


def markdown_link(title: str, url: str) -> str:
    safe_title = title.replace("[", "\\[").replace("]", "\\]")
    safe_url = url.replace(")", "%29")
    return f"[{safe_title}]({safe_url})"


def build_prompt(config: dict[str, Any], items: list[NewsItem], digest_date: str) -> str:
    site = config.get("site", {}) or {}
    editor_notes = site.get("editor_notes", "")
    payload = [
        {
            "title": item.title,
            "source": item.source,
            "published": item.published,
            "summary": item.summary,
            "url": item.url,
            "score": item.score,
            "reasons": item.reasons,
        }
        for item in items
    ]
    return textwrap.dedent(
        f"""
        Create a concise daily AI news digest for {digest_date} from the JSON items below.

        Rules:
        - Use only the supplied titles, summaries, sources, dates, and URLs.
        - Do not invent numbers, quotes, benchmarks, product details, or claims that are not in the input.
        - Every story mention must include a Markdown link to the original URL.
        - Prefer what changed, why it matters, and who is affected.
        - Group related items when possible, but keep the digest skimmable.
        - Output Markdown only.

        Desired structure:
        # Daily AI News — {digest_date}
        _Generated by a GitHub Actions GitOps pipeline._

        ## Executive summary
        3-5 bullets.

        ## Top stories
        6-10 bullets. Include source names.

        ## Signals to watch
        3 bullets about patterns across the sources.

        ## All links
        Bullet list of every selected item.

        Additional editor notes: {editor_notes or "None"}

        JSON items:
        {json.dumps(payload, ensure_ascii=False, indent=2)}
        """
    ).strip()


def call_github_models(config: dict[str, Any], prompt: str) -> str:
    site = config.get("site", {}) or {}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_PAT")
    if not token:
        raise RuntimeError("No GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT available for GitHub Models")

    endpoint = os.environ.get("GITHUB_MODELS_ENDPOINT", DEFAULT_MODEL_ENDPOINT)
    model = os.environ.get("GITHUB_MODELS_MODEL", str(site.get("model", "openai/gpt-4o-mini")))
    system_prompt = str(
        site.get(
            "system_prompt",
            "You are a careful technology-news editor. You summarize only what is supported by the supplied source metadata.",
        )
    )
    payload: dict[str, Any] = {
        "model": model,
        "temperature": float(site.get("temperature", 0.2)),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    max_tokens = site.get("max_completion_tokens")
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)

    response = requests.post(
        endpoint,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=int(site.get("model_timeout_seconds", 90)),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub Models API returned {response.status_code}: {response.text[:600]}")

    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected GitHub Models response shape: {data}") from exc


def deterministic_digest(config: dict[str, Any], items: list[NewsItem], digest_date: str, generated_at: datetime) -> str:
    title = (config.get("site", {}) or {}).get("title", "Daily AI News")
    lines = [
        f"# {title} — {digest_date}",
        "",
        f"_Generated at {generated_at.isoformat()} by a GitHub Actions GitOps pipeline._",
        "",
    ]

    if not items:
        lines.extend(
            [
                "## No fresh items found",
                "",
                "The configured feeds returned no items inside the current time window. Check `config/sources.yml`, feed availability, and workflow logs.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(["## Top stories", ""])
    for item in items[:10]:
        published = item.published_dt.strftime("%Y-%m-%d %H:%M UTC")
        reason = f" Score {item.score:.1f}" if item.score else ""
        summary = f" — {item.summary}" if item.summary else ""
        lines.append(f"- {markdown_link(item.title, item.url)} — {item.source}, {published}.{reason}{summary}")

    lines.extend(["", "## Signals to watch", ""])
    source_counts = Counter(item.source for item in items)
    keyword_counts = Counter(reason for item in items for reason in item.reasons if not reason.startswith("fresh"))
    if source_counts:
        lines.append("- Most represented sources: " + ", ".join(f"{name} ({count})" for name, count in source_counts.most_common(4)) + ".")
    if keyword_counts:
        lines.append("- Recurring themes: " + ", ".join(f"{name} ({count})" for name, count in keyword_counts.most_common(8)) + ".")
    lines.append("- Review the raw JSON output in `data/items/` if you want to audit scoring or feed coverage.")

    lines.extend(["", "## All links", ""])
    for item in items:
        lines.append(f"- {markdown_link(item.title, item.url)} — {item.source}")
    lines.append("")
    return "\n".join(lines)


def add_footer(markdown: str, items: list[NewsItem], model_used: bool) -> str:
    footer = [
        "",
        "---",
        "",
        "### Pipeline metadata",
        "",
        f"- Selected items: {len(items)}",
        f"- AI summarization: {'GitHub Models' if model_used else 'deterministic fallback'}",
        "- Source data: `data/items/` in this repository",
    ]
    return markdown.rstrip() + "\n" + "\n".join(footer) + "\n"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def slugify_anchor(text: str) -> str:
    slug = re.sub(r"[^a-z0-9 -]", "", text.lower())
    slug = re.sub(r"\s+", "-", slug).strip("-")
    return slug or "section"


def render_inline_markdown(text: str) -> str:
    """Render the small Markdown subset generated by this script into safe HTML."""
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" rel="noopener noreferrer">\1</a>', escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"_([^_]+)_", r"<em>\1</em>", escaped)
    return escaped


def markdown_to_html(markdown: str, title: str) -> str:
    """Convert generated Markdown to a lightweight standalone HTML page.

    This intentionally avoids adding a Markdown dependency. It supports headings,
    paragraphs, bullet lists, horizontal rules, inline links, inline code, bold,
    and emphasis, which is enough for the generated digests.
    """
    body: list[str] = []
    in_ul = False

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            body.append("</ul>")
            in_ul = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            close_ul()
            continue
        if line == "---":
            close_ul()
            body.append("<hr>")
            continue
        if line.startswith("### "):
            close_ul()
            value = line[4:]
            body.append(f'<h3 id="{slugify_anchor(value)}">{render_inline_markdown(value)}</h3>')
            continue
        if line.startswith("## "):
            close_ul()
            value = line[3:]
            body.append(f'<h2 id="{slugify_anchor(value)}">{render_inline_markdown(value)}</h2>')
            continue
        if line.startswith("# "):
            close_ul()
            value = line[2:]
            body.append(f'<h1>{render_inline_markdown(value)}</h1>')
            continue
        if line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{render_inline_markdown(line[2:])}</li>")
            continue
        close_ul()
        body.append(f"<p>{render_inline_markdown(line)}</p>")
    close_ul()

    safe_title = html.escape(title)
    html_body = "\n".join(body)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.65;
      max-width: 920px;
      margin: 0 auto;
      padding: 2rem 1rem 4rem;
    }}
    h1, h2, h3 {{ line-height: 1.25; }}
    a {{ text-underline-offset: 0.15em; }}
    code {{
      padding: 0.1rem 0.25rem;
      border-radius: 0.25rem;
      background: color-mix(in srgb, CanvasText 10%, transparent);
    }}
    li {{ margin: 0.35rem 0; }}
    hr {{ margin: 2rem 0; }}
    .nav {{ margin-bottom: 2rem; color: color-mix(in srgb, CanvasText 70%, transparent); }}
  </style>
</head>
<body>
<div class="nav"><a href="../index.html">Home</a></div>
{html_body}
</body>
</html>
"""


def write_markdown_outputs(
    config: dict[str, Any],
    digest_markdown: str,
    items: list[NewsItem],
    digest_date: str,
    generated_at: datetime,
) -> dict[str, str]:
    site = config.get("site", {}) or {}
    digest_dir = Path(site.get("digest_dir", "digests"))
    docs_dir = Path(site.get("docs_dir", "docs"))
    data_dir = Path(site.get("data_dir", "data/items"))

    digest_path = digest_dir / f"{digest_date}.md"
    docs_digest_dir = docs_dir / "digests"
    docs_digest_md_path = docs_digest_dir / f"{digest_date}.md"
    docs_digest_html_path = docs_digest_dir / f"{digest_date}.html"
    data_path = data_dir / f"{digest_date}.json"
    latest_data_path = data_dir.parent / "latest.json"

    digest_path.parent.mkdir(parents=True, exist_ok=True)
    docs_digest_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Required so GitHub Pages serves exactly the static files we upload.
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")

    digest_path.write_text(digest_markdown, encoding="utf-8")
    docs_digest_md_path.write_text(digest_markdown, encoding="utf-8")
    docs_digest_html_path.write_text(markdown_to_html(digest_markdown, f"Daily AI News — {digest_date}"), encoding="utf-8")

    payload = {
        "generated_at": generated_at.isoformat(),
        "digest_date": digest_date,
        "items": [asdict(item) for item in items],
    }
    write_json(data_path, payload)
    write_json(latest_data_path, payload)

    index_paths = write_index(
        config,
        docs_dir=docs_dir,
        digest_date=digest_date,
        latest_items=items,
        generated_at=generated_at,
    )

    return {
        "digest": str(digest_path),
        "docs_digest_md": str(docs_digest_md_path),
        "docs_digest_html": str(docs_digest_html_path),
        "docs_index_md": str(index_paths["md"]),
        "docs_index_html": str(index_paths["html"]),
        "data": str(data_path),
        "latest_data": str(latest_data_path),
    }


def write_index(config: dict[str, Any], docs_dir: Path, digest_date: str, latest_items: list[NewsItem], generated_at: datetime) -> dict[str, Path]:
    site = config.get("site", {}) or {}
    title = str(site.get("title", "Daily AI News"))
    digest_files = sorted((docs_dir / "digests").glob("*.md"), reverse=True)

    lines = [
        f"# {title}",
        "",
        f"_Last updated: {generated_at.isoformat()}._",
        "",
        f"[Read the latest digest](digests/{digest_date}.html)",
        "",
    ]

    if latest_items:
        lines.extend(["## Latest top links", ""])
        for item in latest_items[:8]:
            lines.append(f"- {markdown_link(item.title, item.url)} — {item.source}")
        lines.append("")

    lines.extend(["## Recent digests", ""])
    for path in digest_files[:30]:
        label = path.stem
        lines.append(f"- [{label}](digests/{path.with_suffix('.html').name})")

    lines.extend(
        [
            "",
            "## About this site",
            "",
            "This is a static site generated by GitHub Actions. The repository is the durable state: workflows fetch feeds, generate Markdown/JSON/HTML, and commit the result back to Git.",
            "",
        ]
    )

    index_markdown = "\n".join(lines)
    index_md_path = docs_dir / "index.md"
    index_html_path = docs_dir / "index.html"
    index_md_path.write_text(index_markdown, encoding="utf-8")
    index_html_path.write_text(markdown_to_html(index_markdown, title), encoding="utf-8")
    return {"md": index_md_path, "html": index_html_path}


def maybe_create_github_issue(config: dict[str, Any], digest_date: str, digest_markdown: str) -> None:
    if os.environ.get("CREATE_GITHUB_ISSUE", "false").lower() not in {"1", "true", "yes", "on"}:
        return

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        print("WARN: CREATE_GITHUB_ISSUE is true, but token/repository env is missing", file=sys.stderr)
        return

    state_path = Path("data/issues") / f"{digest_date}.json"
    if state_path.exists():
        print(f"Issue state already exists for {digest_date}; skipping issue creation")
        return

    site = config.get("site", {}) or {}
    title = f"{site.get('title', 'Daily AI News')} — {digest_date}"
    url = f"https://api.github.com/repos/{repository}/issues"
    body = digest_markdown
    if len(body) > 60000:
        body = body[:59000] + "\n\n_Trimmed because GitHub issue body length is limited._"

    try:
        response = requests.post(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
            json={"title": title, "body": body, "labels": ["daily-digest", "ai-news"]},
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"WARN: GitHub issue creation failed: {exc}", file=sys.stderr)
        return

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(response.json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    site = config.get("site", {}) or {}

    tz_name = str(site.get("timezone", "UTC"))
    local_tz = ZoneInfo(tz_name)
    now = datetime.now(timezone.utc)
    local_now = now.astimezone(local_tz)
    digest_date = local_now.date().isoformat()

    items = collect_items(config, now=now, today_local=digest_date)
    prompt = build_prompt(config, items=items, digest_date=digest_date)

    model_used = False
    if not args.no_ai and items:
        try:
            digest = call_github_models(config, prompt)
            model_used = True
        except Exception as exc:
            print(f"WARN: AI summarization failed; using deterministic fallback: {exc}", file=sys.stderr)
            digest = deterministic_digest(config, items=items, digest_date=digest_date, generated_at=local_now)
    else:
        digest = deterministic_digest(config, items=items, digest_date=digest_date, generated_at=local_now)

    digest = add_footer(digest, items=items, model_used=model_used)

    if args.dry_run:
        print(digest)
        return 0

    outputs = write_markdown_outputs(config, digest, items=items, digest_date=digest_date, generated_at=local_now)
    maybe_create_github_issue(config, digest_date=digest_date, digest_markdown=digest)

    print("Generated digest:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    print(f"Selected {len(items)} items; AI summarization used: {model_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
