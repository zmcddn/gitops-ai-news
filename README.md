# GitOps Daily AI News Aggregator

A small, serverless daily AI news pipeline that runs entirely in GitHub:

1. GitHub Actions runs on a schedule.
2. Python fetches RSS/Atom feeds.
3. The workflow calls GitHub Models using the automatic `GITHUB_TOKEN`.
4. The script writes Markdown and JSON into the repository.
5. GitHub Actions commits the generated files back to `main`.
6. GitHub Pages can serve `/docs` as a static daily news site.

No database, server, newsletter provider, or separate LLM key is required for the default GitHub Actions setup.

## What gets generated

```text
digests/YYYY-MM-DD.md          # human-readable digest in the repo
docs/index.md                  # GitHub Pages homepage
docs/digests/YYYY-MM-DD.md     # GitHub Pages copy of the digest
data/items/YYYY-MM-DD.json     # selected source items and scores
data/seen_urls.json            # lightweight de-duplication state
```

Optional issue delivery is included: set repository variable `CREATE_GITHUB_ISSUE=true` and the workflow will also create one GitHub Issue per daily digest.

## Quick start

1. Create a new GitHub repository. A public repository is recommended for the free GitHub-hosted runner setup.
2. Copy these files into the repository.
3. Commit and push to the default branch.
4. Open **Actions** and run **Daily AI News GitOps** manually once.
5. Open **Settings → Pages** and set **Build and deployment → Source** to **GitHub Actions**.
6. Check `digests/`, `data/items/`, and the Pages deployment after the workflow completes.

The workflow is scheduled for 5:17 AM in `America/Vancouver`. Change this in `.github/workflows/daily-ai-news.yml`.

## GitHub Models setup

Inside GitHub Actions, no extra model secret is needed. The workflow grants:

```yaml
permissions:
  contents: write
  models: read
  issues: write
  pages: write
  id-token: write
```

The script calls `https://models.github.ai/inference/chat/completions` with `${{ secrets.GITHUB_TOKEN }}`.

For local AI testing, create a GitHub personal access token with the `models` scope and run:

```bash
export GITHUB_PAT="your-token"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ai_news_aggregator/main.py --config config/sources.yml --dry-run
```

For local non-AI testing:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ai_news_aggregator/main.py --config config/sources.yml --no-ai --dry-run
```

## Customize sources

Edit `config/sources.yml`.

Each feed supports:

```yaml
- name: "Example AI Blog"
  url: "https://example.com/feed.xml"
  weight: 1.5
  entry_limit: 40
```

Scoring is intentionally simple and auditable:

- source weight
- recency
- keyword matches
- de-duplication by canonical URL and normalized title
- optional suppression of already-seen URLs for a configurable number of days

## Useful repository variables

Set these in **Settings → Secrets and variables → Actions → Variables**:

| Variable | Default | Purpose |
|---|---:|---|
| `CREATE_GITHUB_ISSUE` | `false` | Create one GitHub Issue per digest so watchers can get notifications. |

Set these only if you need to override the defaults:

| Environment variable | Purpose |
|---|---|
| `GITHUB_MODELS_MODEL` | Override the model in `config/sources.yml`. |
| `GITHUB_MODELS_ENDPOINT` | Override the GitHub Models endpoint. |

## GitOps notes

This is GitOps in the practical sense: the repository contains the workflow, source configuration, generated artifacts, and state. Changes to sources or scoring are pull requests. Daily output is committed history. Rollback is a git revert.

The design works best for a public static digest. Do not commit private feeds, paid-content excerpts, API keys, cookies, or personal emails to the repository.

## Limits to expect

- GitHub scheduled workflows are not real-time cron. Runs can be delayed and should be scheduled away from the top of the hour.
- Public-repository scheduled workflows can be disabled after long periods with no repository activity. Daily commits usually keep the repo active.
- GitHub Models free usage is rate-limited. This project uses one model call per run by default.
- Some websites block bots or do not provide RSS. Prefer official RSS/Atom feeds.
- GitHub Pages is static hosting; it will not run a backend or database. The workflow deploys Pages via `actions/upload-pages-artifact` and `actions/deploy-pages`, rather than relying on a separate branch-build trigger.

## Project files

```text
.github/workflows/daily-ai-news.yml      # daily scheduled GitHub Actions workflow
.github/prompts/daily_digest.prompt.yml  # reusable prompt for GitHub Models UI
config/sources.yml                       # feed list, scoring, model settings
src/ai_news_aggregator/main.py           # collector, scorer, summarizer, writer
requirements.txt                         # Python dependencies
docs/                                    # GitHub Pages output
digests/                                 # generated Markdown archive
data/                                    # generated JSON/state
```
