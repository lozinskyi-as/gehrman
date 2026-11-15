# Gehrman

Local CLI assistant that helps a bug bounty researcher choose programs.
Developed as a thesis MVP under a Secure Software Development Life Cycle
(NIST SSDF / SP 800-218A). The research experiment measures how well
layered defenses protect the agent against **indirect prompt injection**.

Gehrman is **passive and fully local**: it only reads public program
catalogs and talks to a local [Ollama](https://ollama.com/) endpoint. It
never scans, probes, or contacts a program's targets, and it stores no
API keys.

## Scope

- Platforms: HackerOne, Bugcrowd, and Intigriti, via [arkadiyt/bounty-targets-data](https://github.com/arkadiyt/bounty-targets-data).
- Single user, local Python CLI. No accounts, no web UI.
- Out of scope: git-history analysis, ML prediction, scraping.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependencies
- A running local [Ollama](https://ollama.com/) with at least one model
  pulled (for example `ollama pull llama3.1`). `gehrman init` lists the
  models it finds on the endpoint so you can pick one.

## Install

```bash
uv sync
uv run gehrman --help
```

The `dev` group (pytest, ruff, bandit, pip-audit, cyclonedx-bom) is opt-in.
Install it only when you need those tools:

```bash
uv sync --group dev
```

## Commands

| Command | Status | Purpose |
| --- | --- | --- |
| `gehrman init` | Milestone 1 | Write researcher profile and pick a local Ollama model |
| `gehrman update` | Milestone 2 | Download catalogs, store snapshots, score |
| `gehrman top` | Milestone 2 | List the highest-scored programs from the local snapshot |
| `gehrman chat` | Later | Talk to the agent (local Ollama) |

### `gehrman init`

Interactive run: Gehrman queries the Ollama endpoint, prints a numbered
list of installed models, and asks you to pick one. The selected model
is used both as the agent model and (by default) as the AI-guard
classifier.

```bash
uv run gehrman init
```

Non-interactive (scripts, tests): pass flags to skip every prompt. The
`--model` flag skips the menu; `--ai-guard-model` overrides the
classifier model if you want it separate from the agent model.

```bash
uv run gehrman init \
  --skill web --skill api \
  --bounty-only \
  --model llama3.1:latest \
  --ai-guard-model llama3.1:latest
```

| Flag | Purpose |
| --- | --- |
| `--skill` | Repeatable. `web` or `api` |
| `--bounty-only` / `--all-programs` | Filter programs that pay bounties |
| `--base-url` | Ollama OpenAI-compatible endpoint (default `http://localhost:11434/v1`) |
| `--model` | Ollama model identifier; skips the interactive menu |
| `--ai-guard-model` | Classifier model for prompt-injection guards; defaults to `--model` |

If `config.toml` already exists, `init` asks before overwriting. If
Ollama is unreachable or has no models pulled, the interactive menu
falls back to a manual model-name prompt. The written config lives at
`~/.config/gehrman/config.toml` and never contains secrets.

### `gehrman update`

Downloads the three public catalogs, validates them with pydantic,
writes raw snapshots to `~/.local/share/gehrman/snapshots/<platform>/`
(mode `0600`), and populates `~/.local/share/gehrman/gehrman.db` with
normalized, scored program rows. The command only talks to the static
CDN; it never contacts a program's targets.

```bash
uv run gehrman update                    # all three platforms
uv run gehrman update --platform hackerone
uv run gehrman update --dry-run          # fetch and parse, write nothing
```

Scoring is deterministic and is a pure function of `(program, profile)`;
re-running `update` reproduces the same scores for the same snapshot.

### `gehrman top`

Reads the SQLite store opened read-only and prints the highest-scored
programs. Catalog strings (program name, URL, asset types) are escaped
before rendering so a malicious name cannot inject Rich markup into your
terminal.

```bash
uv run gehrman top                                   # top 10 overall
uv run gehrman top --limit 25
uv run gehrman top --platform bugcrowd
uv run gehrman top --min-bounty 5000
```

| Flag | Purpose |
| --- | --- |
| `--limit` / `-n` | Rows to show (1–100, default 10) |
| `--platform` | Restrict to `hackerone`, `bugcrowd`, or `intigriti` |
| `--min-bounty` | Only programs with `bounty_max` ≥ this amount |

Override locations in tests or unusual setups:

- `GEHRMAN_CONFIG_DIR` — directory for `config.toml` (default `~/.config/gehrman`)
- `GEHRMAN_DATA_DIR` — directory for the database and snapshots (default `~/.local/share/gehrman`)

## Security

See [SECURITY.md](SECURITY.md) for trust boundaries.

Defense switches live under `[defenses]` in `config.toml` and default to
on. They are implemented in a later milestone; the flags exist now so
experiments can turn controls on and off without a schema change.

## Development

Install the dev group first (`uv sync --group dev`), then:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run bandit -r src --severity-level high
uv run pip-audit
```

CI on every push runs ruff, pytest, bandit (fail on high), pip-audit,
gitleaks, and publishes a CycloneDX SBOM.
