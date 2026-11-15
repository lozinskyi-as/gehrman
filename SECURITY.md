# Security policy

Gehrman is a **local, single-user CLI**. It does not scan, probe, or send
requests to any bug bounty program's targets. It only downloads three public
JSON catalogs (HackerOne, Bugcrowd, and Intigriti) and talks to a local
Ollama endpoint on the user's machine.

This document describes trust boundaries for the thesis experiment
(indirect prompt injection against an LLM agent) and for everyday use.

## Trust boundaries

| Source | Trust | Notes |
| --- | --- | --- |
| Structured catalog fields (handles, URLs, booleans, numeric metrics, asset types, bounty amounts) | **Conditionally trusted** | Taken from the public [arkadiyt/bounty-targets-data](https://github.com/arkadiyt/bounty-targets-data) snapshot. Validated against pydantic schemas before storage. Still treated as untrusted for *rendering* (terminal markup) and never executed. |
| Free-text catalog fields (`instruction`, `description`) | **Untrusted** | May contain injected instructions or false facts. Must be sanitized, marked, and optionally classified before they enter the model context. They must never influence scores. |
| Model output (chat answers, tool-argument proposals, AI-guard JSON) | **Untrusted until checked** | Subject to `scope_check` and `number_check` (deterministic) and optional `ai_guard_output`. Displayed only after escaping. |
| Researcher profile and defense switches in `config.toml` | **Trusted** | Written by `gehrman init` on the local machine. No secrets. |
| Local Ollama endpoint (`http://localhost:11434/v1`) | **Trusted** | Loopback-only. Gehrman stores no credentials; the model runs on the user's machine. |
| Tool implementations (search, get, explain, check_scope) | **Trusted code** | Read-only, pydantic-validated arguments, no network, no shell, no arbitrary file access. The agent opens SQLite with `mode=ro`. |

## What Gehrman will not do

- Scan, crawl, or send traffic to program targets.
- Run `eval`, `exec`, or `subprocess` on the agent path.
- Let a tool write data, read arbitrary files, or make network requests.
- Render raw program text as Rich markup.

## Secrets

Gehrman uses a local Ollama endpoint with no authentication, so it stores
no API keys. `~/.config/gehrman/config.toml` is still written through a
secret-key guard: any key named `api_key`, `apikey`, `secret`, `token`,
or `password` is rejected, so a future provider that needs auth cannot
silently land a credential on disk here.

## Reporting a vulnerability

This is a thesis MVP, not a production service. Please open a private report
with the author (repository owner) rather than a public issue if you believe
you have found a vulnerability in Gehrman itself.
