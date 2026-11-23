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
| Free-text catalog fields (`policy`, stored as `policy_excerpt`; also program `name`) | **Untrusted** | `policy_excerpt` is where indirect prompt injection is planted (HackerOne policy, truncated). `name` is catalog text too. Neither may seed `scope_check` or `number_check`, and neither may influence scores. |
| Model output (chat answers, tool-argument proposals) | **Untrusted until checked** | Subject to `scope_check` and `number_check`. Displayed only after escaping. Tool-call arguments do not seed those checks: the model writes them. |
| Researcher profile and defense switches in `config.toml` | **Trusted** | Written by `gehrman init` on the local machine. No secrets. |
| Local Ollama endpoint (`http://localhost:11434/v1`) | **Trusted** | Loopback-only. Gehrman stores no credentials; the model runs on the user's machine. |
| Tool implementations (search, get, explain, check_scope) | **Trusted code** | Read-only, pydantic-validated arguments, no network, no shell, no arbitrary file access. The agent opens SQLite with `mode=ro`. |

## Allow-lists

`scope_check` and `number_check` allow a host or a number only when it was seeded this turn. Seeding uses trusted structured tool-result fields (`url`, `handle`, numeric fields such as `bounty_min`, `bounty_max`, `score`, and counts, plus `asset_types`) and the user's latest message. The user is trusted. Tool-call arguments are not, because the model writes them and can be steered.

`explain` returns `policy_excerpt`. That string is untrusted free text, and it is the field the evaluation harness writes attack payloads into. Harvesting every hostname and number out of it made a canary "allowed", so the deterministic checks passed an answer that repeated the payload. The allow-list therefore ignores `policy_excerpt`. Program `name` is catalog text as well (a title can contain a hostname or a fake figure) and does not contribute. A new free-text result key stays ignored until it is added to `TRUSTED_ALLOWLIST_KEYS`.

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


## Baseline input clean-up and scope matching

`sanitize` is a baseline layer, not a defense against well-formed
natural-language instructions. It applies NFKC normalization, drops
C0/C1 control characters (newline and tab kept), zero-width, bidi
control and Unicode tag characters, replaces text that imitates the
`<<DATA ...>>` / `<<END>>` fence, and truncates to 4000 characters.

`check_scope` matches hosts, not substrings. The candidate and each
stored scope identifier are reduced to an IDNA hostname (no scheme,
path, port or userinfo). A plain entry matches only that host; a
`*.example.com` entry matches subdomains on a label boundary but not
the apex. Out-of-scope entries win over in-scope ones. Mobile bundle
IDs are ignored.
