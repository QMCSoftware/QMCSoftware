# AGENTS.md

## Project purpose

This repository maintains the QMCPy package and the GitHub Pages site at
`https://qmcsoftware.github.io/QMCSoftware/`.

The active migration purpose is to move the useful public-facing content from
`qmcpy.org` into this repository so the project can eventually serve the
`qmcpy.org` domain from GitHub Pages.

## Current migration priority

Prioritize work that one developer can do independently after MCQMC:

- Improve the minimum viable homepage using already-confirmed repository
  content.
- Improve navigation clarity without removing pages or changing site ownership
  assumptions.
- Map already-inventoried WordPress URLs to existing or newly proposed GitHub
  Pages targets.
- Add missing migration documentation, source URL metadata, and redirect notes
  when the source is already recorded in `MIGRATE_qmcpy_inventory.md`.
- Fix mechanical documentation issues that do not change project policy, such
  as malformed Markdown or stale internal links.

Put collaborator-dependent decisions aside unless the user explicitly asks for
them. This includes retiring WordPress pages, deciding whether Donation, Videos,
Dev Tools, or draft posts should survive, and approving external backlink
importance.

## Migration source of truth

- Use `MIGRATE_qmcpy_checklist.md` for status and sequencing.
- Use `MIGRATE_qmcpy_inventory.md` for old WordPress URLs, known migration
  targets, and unresolved content decisions.
- If a task changes migration status, update the checklist or inventory in the
  same change.
- Treat the inventory as a working inventory, not a final redirect plan.

## Repository expectations

- Inspect before editing. Run `git status --short --branch` before making
  changes.
- Preserve mathematical meaning and public APIs. Do not change algorithms,
  estimators, sampling rules, benchmark definitions, or theoretical assumptions
  as part of documentation migration work.
- Keep changes small and reviewable.
- Do not commit, push, open pull requests, create branches, or modify remotes
  unless the user explicitly asks.
- Do not edit generated `site/` output directly.
- Do not read or print local backup files, credentials, `.env` files, tokens,
  or WordPress configuration unless the user explicitly asks and the task
  requires it.

## Hosting and DNS boundary

Do not change GitHub Pages settings, DNS records, WordPress settings, domain
forwarding, or HTTPS/custom-domain configuration unless the user explicitly asks.
Documentation can prepare the plan, but the actual hosting migration is a
stateful external action.

## Documentation conventions

- Prefer existing Markdown and MkDocs conventions.
- Preserve relative links and image paths when moving content.
- Keep canonical documentation links pointed at
  `https://qmcsoftware.github.io/QMCSoftware/` unless the task is specifically
  about old-domain preservation.
- Keep `qmcpy.org` URLs when they are source metadata, redirect inventory
  entries, or intentionally preserved old URLs.
- If adding a new documentation page, add it to `mkdocs.yml` only when it should
  be part of the published navigation.

## Verification

For documentation-only changes, run the smallest relevant checks:

- `git diff --check`
- `conda run -n qmcpy python -m mkdocs build`

If the `qmcpy` conda environment is unavailable, try `python -m mkdocs build`
and report the environment limitation if it fails. Do not install new
dependencies unless the user asks.

After each substantial site or documentation change, also build the site for
local preview and provide a localhost URL that can be opened in the Codex
in-app browser. Prefer the project workflow (`make copydocs` followed by the
MkDocs build) when the documented dependencies are available. If part of that
workflow is unavailable, report the limitation, still build the closest valid
local `site/` preview, start a local server bound to `127.0.0.1`, and include
the preview URL in the handoff.

For code changes, run the narrowest relevant test first, then broaden only when
the change affects shared behavior.

## Final handoff

Summarize:

1. Files changed.
2. Commands run.
3. Build, test, or validation result.
4. Remaining collaborator-dependent decisions, if any.
5. Current branch when relevant.
