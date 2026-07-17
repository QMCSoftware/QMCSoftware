# QMCPy.org Before-MCQMC Completion Report

_Completed 2026-07-16_

## Verdict

**GO** for completion of the Before-MCQMC preservation and evidence scope.

This verdict means that governance, backups, inventory, classification, public
baseline checks, and documentation validation are sufficient to close the
Before-MCQMC stage. It does not authorize Post-MCQMC URL mapping, redirects,
GitHub Pages configuration, DNS changes, WordPress changes, or final hosting
migration.

## Scope and Authority

- Project owner, sole collaborator, and migration decision authority: Kang
  Jiangrui.
- Status and sequence source of truth: `MIGRATE_qmcpy_checklist.md`.
- Content and evidence source of truth: `MIGRATE_qmcpy_inventory.md`.
- No external hosting, DNS, registrar, GitHub Pages, or WordPress setting was
  changed during this work.

## Gate Summary

| Gate | Result | Evidence summary |
|---|---|---|
| Governance and scope | Pass | One owner and decision authority; Before/Post boundary recorded |
| Backup existence and integrity | Pass | XML and Jetpack backups exist; SHA-256, syntax, gzip, and archive structure checks passed |
| Inventory completeness | Pass | All 50 unique exported page/post URLs match 50 inventory rows exactly |
| Migrated-blog evidence | Pass | All 18 targets exist and have exact Source WordPress URL metadata |
| Retirement classification | Pass | Owner approved conservative policy 4A |
| External backlink requirements | Pass | Owner approved tiered policy 5A |
| Public infrastructure baseline | Pass with expected Post gaps | Core GitHub Pages, GitHub, PyPI, docs, demo, and a direct blog URL return HTTP 200 |
| Documentation build | Pass with environment limitation | Direct MkDocs build succeeded; full `make copydocs` requires `pandoc` |

## State Separation

These states are intentionally reported separately:

- **Files exist:** Pass. Both backups, both migration source-of-truth documents,
  all 18 migrated blog targets, and this report exist.
- **Static validity:** Pass. XML and gzip checks passed, inventory reconciliation
  found no URL gaps, `git diff --check` passed, and MkDocs built successfully.
- **Discoverability:** Partial by design. Direct migrated blog URLs work, while
  the GitHub Pages `/blogs/` and `/publications/` indexes are not yet published.
- **Actual invocation:** Pass with limitation. Direct MkDocs invocation passed;
  the broader `make copydocs` workflow stopped because `pandoc` is absent.

## Confirmed Preservation Decisions

- Preserve Home, Publications, Blogs, GitHub, Docs, PyPI, contributor content,
  all 18 migrated blogs, and the five additional published posts listed in the
  inventory.
- Retire Donation, Videos, Dev Tools, the three unpublished drafts, and
  unmatched old news, event, announcement, poster, and talk posts.
- High-importance backlinks require main-domain preservation and preservation
  of specifically cited old paths. Medium references require main-domain
  preservation and path preservation only for explicit deep links. Low and
  low-value SEO references require no migration action.
- The Illinois Tech Elevate reference requires preservation of the old "Why Add
  Q to MC?" path during Post-MCQMC URL mapping.
- Read-only checks confirmed 9 of 13 recorded external sources at HTTP 200.
  ResearchGate, FNAL, LinkedIn, and the IIT Journal source were inconclusive due
  to access controls, unsupported HEAD requests, or timeout; none was treated
  as confirmed removal.

## Current External Baseline

- `https://qmcpy.org/` is served by GitHub Pages as a temporary 398-byte "under
  construction" homepage.
- The apex uses the four GitHub Pages IPv4 addresses and `www` points to
  `qmcsoftware.github.io`.
- WordPress nameservers, email forwarding, SPF, DMARC, and the wildcard record
  remain in place. DNSSEC is unsigned.
- GitHub Pages `/blogs/`, `/publications/`, and important old-domain deep links
  remain Post-MCQMC gaps.

## Non-Blocking Limitations

- `pandoc` is unavailable, so `make copydocs` cannot currently finish.
- Ten existing API documentation links reference UML SVG files that are not in
  the documentation tree.
- Local untracked `AGENTS.md` files create two navigation information messages
  during local builds but are not part of the branch.
- DKIM selectors were not recorded, so their records were not guessed or
  independently enumerated.
- Four external backlink sources could not be automatically revalidated; their
  preservation classification remains based on the recorded evidence and owner
  decision.

## Next Stage Boundary

Post-MCQMC work may begin from this evidence baseline. It must separately plan
and review homepage changes, navigation, Publications and Blogs indexes,
old-to-new URL targets, redirect implementation, and final hosting actions.
