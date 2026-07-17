# QMCPy.org Transition Checklist

_revised 2026-07-16_

## Goals

- Retire expensive WordPress Business hosting
- Preserve ownership/control of `qmcpy.org`
- Transition hosting to GitHub Pages
- Preserve important URLs and discoverability
- Keep transition low-stress during MCQMC 2026 preparation
- Allow gradual improvement of homepage/navigation later

---

# Current Status

## Domain & Billing

- [x] Domain registration active through 2028-08-31
- [x] WordPress Business renewal canceled
- [x] WordPress Business hosting end date recorded as 2026-07-16
- [x] Current annual domain cost approximately $19
- [x] Avoided upcoming ~$300 hosting renewal

## Current Infrastructure

- [x] Existing GitHub Pages site exists:
  - https://qmcsoftware.github.io/QMCSoftware/
- [x] Blog posts already migrated to GitHub
- [x] WordPress.com nameservers remain authoritative for DNS
- [x] `qmcpy.org` web records point to GitHub Pages
- [x] `www.qmcpy.org` points to `qmcsoftware.github.io`

Current-state note:

- On 2026-07-16, `https://qmcpy.org/` was served by GitHub Pages as a
  398-byte temporary "under construction" page.

---

# Current DNS Notes

## Current Important Records

### Web Hosting

- `A @` points to the four GitHub Pages IPv4 addresses:
  `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, and
  `185.199.111.153`.
- `CNAME www -> qmcsoftware.github.io`.
- No apex AAAA record was returned during the 2026-07-16 check.

### Email

- `MX @` currently handled by WordPress.com email forwarding

### Wildcard

- `CNAME * -> qmcpy.org`

### Other Records

- SPF: `v=spf1 include:_spf.wpcloud.com ~all`.
- DMARC: `v=DMARC1;p=none;`.
- DKIM was previously observed in WordPress DNS settings, but selector names
  were not recorded and were not independently enumerated in this check.
- DNSSEC is unsigned and no DS record is published.

---

# Immediate Priority (Before MCQMC 2026)

## Administrative

- [x] Create migration feature branch
- [x] Add this checklist to repo
- [x] Decide location of checklist within repo
- [x] Identify collaborators who may help with migration

Notes:

- Migration branch: `migrate_qmcpy`
- Checklist location: repository root, `MIGRATE_qmcpy_checklist.md`
- Project owner, sole collaborator, and migration decision authority: Kang
  Jiangrui.
- Codex may gather evidence and implement approved repository changes; content
  retention and external-link decisions are confirmed by Kang Jiangrui.
- Migration source of truth: `MIGRATE_qmcpy_checklist.md` for status and
  sequencing, and `MIGRATE_qmcpy_inventory.md` for content evidence.
- Before-MCQMC scope is limited to governance, backup preservation, content
  classification, and baseline verification. Homepage redesign, navigation
  changes, URL mapping, redirects, and hosting changes remain Post-MCQMC work.

## Backups / Preservation

- [x] Export/download WordPress content backup
- [x] Export/download WordPress media/uploads
- [x] Preserve homepage text/content
- [x] Preserve important graphics/assets
- [x] Preserve navigation/menu structure

Checked backup files:

- `/Users/kangjiangrui/Downloads/qmcpy.WordPress.2026-05-26.xml`
- `/Users/kangjiangrui/Downloads/jetpack-backup-qmcpy-org-2026-05-25-20-48-57.tar.gz`

Backup notes:

- WordPress XML parses successfully and contains pages, posts, attachments, and navigation menu items.
- Jetpack backup gzip check passes and includes `wp-content/uploads/`, SQL tables, plugins, themes, and `wp-config.php`.
- WordPress homepage is exported as page ID 5, `Blog`, at `https://qmcpy.org/`.
- Backup sizes, modification times, SHA-256 digests, archive counts, and the
  minimum recovery procedure are recorded in `MIGRATE_qmcpy_inventory.md`.
- Backup validation did not extract the archive or read configuration or
  database contents.

## Content Inventory

- [x] List important existing pages
- [x] Identify pages already migrated
- [x] Identify pages that can be retired
- [x] Identify important external URLs/backlinks

Inventory draft:

- `MIGRATE_qmcpy_inventory.md`

Current must-preserve scope:

- Home
- Publications
- Blogs
- GitHub
- Docs
- PyPI

Collaborator decisions confirmed by Kang Jiangrui on 2026-07-16:

- Preserve contributor content; evaluate the existing `community.md` target
  during Post-MCQMC URL mapping.
- Retire the standalone Donation, Videos, and Dev Tools pages.
- Retire unpublished WordPress drafts.
- Retire old news, event, and announcement posts without a confirmed current
  target.

Inventory notes:

- Already migrated content and confirmed retirement decisions are listed in
  `MIGRATE_qmcpy_inventory.md`.
- External URL/backlink importance has been confirmed for the current
  inventory. Exact old-to-new mappings and redirect implementation remain
  Post-MCQMC work.

## Before-MCQMC Evidence Hardening

- [x] 1. Confirm the owner, scope boundary, source-of-truth files, and decision
  authority
- [x] 2. Record backup metadata, SHA-256 digests, structural integrity, and a
  minimum recovery procedure
- [x] 3. Reconcile the WordPress export, inventory rows, migrated blog targets,
  and source URL metadata
- [x] 4. Review retirement rationales and exceptions with the project owner
- [x] 5. Review backlink importance and preservation requirements with the
  project owner
- [x] 6. Refresh live endpoint, infrastructure, and read-only DNS evidence
- [x] 7. Re-run the documentation build and classify all warnings
- [x] 8. Write the final Before-MCQMC completion report and gate verdict

Evidence-hardening notes:

- Items 1-3 were completed on 2026-07-16.
- Items 4-5 were completed on 2026-07-16 using the owner-approved conservative
  retirement policy and tiered backlink-preservation policy recorded in
  `MIGRATE_qmcpy_inventory.md`.
- Item 6 was completed on 2026-07-16 using public HTTP, DNS, and WHOIS checks.
- Item 7 was completed on 2026-07-16. `make copydocs` was unavailable because
  `pandoc` is missing, but the direct MkDocs build succeeded and all warnings
  were classified as non-blocking baseline or local-worktree issues.
- Item 8 is recorded in `MIGRATE_qmcpy_before_mcqmc_report.md` with a `GO`
  verdict for completion of the Before-MCQMC scope.
- This evidence work does not authorize any Post-MCQMC URL mapping, redirect,
  DNS, GitHub Pages, or WordPress configuration change.

## GitHub Pages

- [x] Verify existing GitHub Pages deployment is stable
- [x] Verify installation instructions
- [x] Verify documentation links
- [x] Verify PyPI links
- [x] Verify GitHub links

Verification notes:

- Checked on 2026-07-16: the GitHub Pages homepage, `CONTRIBUTING/`, the
  `qmcpy_intro` demo, one migrated blog, PyPI, and GitHub returned HTTP 200.
- The rendered GitHub Pages homepage contains the `pip install qmcpy`
  installation command.
- GitHub Pages `/blogs/` and `/publications/` indexes returned HTTP 404; one
  directly addressed migrated blog returned HTTP 200.
- `https://qmcpy.org/` returned HTTP 200 from GitHub Pages, while its
  `/publications/` and old "Why Add Q to MC?" paths returned HTTP 404.
- Documentation build evidence is maintained separately under evidence item 7.

---

# Post-MCQMC Transition Work

## Homepage

- [x] Decide minimum viable homepage
- [x] Improve landing page professionalism
- [x] Add/verify:
  - [x] Installation link
  - [x] Documentation link
  - [x] GitHub link
  - [x] PyPI link
  - [x] Citation information
  - [x] Publications/references
  - [x] Contributors/collaborators
  - [x] QMC software ecosystem table

## Navigation

- [x] Review navbar structure
- [x] Simplify navigation if needed
- [x] Remove obsolete WordPress-era structure

## URL Preservation

- [x] Inventory important old WordPress URLs
- [x] Map old URLs to new locations
- [x] Decide which redirects are worth preserving

## Implementation Evidence

Completed on `migrate_qmcpy` on 2026-07-17:

- Reworked `README.md` into the minimum viable homepage with direct entry
  points for installation, documentation, demos, blogs, publications,
  citation, community, GitHub, PyPI, and the QMC software ecosystem.
- Added `docs/blogs/index.md` and `docs/publications.md` as stable public index
  targets.
- Kept the preserved blog archive at the 18 already migrated posts after the
  2026-07-17 owner instruction to remove the five extra post candidates from
  the current scope.
- Reorganized `mkdocs.yml` into Getting Started, API Reference, Demos, Blogs,
  Research and Community, and For Developers sections.
- Recorded 20 required old paths in `data/redirects.yml`. The generated pages
  preserve two contributor URLs and the old paths for all 18 migrated posts.
- `scripts/make_redirect_pages.py` creates path-exact redirect pages without
  depending on a new MkDocs plugin. It validates paths, targets, duplicates,
  and ownership before writing files.
- `conda run -n qmcpy make copydocs` completed after adding Pandoc to the local
  `qmcpy` environment. `conda run -n qmcpy make uml` also completed after
  adding Graphviz.
- `conda run -n qmcpy python -m mkdocs build --clean` completed in about 53
  seconds with no warnings or unresolved relative links.
- Generated-output validation found all 20 redirect pages and all 20 targets,
  with no missing files or redirect markup.
- Browser validation confirmed 18 indexed blog posts, the Publications page,
  the high-priority "Why Add Q to MC?" redirect, desktop layout, and a 390 px
  mobile layout without horizontal overflow or console errors.

These checks validate the branch implementation only. Live `qmcpy.org`
verification remains part of Final Hosting Migration after publication.

---

# Final Hosting Migration

## Preferred Long-Term Architecture

### Desired Outcome

Users visit:

- `https://qmcpy.org`

Browser remains on:

- `qmcpy.org`

Content served from:

- GitHub Pages

### NOT preferred long-term

Simple forwarding:

- `qmcpy.org -> qmcsoftware.github.io/QMCSoftware/`

because browser URL changes away from `qmcpy.org`

---

# GitHub Pages Custom Domain Setup

## GitHub Side

- [ ] Configure custom domain in GitHub Pages settings
- [ ] Add `qmcpy.org` as custom domain
- [ ] Enable HTTPS

## DNS Side (WordPress DNS)

### Replace WordPress Hosting Records

Current:

- `A @ -> WordPress.com`

Future:

- GitHub Pages A records

Expected GitHub records:

```text
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```
