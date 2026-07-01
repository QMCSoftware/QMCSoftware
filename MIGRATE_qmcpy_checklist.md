# QMCPy.org Transition Checklist

_revised 2026-05-26_

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

- [x] Domain registration active through 2028-07-31
- [x] WordPress Business renewal canceled
- [x] WordPress hosting remains active until 2026-07-16
- [x] Current annual domain cost approximately $19
- [x] Avoided upcoming ~$300 hosting renewal

## Current Infrastructure

- [x] Existing GitHub Pages site exists:
  - https://qmcsoftware.github.io/QMCSoftware/
- [x] Blog posts already migrated to GitHub
- [x] DNS currently managed by WordPress.com
- [x] Domain forwarding feature available
- [x] DNS records editable

---

# Current DNS Notes

## Current Important Records

### Web Hosting

- `A @` currently handled by WordPress.com

### Email

- `MX @` currently handled by WordPress.com email forwarding

### Wildcard

- `CNAME * -> qmcpy.org`

### Other Records

- DKIM records present
- SPF record present
- DMARC record present
- DNSSEC options available
- Domain security settings available

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
- Collaborators for content review: Fred J. Hickernell, Sou-Cheng Choi, Aleksei Sorokin

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

## Content Inventory

- [x] List important existing pages
- [x] Identify pages already migrated
- [ ] Identify pages that can be retired
- [ ] Identify important external URLs/backlinks

Inventory draft:

- `MIGRATE_qmcpy_inventory.md`

Current must-preserve scope:

- Home
- Publications
- Blogs
- GitHub
- Docs
- PyPI

Items outside this must-preserve list remain pending collaborator review.

Inventory notes:

- Already migrated candidates are listed in `MIGRATE_qmcpy_inventory.md`.
- Retirement candidates and external URL/backlink candidates are listed for collaborator review, but are not finalized.
- External URL/backlink candidates include preliminary importance estimates; collaborator confirmation is still needed before marking that checklist item complete.

## GitHub Pages

- [x] Verify existing GitHub Pages deployment is stable
- [x] Verify installation instructions
- [x] Verify documentation links
- [x] Verify PyPI links
- [x] Verify GitHub links

Verification notes:

- `https://qmcsoftware.github.io/QMCSoftware/` returned HTTP 200.
- `https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/` returned HTTP 200.
- `https://qmcsoftware.github.io/QMCSoftware/demos/qmcpy_intro` redirects to the trailing-slash URL and then returns HTTP 200.
- `https://pypi.org/project/qmcpy/` returned HTTP 200.
- `https://github.com/QMCSoftware/QMCSoftware` returned HTTP 200.
- `conda run -n qmcpy python -m mkdocs build` completed successfully. Existing MkDocs warnings remain and are recorded in `MIGRATE_qmcpy_inventory.md`.

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

Notes:

- First independent homepage pass added public links for documentation, GitHub,
  PyPI, migrated blogs, the QMC software ecosystem table, community, and
  citation information.
- Follow-up homepage pass tightened the entry points into a `Start Here`
  section for users, developers, project background, and citation.
- `https://qmcpy.org/publications/` has been migrated to `docs/publications.md`
  and added to the published navigation.

## Navigation

- [x] Review navbar structure
- [x] Simplify navigation if needed
- [ ] Remove obsolete WordPress-era structure

Notes:

- Added a visible migrated blog index to the Blogs navigation.
- Added the migrated Publications page to Community Resources.
- Completed a conservative navigation label/group cleanup: the homepage label
  is `Home`, and Blogs now separates notebook/demo posts from migrated
  WordPress posts without deleting pages or changing paths.
- Obsolete WordPress-era structure has not been removed; any retirement or
  deletion decision remains collaborator-dependent.

## URL Preservation

- [x] Inventory important old WordPress URLs
- [x] Map old URLs to new locations
- [ ] Decide which redirects are worth preserving

Notes:

- Added exact source URL metadata to the four already-migrated blog posts that
  were listed as metadata gaps in the inventory.
- Mapped `https://qmcpy.org/publications/` to `publications.md`.
- Added a `Clear Old URL Mappings` inventory section for Home, Publications,
  migrated blog posts, and preserved GitHub/Docs/PyPI navigation targets.
- Mapping completion is limited to entries with clear targets. Redirect scope,
  redirect priority, and collaborator-dependent old content decisions remain
  open.

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
