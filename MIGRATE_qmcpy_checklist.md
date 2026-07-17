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

- [ ] Decide minimum viable homepage
- [ ] Improve landing page professionalism
- [ ] Add/verify:
  - [ ] Installation link
  - [ ] Documentation link
  - [ ] GitHub link
  - [ ] PyPI link
  - [ ] Citation information
  - [ ] Publications/references
  - [ ] Contributors/collaborators
  - [ ] QMC software ecosystem table

## Navigation

- [ ] Review navbar structure
- [ ] Simplify navigation if needed
- [ ] Remove obsolete WordPress-era structure

## URL Preservation

- [ ] Inventory important old WordPress URLs
- [ ] Map old URLs to new locations
- [ ] Decide which redirects are worth preserving

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
