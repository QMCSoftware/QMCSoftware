# QMCPy.org Transition Checklist

_revised 2026-05-24_

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

- [ ] Create migration feature branch
- [ ] Add this checklist to repo
- [ ] Decide location of checklist within repo
- [ ] Identify collaborators who may help with migration

## Backups / Preservation

- [ ] Export/download WordPress content backup
- [ ] Export/download WordPress media/uploads
- [ ] Preserve homepage text/content
- [ ] Preserve important graphics/assets
- [ ] Preserve navigation/menu structure

## Content Inventory

- [ ] List important existing pages
- [ ] Identify pages already migrated
- [ ] Identify pages that can be retired
- [ ] Identify important external URLs/backlinks

## GitHub Pages

- [ ] Verify existing GitHub Pages deployment is stable
- [ ] Verify installation instructions
- [ ] Verify documentation links
- [ ] Verify PyPI links
- [ ] Verify GitHub links

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