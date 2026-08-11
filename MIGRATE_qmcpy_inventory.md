# QMCPy.org Migration Inventory

_Generated from `/Users/kangjiangrui/Downloads/qmcpy.WordPress.2026-05-26.xml` on 2026-05-26._

This file supports the **Content Inventory** section of `MIGRATE_qmcpy_checklist.md`. It records the Before-MCQMC preservation, retirement, and backlink-importance decisions and the current Post-MCQMC mapping state on `migrate_qmcpy`.

## Backup Sources Checked

- WordPress WXR export: `/Users/kangjiangrui/Downloads/qmcpy.WordPress.2026-05-26.xml`
- Jetpack backup: `/Users/kangjiangrui/Downloads/jetpack-backup-qmcpy-org-2026-05-25-20-48-57.tar.gz`
- Jetpack backup includes `wp-content/uploads/`, SQL tables, plugins, themes, and `wp-config.php`.

### Backup Integrity Evidence

Checked on 2026-07-16 without extracting the archive or reading configuration or database contents:

| Backup | Size | Local modification time | SHA-256 | Validation |
|---|---:|---|---|---|
| `qmcpy.WordPress.2026-05-26.xml` | 948,082 bytes | 2026-05-26 00:06:40 -0500 | `300e0c1153363a71eb07e9095ec314a346862825bbdca4bd1c13679bd0c3a005` | `xmllint --noout` passed |
| `jetpack-backup-qmcpy-org-2026-05-25-20-48-57.tar.gz` | 103,522,108 bytes | 2026-05-26 00:12:59 -0500 | `3228cca4344b3739a940ae42f619821667d13e06895bdd831ebb1dee08f1d0e4` | `gzip -t` passed; tar headers parsed successfully |

Jetpack archive structure:

- 9,785 archive members, all regular files.
- 231 upload files, 9,018 plugin files, and 518 theme files.
- 14 SQL files.
- A `wp-config.php` member is present; its contents were not read.

Minimum recovery procedure:

1. Work from copies of both backups and verify the recorded SHA-256 digests.
2. Restore only into an isolated WordPress test installation, never directly over the public site.
3. Import the WXR file for pages, posts, attachments, and menu metadata; use `wp-content/uploads/` from the Jetpack archive for media recovery.
4. Use the SQL and configuration files only for a separately approved full restore, after reviewing and sanitizing credentials and environment-specific values.
5. Keep both backups local; do not commit them or any extracted secrets.

## Summary

- WordPress pages: 8
- WordPress posts: 42
- WordPress attachments in XML: 93
- Files under `wp-content/uploads/` in Jetpack backup: 231
- Navigation menu items in XML: 12

### Inventory Reconciliation Evidence

Historical evidence checked on 2026-07-16 with structured XML parsing:

- Export status counts: 7 published pages, 1 draft page, 40 published posts, and 2 draft posts.
- The export contains 50 unique page/post URLs; the inventory contains the same 50 unique URLs, with no missing or extra entries.
- The export also contains 93 attachments and 12 navigation menu items, matching the summary above.
- All 18 migrated blog targets listed below exist in `docs/blogs/`.
- All 18 migrated blog files now contain an exact `Source WordPress URL` metadata match.
- Post-MCQMC verification on 2026-07-17 confirmed the 18 selected blog targets and exact source URL metadata after the five extra post candidates were removed from the current migration scope.
- On 2026-08-10, the owner superseded the earlier in-repository publishing
  decision and removed all 18 blog targets. Their old-path redirect entries now
  point to the canonical posts maintained by `QMCSoftware/qmcsoftware-website`.
  The local backups remain available for recovery.

## Current Must-Preserve Scope

Must-preserve scope was confirmed by Kang Jiangrui on 2026-05-26. Retirement and backlink-importance decisions were confirmed on 2026-07-16 and updated by the owner on 2026-08-10.

| Item | Preserve? | Current/new location | Status |
|---|---|---|---|
| Home | Yes | `README.md` | Minimum viable homepage completed and locally verified |
| Publications | Yes | `docs/publications.md` | Stable GitHub Pages target added and locally verified |
| Blogs | No (in this repo) | [`QMCSoftware/qmcsoftware-website`](https://github.com/QMCSoftware/qmcsoftware-website) | Blog publishing belongs to the separate website repository; all 18 migrated posts and their MkDocs publishing artifacts were removed here, while old paths remain as external redirect-only compatibility pages |
| Contributors | Yes | `community.md` | Both old contributor paths map to the current Community page |
| GitHub | Yes | `https://github.com/QMCSoftware/QMCSoftware` | Link present in README and WordPress nav export |
| Docs | Yes | `https://qmcsoftware.github.io/QMCSoftware/` | Link present in README and WordPress nav export |
| PyPI | Yes | `https://pypi.org/project/qmcpy/` | Link present in README and WordPress nav export |

Standalone Donation, Videos, and Dev Tools pages and unpublished drafts are confirmed for retirement. Contributor content is preserved in `community.md`, with both old contributor paths included in the redirect manifest.

Blog responsibility now belongs to
[`QMCSoftware/qmcsoftware-website`](https://github.com/QMCSoftware/qmcsoftware-website).
The MkDocs documentation site in this repository will not add blog pages or
blog navigation. Its preserved old blog paths only redirect readers to the
[QMCSoftware Blog](https://qmcsoftware.org/blog/).

## Before-MCQMC Preservation Policies

Confirmed by the project owner and sole collaborator on 2026-07-16. Updated by the project owner on 2026-07-17 to remove the five extra post candidates from the current migration scope, then superseded on 2026-08-10 for the 18 previously migrated blogs.

### Retirement Policy

Use a conservative classification based on whether public content has a confirmed preservation reason or an existing current target:

- Retire the standalone Donation, Videos, and Dev Tools pages.
- Retire the three unpublished WordPress drafts.
- Retire published news, event, announcement, poster, and talk posts that have no confirmed current target or preservation exception.
- Preserve contributor content for later mapping to an appropriate current page.
- Remove all 18 previously migrated blogs from the repository, navigation, and
  author snippets. Preserve their old paths only as redirects to the canonical
  copies maintained by `QMCSoftware/qmcsoftware-website`.
- Maintain future blog content in `QMCSoftware/qmcsoftware-website`; do not add
  blog pages or blog navigation to this repository's MkDocs site.
- Do not migrate the five extra published post candidates previously selected: the MCQMC 2020 tutorial, QMCPy v1.0 announcement, QMC software article, elliptic PDE demo, and UM-Bridge article.
- The 2026-08-10 removal applies to repository content, not the local WordPress
  export or Jetpack backup.

### External Backlink Preservation Policy

Apply the following tiered requirements to the external references recorded below:

- **High:** preserve access through the `qmcpy.org` main domain. If the external source cites a specific old path, preserve that path during Post-MCQMC URL mapping.
- **Medium:** preserve main-domain access. Preserve a specific old path only when the external source explicitly cites that path.
- **Low:** retain the reference as inventory evidence; no migration action is required.
- Low-value directory and SEO results require no preservation work.

The Illinois Tech Elevate reference remains recorded as evidence that explicitly
identifies the "Why Add Q to MC?" blog. Its old path remains in
`data/redirects.yml` and redirects to the canonical post maintained by
`QMCSoftware/qmcsoftware-website`. All other high- and medium-importance
references currently recorded cite the main domain or general project resources.

Read-only HTTP checks on 2026-07-16 returned HTTP 200 for 9 of the 13 recorded external sources. Four automated checks were inconclusive: ResearchGate and FNAL returned HTTP 403, LinkedIn returned HTTP 405 to a HEAD request, and the IIT Undergraduate Research Journal request timed out after 30 seconds. These results indicate access-control or availability uncertainty, not confirmed content removal, so the approved preservation requirements remain unchanged.

This policy defines preservation requirements only. Exact targets and the redirect mechanism are recorded in the Post-MCQMC sections below; live verification remains Final Hosting Migration work.

## Working Classification Summary

### Historical Before-MCQMC Migrated Content

The following WordPress posts had GitHub Pages targets in `docs/blogs/` before
the 2026-08-10 owner-approved removal:

- Why Add Q to MC?
- A QMCPy Quick Start
- What Makes a Sequence "Low Discrepancy"?
- qEI with QMCPy
- Safe Handling of QMC Points
- Speeding up QMCPy with Distributable C Code
- Visualizing the Internals of Object Classes in QMCPy
- Digital Sequences, the Niederreiter Construction
- Bayesian Stopping Criteria
- Accelerating Rare-event Reliability Simulations for CERN's Large Hadron Collider using QMCPy
- Random Lattice Generators are Not Bad
- Analysis of Quasi-Monte Carlo Efficiency for Asian Option Pricing
- Linear Matrix Scrambling and Digital Shift for Halton
- Highly Efficient Geometric Brownian Motion Modeling with QMCPy
- Parsl Accelerated QMCPy Notebook Tests
- CubMCCLTVec: Vectorizing the CubMCCLT Algorithm
- Visualizing the Generated Samples Helps
- Extending SciPyWrapper of QMCPy to Support Dependent and Custom Distributions

All 18 targets, their publishing metadata, and their post-specific assets have
now been removed from this branch. The backups listed above remain the recovery
source.

### Post-MCQMC Mapping Completion

- `https://qmcpy.org/publications/` maps to `docs/publications.md`.
- `https://qmcpy.org/` maps to the reviewed minimum viable homepage in `README.md`.
- `https://qmcpy.org/contributors/` and `https://qmcpy.org/contributors-2/` map to `community.md`.
- The 18 previously migrated posts were removed from this repository; their
  path-exact redirect entries now target the canonical posts at
  `https://qmcsoftware.org/blog/`.
- The five extra post candidates are intentionally excluded from the current migration scope.

### Confirmed Retirement Decisions

Confirmed by the project owner and sole collaborator on 2026-07-16:

- WordPress draft content:
  - `https://qmcpy.org/?page_id=689`
  - `https://qmcpy.org/?p=1670`
  - `https://qmcpy.org/?p=1710`
- Standalone non-must-preserve WordPress pages:
  - Donation
  - Dev Tools
  - Videos
- Old news/event/announcement posts not currently matched to GitHub Pages content.
- Preserve contributor content. `community.md` was selected as the target during Post-MCQMC URL mapping.

### Confirmed External URLs / Backlinks

Public search references were found on 2026-05-26. Their importance ratings were confirmed by the project owner and sole collaborator on 2026-07-16 for the current inventory. Exact redirect decisions remain Post-MCQMC work. Search queries included:

- `"qmcpy.org" -site:qmcpy.org`
- `"https://qmcpy.org" -site:qmcpy.org`
- `"qmcpy.org/2020" -site:qmcpy.org`
- `"qmcpy.org/publications" -site:qmcpy.org`
- `"qmcpy.org" "Why Add Q to MC" -site:qmcpy.org`
- `"qmcpy.org" "QMCPy Version 1.0" -site:qmcpy.org`

| External source | URL | Confirmed importance | Confirmed preservation requirement | 2026-07-16 HTTP check |
|---|---|---|---|---|
| ResearchGate publication page | https://www.researchgate.net/publication/379192764_Quasi-Monte_Carlo_Improtance_Sampling_with_QMCPY | Medium | Preserve main-domain access; no specific old path is recorded. | 403; automated access blocked |
| IIT Undergraduate Research Journal PDF | https://urj.library.iit.edu/index.php/urj/article/download/48/10/ | High | Preserve main-domain access; no specific old path is recorded. | Timed out after 30 seconds; inconclusive |
| UPC thesis PDF | https://upcommons.upc.edu/bitstream/handle/2117/411422/Adaptive%20Quasi-Monte%20Carlo%20Methods%20for%20Density%20Estimation%20in%20Quantitative%20Finance.pdf?sequence=4 | Medium | Preserve main-domain access; no specific old path is recorded. | 200 |
| NA Digest 2021 announcement | https://www.netlib.org/na-digest-html/21/v21n06.html | High | Preserve main-domain access; no specific old path is recorded. | 200 |
| Illinois Tech Elevate page | https://elevate.iit.edu/experiences/speedy-simulations/ | High | Preserve main-domain access and the old "Why Add Q to MC?" blog path. | 200 |
| Aleksei Sorokin poster PDF | https://alegresor.github.io/posters/2021_QMCPy_SIAMCSE.pdf | Medium | Preserve main-domain access; no specific old path is recorded. | 200 |
| Argonne 2023 talk PDF | https://indico.fnal.gov/event/59808/contributions/267047/attachments/167069/223340/Argonne_2023_May_Talk.pdf | High | Preserve main-domain access; no specific old path is recorded. | 403; automated access blocked |
| MCM 2021 program PDF | https://www.uni-mannheim.de/media/Lehrstuehle/wim/neuenkirch/program_mcm_08102021.pdf | Medium | Preserve main-domain access; no specific old path is recorded. | 200 |
| SIAM News PDF | https://www.siam.org/media/fixop3gp/sn_october2023.pdf | High | Preserve main-domain access; no specific old path is recorded. | 200 |
| Sou-Cheng Choi LinkedIn project entry | https://www.linkedin.com/in/sou-cheng-choi-7682b65 | Medium | Preserve main-domain access; no specific old path is recorded. | 405 to HEAD request; inconclusive |
| Hugging Face dataset excerpt | https://huggingface.co/datasets/MathGenie/MathCode-Pile/viewer/default/train?p=1 | Low | Retain as evidence; no migration action is required. | 200 |
| Speaker Deck: Low Discrepancy at Argonne 2023 May | https://speakerdeck.com/fjhickernell/low-discrepancy-at-argonne-2023-may | Low | Retain as evidence; no old-domain action is required. | 200 |
| Speaker Deck: SURE 2024 Kickoff | https://speakerdeck.com/fjhickernell/sure-2024-kickoff | Medium | Preserve main-domain access; no specific old path is recorded. | 200 |

The following low-value directory/SEO results do not require preservation work:

- https://obake.pages.dev/19/ODhIMRrXdF
- https://seol.store/domain/domain/part/192471/lan/en
- https://www.webscan.cc/site_www.truffle.report/

## GitHub Pages Verification Notes

Checked on 2026-07-16:

| Item | Result | Evidence |
|---|---|---|
| Existing GitHub Pages deployment | Pass | Homepage, `CONTRIBUTING/`, `demos/qmcpy_intro/`, and a directly addressed migrated blog returned HTTP 200. |
| Installation instructions | Pass | The rendered homepage contains `pip install qmcpy`; PyPI returned HTTP 200. |
| Documentation links | Pass with known gaps | Core documentation and direct blog URLs work, but `/blogs/` and `/publications/` indexes returned HTTP 404. |
| PyPI link | Pass | `curl -I -L https://pypi.org/project/qmcpy/` returned HTTP 200. |
| GitHub link | Pass | `curl -I -L https://github.com/QMCSoftware/QMCSoftware` returned HTTP 200. |
| `qmcpy.org` temporary homepage | Pass with migration gap | The apex returned HTTP 200 from GitHub Pages with a 398-byte "under construction" page. |
| Important old paths | Expected Post-MCQMC gap | `https://qmcpy.org/publications/` and the old "Why Add Q to MC?" path returned HTTP 404. |

### Domain and DNS Evidence

Read-only checks on 2026-07-16:

- Public WHOIS reports Automattic Inc. as registrar, registry expiration at `2028-08-31T22:25:48Z`, and DNSSEC as unsigned.
- Authoritative nameservers remain `ns1.wordpress.com`, `ns2.wordpress.com`, and `ns3.wordpress.com`.
- The apex A records are the four GitHub Pages addresses: `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, and `185.199.111.153`.
- No apex AAAA record was returned.
- `www.qmcpy.org` is a CNAME to `qmcsoftware.github.io` and redirects to the apex homepage.
- A wildcard probe resolves through `CNAME * -> qmcpy.org` to the same GitHub Pages IPv4 addresses.
- Email forwarding remains `MX 0 smtp-fwd.wordpress.com`.
- SPF is `v=spf1 include:_spf.wpcloud.com ~all`; DMARC is `v=DMARC1;p=none;`.
- No DS record is published. DKIM selector names were not present in the repository evidence and were not guessed or enumerated.
- No DNS, registrar, GitHub Pages, or WordPress setting was changed.

Build command:

```bash
conda run -n qmcpy python -m mkdocs build
```

Build result:

- The 2026-05-26 build completed successfully.
- On 2026-07-16, `conda run -n qmcpy make copydocs` stopped at the paper-render step because `pandoc` is not installed in the environment. The failed command produced no additional tracked file changes.
- `conda run -n qmcpy python -m mkdocs build` then completed successfully in approximately 52 seconds.

2026-07-16 warning classification:

| Message class | Count | Classification | Before-MCQMC impact |
|---|---:|---|---|
| Missing UML SVG link targets | 10 | Existing generated-asset gap in API documentation | Non-blocking; unrelated to the migration inventory and source metadata changes |
| Missing `QMCPy_Shared_Leadership.md` navigation target | 1 | Latest `develop` references a root document that `copydocs` does not place in `docs/` | Non-blocking for the merge; current `develop` documentation workflow gap |
| Untracked `AGENTS.md` pages omitted from navigation | 2 | Local-worktree information message | Non-blocking; files are untracked and not part of the branch |
| Black or Ruff unavailable for signature formatting | 1 | Optional mkdocstrings formatting information | Non-blocking; API documentation still builds |
| Material for MkDocs 2.0 advisory | 1 | Upstream compatibility announcement | Non-blocking for the current MkDocs 1.6.1 build |
| Missing `pandoc` in `make copydocs` | 1 | Local environment limitation | Does not block the successful direct MkDocs build; must be resolved before requiring the full copy workflow |

No warning was introduced by the four `Source WordPress URL` metadata changes.

### Post-MCQMC Local Implementation Verification

Historical check on `migrate_qmcpy` on 2026-07-17; blog-specific results below
were superseded on 2026-08-10:

- `docs/blogs/index.md` links all 18 preserved posts.
- `docs/publications.md` preserves the old publication list and adds the current QMCPy software paper from repository-visible sources.
- `data/redirects.yml` contains 20 unique old paths and no missing targets.
- `scripts/make_redirect_pages.py` generated all 20 path-exact pages.
- The full copy, UML, and MkDocs build workflows completed successfully; the final build reported no warnings or unresolved relative links.
- Browser checks covered desktop and mobile layouts and the required old "Why Add Q to MC?" path.
- No GitHub Pages setting, DNS record, WordPress setting, or live site was changed. Live URL checks remain pending publication and Final Hosting Migration.

### Latest Develop Synchronization Verification

Checked after merging `origin/develop` at `a1ab9969` on 2026-07-16:

- The merge completed without conflicts.
- The 50 exported page/post URLs still match the 50 inventory entries exactly.
- All 18 migrated blog targets and Source WordPress URL metadata still match.
- Relative to `origin/develop`, the migration branch adds only the three Before-MCQMC migration documents and four blog metadata changes.
- `conda run -n qmcpy make copydocs` still stops because `pandoc` is missing and produces no tracked file changes.
- `conda run -n qmcpy python -m mkdocs build` completed successfully in approximately 60 seconds.
- The latest `develop` adds one non-blocking navigation warning for `QMCPy_Shared_Leadership.md`, in addition to the previously classified baseline messages.

## Pages

| Old URL | Title | Status | Current/new location | Inventory note |
|---|---|---|---|---|
| https://qmcpy.org/ | Blog | publish | README.md | preserved: minimum viable homepage completed and locally verified |
| https://qmcpy.org/blog/donation/ | Donation | publish |  | retire: standalone page is outside the must-preserve scope |
| https://qmcpy.org/contributors-2/ | Contributors | publish | community.md | preserved: path-exact redirect maps to the Community page |
| https://qmcpy.org/contributors/ | Contributors | publish | community.md | preserved: path-exact redirect maps to the Community page |
| https://qmcpy.org/publications/ | Publications | publish | publications.md | preserved: stable Publications page added |
| https://qmcpy.org/references-for-python-and-mathematical-software-development/ | Dev Tools | publish |  | retire: standalone page is outside the must-preserve scope |
| https://qmcpy.org/videos/ | Videos | publish |  | retire: standalone page is outside the must-preserve scope |
| https://qmcpy.org/?page_id=689 | Tools | draft |  | retire: unpublished WordPress draft |

## Posts

| Old URL | Title | Status | Current/new location | Inventory note |
|---|---|---|---|---|
| https://qmcpy.org/2020/06/25/why_add_q_to_mc/ | Why Add Q to MC? | publish | https://qmcsoftware.org/blog/why-add-q-to-mc/ | preserved by external redirect |
| https://qmcpy.org/2020/07/06/a-qmcpy-quick-start/ | A QMCPy Quick Start | publish | https://qmcsoftware.org/blog/a-qmcpy-quick-start/ | preserved by external redirect |
| https://qmcpy.org/2020/07/08/what-makes-a-sequence-low-discrepancy/ | What Makes a Sequence "Low Discrepancy"? | publish | https://qmcsoftware.org/blog/what-makes-a-sequence-low-discrepancy/ | preserved by external redirect |
| https://qmcpy.org/2020/07/19/qei-with-qmcpy/ | qEI with QMCPy | publish | https://qmcsoftware.org/blog/qei-with-qmcpy/ | preserved by external redirect |
| https://qmcpy.org/2020/08/31/safe-handling-of-qmc-points/ | Safe Handling of QMC Points | publish | https://qmcsoftware.org/blog/safe-handling-of-qmc-points/ | preserved by external redirect |
| https://qmcpy.org/2020/09/03/mcqmc-2020-tutorial/ | A Tutorial at MCQMC 2020 | publish |  | not migrated: removed from current scope per 2026-07-17 owner instruction |
| https://qmcpy.org/2020/09/03/posters/ | A Collection of QMCPy Posters | publish |  | retire: no confirmed current target |
| https://qmcpy.org/2020/09/03/pydata-chicago-talk/ | A Seminar at PyData Chicago | publish |  | retire: no confirmed current target |
| https://qmcpy.org/2021/02/12/qmcpy-version-1-0/ | QMCPy Version 1.0 | publish |  | not migrated: removed from current scope per 2026-07-17 owner instruction |
| https://qmcpy.org/2021/02/25/quasi-monte-carlo-software-article/ | Quasi-Monte Carlo Software Article | publish |  | not migrated: removed from current scope per 2026-07-17 owner instruction |
| https://qmcpy.org/2021/02/25/speeding-up-qmcpy-with-distributable-c-code/ | Speeding up QMCPy with Distributable C Code | publish | https://qmcsoftware.org/blog/speeding-up-qmcpy-with-distributable-c-code/ | preserved by external redirect |
| https://qmcpy.org/2021/02/25/visualizing-the-internals-of-object-classes-in-qmcpy/ | Visualizing the Internals of Object Classes in QMCPy | publish | https://qmcsoftware.org/blog/visualizing-the-internals-of-object-classes-in-qmcpy/ | preserved by external redirect |
| https://qmcpy.org/2021/03/11/monte-carlo-methods-2021/ | Monte Carlo Methods 2021 | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/04/27/a-presentation-at-iits-computational-mathematics-seminar/ | A Presentation at IIT's Computational Mathematics Seminar | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/04/27/a-talk-at-the-chicago-area-siam-student-conference/ | A Talk at the Chicago Area SIAM Student Conference | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/04/27/a-talk-at-the-great-lakes-siam-conference/ | A Talk at the Great Lakes SIAM Conference | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/06/04/digital-sequences-the-niederreiter-construction/ | Digital Sequences, the Niederreiter Construction | publish | https://qmcsoftware.org/blog/digital-sequences-the-niederreiter-construction/ | preserved by external redirect |
| https://qmcpy.org/2021/06/04/elliptic-pde-demo/ | Elliptic PDE Demo | publish |  | not migrated: removed from current scope per 2026-07-17 owner instruction |
| https://qmcpy.org/2022/02/22/qmcpy-events-coming-soon/ | QMCPy Events Coming Soon | publish |  | retire: old event announcement without a confirmed current target |
| https://qmcpy.org/2022/05/19/bayesian-stopping-criteria/ | Bayesian Stopping Criteria | publish | https://qmcsoftware.org/blog/bayesian-stopping-criteria/ | preserved by external redirect |
| https://qmcpy.org/2023/04/07/accelerating-rare-event-reliability-simulations-for-cerns-large-hadron-collider-using-qmcpy/ | Accelerating Rare-event Reliability Simulations for CERN's Large Hadron Collider using QMCPy | publish | https://qmcsoftware.org/blog/accelerating-rare-event-reliability-simulations-for-cerns-large-hadron-collider-using-qmcpy/ | preserved by external redirect |
| https://qmcpy.org/2023/04/28/illinois-tech-receives-nsf-grant-to-offer-intensive-summer-research-program-in-computational-mathematics-and-data-science-for-undergraduates/ | Illinois Tech Receives NSF Grant to Offer Intensive Summer Research Program in Computational Mathematics and Data Science for Undergraduates | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/05/08/combining-the-expertise-of-the-stasasticians-and-a-commitment-to-esg-principles-to-deliver-comprehensive-wealth-management-and-investment/ | Combining the Expertise of the StaSASticians and a Commitment to ESG Principles to Deliver Comprehensive Wealth Management and Investment | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/05/13/boosting-numerical-efficiency-with-low-discrepancy-sampling-enhancing-estimation-and-integration-in-diverse-fields-from-fred-hickernell-and-the-qmcpy-library/ | Boosting Numerical Efficiency with Low Discrepancy Sampling: Enhancing Estimation and Integration in Diverse Fields from Fred Hickernell and the QMCPy Library | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/05/16/random-lattice-generators-are-not-bad/ | Random Lattice Generators are Not Bad | publish | https://qmcsoftware.org/blog/random-lattice-generators-are-not-bad/ | preserved by external redirect |
| https://qmcpy.org/2023/07/04/open-source-an-important-tool-to-advancing-science/ | Open Source Software: An Important Tool for Advancing Science | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/07/05/qmcpy-talks-at-the-14th-international-conference-on-monte-carlo-methods-and-applications/ | QMCPY talks at the 14th International Conference on Monte Carlo Methods and Applications | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2023/08/06/innovative-investment-tool-qualifies-students-for-international-hackathon/ | Innovative Investment Tool Qualifies Students for International Hackathon | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/08/28/from-speedy-simulations-to-trustworthy-ai-undergraduates-take-on-research-challenges-at-illinois-tech-for-the-summer/ | From Speedy Simulations to Trustworthy AI: Undergraduates Take on Research Challenges at Illinois Tech for the Summer | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/10/04/simplifying-uncertainty-modeling-by-handing-the-backend-modeling-with-um-bridge/ | Simplifying and Improving Uncertainty Quantification with UM-Bridge | publish |  | not migrated: removed from current scope per 2026-07-17 owner instruction |
| https://qmcpy.org/2024/05/13/exploring-the-frontiers-the-15th-international-conference-on-monte-carlo-methods-and-applications-mcm-2025-at-illinois-institute-of-technology/ | Exploring the Frontiers: The 15th International Conference on Monte Carlo Methods and Applications (MCM 2025) at Illinois Institute of Technology | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2024/05/13/karl-menger-graduate-teaching-assistant-award/ | Karl Menger Graduate Teaching Assistant Award | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2024/10/20/aleksei-secures-doe-research-fellowship/ | Aleksei Sorokin Awarded Prestigious DOE Research Fellowship | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2025/07/15/analysis-of-quasi-monte-carlo-efficiency-for-asian-option-pricing/ | Analysis of Quasi-Monte Carlo Efficiency for Asian Option Pricing | publish | https://qmcsoftware.org/blog/analysis-of-qmc-efficiency-for-asian-option-pricing/ | preserved by external redirect |
| https://qmcpy.org/2025/09/29/linear-matrix-scrambling-and-digital-shift-for-halton/ | Linear Matrix Scrambling and Digital Shift for Halton | publish | https://qmcsoftware.org/blog/linear-matrix-scrambling-and-digital-shift-for-halton/ | preserved by external redirect |
| https://qmcpy.org/2025/10/22/highly-efficient-geometric-brownian-motion-modeling-with-qmcpy/ | Highly Efficient Geometric Brownian Motion Modeling with QMCPy | publish | https://qmcsoftware.org/blog/gbm-qmcpy/ | preserved by external redirect |
| https://qmcpy.org/2025/11/29/parsl-accelerated-qmcpy-notebook-tests/ | Parsl Accelerated QMCPy Notebook Tests | publish | https://qmcsoftware.org/blog/accelerating-qmcpy-notebook-tests-with-parsl/ | preserved by external redirect |
| https://qmcpy.org/2026/02/25/cubmccltvec-vectorizing-the-cubmcclt-algorithm/ | CubMCCLTVec: Vectorizing the CubMCCLT Algorithm | publish | https://qmcsoftware.org/blog/cubmccltvec-vectorizing-the-cubmcclt-algorithm/ | preserved by external redirect |
| https://qmcpy.org/2026/02/25/visualizing-the-generated-samples-helps/ | Visualizing the Generated Samples Helps | publish | https://qmcsoftware.org/blog/visualizing-the-generated-samples-helps/ | preserved by external redirect |
| https://qmcpy.org/2026/04/18/extending-scipywrapper-of-qmcpy-to-support-dependent-and-custom-distributions/ | Extending SciPyWrapper of QMCPy to Support Dependent and Custom Distributions | publish | https://qmcsoftware.org/blog/scipywrapper/ | preserved by external redirect |
| https://qmcpy.org/?p=1670 | Visualizing Discrete Distributions and True Measures: Unveiling Insights with QMCPy Plot Projections | draft |  | retire: unpublished WordPress draft |
| https://qmcpy.org/?p=1710 | Visualizing Discrete Distributions and True Measure Objects with QMCPy Plot Projections: Enhancing Understanding Through Graphical Representation | draft |  | retire: unpublished WordPress draft |

## Current Confirmed Decisions

- Preserve Home, Publications, GitHub, Docs, PyPI, and contributor content.
- Remove all 18 previously migrated blog posts and their repository publishing
  artifacts, as confirmed by the owner on 2026-08-10.
- Assign future blog publishing to `QMCSoftware/qmcsoftware-website`; the MkDocs
  documentation site in this repository remains blog-free.
- Preserve the 18 old blog URLs as redirect-only pages to the corresponding
  canonical posts at `https://qmcsoftware.org/blog/`.
- Do not migrate the five extra published post candidates removed from the current scope on 2026-07-17.
- Retire Donation, Videos, Dev Tools, unpublished drafts, and unmatched old news, event, and announcement posts.
- Keep the tiered external-backlink evidence above, including the old
  "Why Add Q to MC?" path requirement.
- `data/redirects.yml` contains two internal contributor mappings and 18
  external blog mappings; live old-domain verification remains Final Hosting
  Migration work.
