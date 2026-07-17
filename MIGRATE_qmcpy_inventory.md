# QMCPy.org Migration Inventory

_Generated from `/Users/kangjiangrui/Downloads/qmcpy.WordPress.2026-05-26.xml` on 2026-05-26._

This file supports the **Content Inventory** section of
`MIGRATE_qmcpy_checklist.md`. It records the Before-MCQMC preservation,
retirement, and backlink-importance decisions. Old-to-new URL mapping and
redirect implementation remain Post-MCQMC work.

## Backup Sources Checked

- WordPress WXR export: `/Users/kangjiangrui/Downloads/qmcpy.WordPress.2026-05-26.xml`
- Jetpack backup: `/Users/kangjiangrui/Downloads/jetpack-backup-qmcpy-org-2026-05-25-20-48-57.tar.gz`
- Jetpack backup includes `wp-content/uploads/`, SQL tables, plugins, themes, and `wp-config.php`.

## Summary

- WordPress pages: 8
- WordPress posts: 42
- WordPress attachments in XML: 93
- Files under `wp-content/uploads/` in Jetpack backup: 231
- Navigation menu items in XML: 12

## Current Must-Preserve Scope

Must-preserve scope confirmed by Kang Jiangrui on 2026-05-26. Retirement and
backlink-importance decisions confirmed by Kang Jiangrui as collaborator
representative on 2026-07-16.

| Item | Preserve? | Current/new location | Status |
|---|---|---|---|
| Home | Yes | `README.md` | Existing GitHub Pages homepage candidate; needs final content review |
| Publications | Yes | TBD | Must be preserved; no GitHub Pages target identified yet |
| Blogs | Yes | `docs/blogs/` and `mkdocs.yml` Blogs navigation | Preserve the 18 already-migrated posts; retire unmatched old news, event, and announcement posts |
| GitHub | Yes | `https://github.com/QMCSoftware/QMCSoftware` | Link present in README and WordPress nav export |
| Docs | Yes | `https://qmcsoftware.github.io/QMCSoftware/` | Link present in README and WordPress nav export |
| PyPI | Yes | `https://pypi.org/project/qmcpy/` | Link present in README and WordPress nav export |

Standalone Donation, Videos, and Dev Tools pages and unpublished drafts are
confirmed for retirement. Contributor content remains preserved; its final URL
mapping is deferred to Post-MCQMC work.

## Working Classification Summary

### Already Migrated Candidates

The following WordPress posts have clear GitHub Pages targets in `docs/blogs/`:

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

### Preservation Gaps

- `https://qmcpy.org/publications/` is marked must-preserve and needs a future GitHub Pages target.
- `https://qmcpy.org/` maps to the current GitHub Pages homepage candidate, `README.md`, but still needs final content review.

### Confirmed Retirement Decisions

Confirmed by the collaborator representative on 2026-07-16:

- WordPress draft content:
  - `https://qmcpy.org/?page_id=689`
  - `https://qmcpy.org/?p=1670`
  - `https://qmcpy.org/?p=1710`
- Standalone non-must-preserve WordPress pages:
  - Donation
  - Dev Tools
  - Videos
- Old news/event/announcement posts not currently matched to GitHub Pages content.
- Preserve contributor content. Evaluate `community.md` as the target during
  Post-MCQMC URL mapping.

### Confirmed External URLs / Backlinks

Public search references were found on 2026-05-26. Their importance ratings
were confirmed by the collaborator representative on 2026-07-16 for the
current inventory. Exact redirect decisions remain Post-MCQMC work. Search
queries included:

- `"qmcpy.org" -site:qmcpy.org`
- `"https://qmcpy.org" -site:qmcpy.org`
- `"qmcpy.org/2020" -site:qmcpy.org`
- `"qmcpy.org/publications" -site:qmcpy.org`
- `"qmcpy.org" "Why Add Q to MC" -site:qmcpy.org`
- `"qmcpy.org" "QMCPy Version 1.0" -site:qmcpy.org`

| External source | URL | Confirmed importance | Notes / suggested handling |
|---|---|---|---|
| ResearchGate publication page | https://www.researchgate.net/publication/379192764_Quasi-Monte_Carlo_Improtance_Sampling_with_QMCPY | Medium | Mentions `qmcpy.org` and GitHub repository. Academic context, but not an official project-controlled source. Main-domain forwarding should be sufficient unless collaborators know this page is actively used. |
| IIT Undergraduate Research Journal PDF | https://urj.library.iit.edu/index.php/urj/article/download/48/10/ | High | Institutional academic PDF mentioning `qmcpy.org`. Preserve main-domain access; page-specific redirects are probably unnecessary unless the PDF cites a specific old path. |
| UPC thesis PDF | https://upcommons.upc.edu/bitstream/handle/2117/411422/Adaptive%20Quasi-Monte%20Carlo%20Methods%20for%20Density%20Estimation%20in%20Quantitative%20Finance.pdf?sequence=4 | Medium | Thesis bibliography references `https://qmcpy.org/`. Preserve main-domain access. |
| NA Digest 2021 announcement | https://www.netlib.org/na-digest-html/21/v21n06.html | High | Stable numerical analysis community archive. Mentions `https://qmcpy.org` as the entry point for repository, documentation, blog, and tutorials. Preserve main-domain access and make the new homepage clearly route to docs/blogs/GitHub. |
| Illinois Tech Elevate page | https://elevate.iit.edu/experiences/speedy-simulations/ | High | Institutional outreach page directing readers to `qmcpy.org` and the "Why Add Q to MC?" blog post. Preserve main-domain access and ensure the blog remains discoverable from the new site. |
| Aleksei Sorokin poster PDF | https://alegresor.github.io/posters/2021_QMCPy_SIAMCSE.pdf | Medium | Project-related poster listing `qmcpy.org` as the QMC blog posts website. Preserve main-domain access; blogs should remain easy to find. |
| Argonne 2023 talk PDF | https://indico.fnal.gov/event/59808/contributions/267047/attachments/167069/223340/Argonne_2023_May_Talk.pdf | High | Conference/talk PDF hosted on an institutional event system; contains "Visit us at qmcpy.org". Preserve main-domain access. |
| MCM 2021 program PDF | https://www.uni-mannheim.de/media/Lehrstuehle/wim/neuenkirch/program_mcm_08102021.pdf | Medium | Conference program references "QMCPy Homepage, 2020+. qmcpy.org." Preserve main-domain access. |
| SIAM News PDF | https://www.siam.org/media/fixop3gp/sn_october2023.pdf | High | SIAM News is a high-credibility public source and contains a `qmcpy.org` reference. Preserve main-domain access. |
| Sou-Cheng Choi LinkedIn project entry | https://www.linkedin.com/in/sou-cheng-choi-7682b65 | Medium | Project entry lists the QMCPy website as `https://qmcpy.org`. Could be updated manually later if desired, but main-domain forwarding should be enough. |
| Hugging Face dataset excerpt | https://huggingface.co/datasets/MathGenie/MathCode-Pile/viewer/default/train?p=1 | Low | Contains a copied notebook/text excerpt linking to `https://qmcpy.org/`; likely not a user-facing source. Record for completeness only. |
| Speaker Deck: Low Discrepancy at Argonne 2023 May | https://speakerdeck.com/fjhickernell/low-discrepancy-at-argonne-2023-may | Low | Uses the GitHub Pages documentation URL, not primarily an old `qmcpy.org` dependency. No special old-domain action likely needed. |
| Speaker Deck: SURE 2024 Kickoff | https://speakerdeck.com/fjhickernell/sure-2024-kickoff | Medium | Fred's deck directs readers to visit `qmcpy.org`. Preserve main-domain access. |

The following low-value directory/SEO results do not require preservation work:

- https://obake.pages.dev/19/ODhIMRrXdF
- https://seol.store/domain/domain/part/192471/lan/en
- https://www.webscan.cc/site_www.truffle.report/

## GitHub Pages Verification Notes

Checked on 2026-05-26:

| Item | Result | Evidence |
|---|---|---|
| Existing GitHub Pages deployment | Pass | `curl -I -L https://qmcsoftware.github.io/QMCSoftware/` returned HTTP 200. |
| Installation instructions | Pass | `README.md` contains `pip install qmcpy`; PyPI page returned HTTP 200. |
| Documentation links | Pass with warnings | Docs homepage and selected documentation/demo URLs returned HTTP 200. Local `mkdocs build` completed successfully in the `qmcpy` conda environment, with existing warnings. |
| PyPI link | Pass | `curl -I -L https://pypi.org/project/qmcpy/` returned HTTP 200. |
| GitHub link | Pass | `curl -I -L https://github.com/QMCSoftware/QMCSoftware` returned HTTP 200. |

Build command:

```bash
conda run -n qmcpy python -m mkdocs build
```

Build result:

- Completed successfully.
- Warnings remain for existing nav references and resource links; these were not changed during this Before MCQMC inventory/verification pass.

## Pages

| Old URL | Title | Status | Current/new location | Inventory note |
|---|---|---|---|---|
| https://qmcpy.org/ | Blog | publish | README.md | Must preserve: existing GitHub Pages homepage candidate; needs final content review |
| https://qmcpy.org/blog/donation/ | Donation | publish |  | retire: standalone page is outside the must-preserve scope |
| https://qmcpy.org/contributors-2/ | Contributors | publish | community.md | preserve contributor content; confirm URL mapping during Post-MCQMC work |
| https://qmcpy.org/contributors/ | Contributors | publish | community.md | preserve contributor content; confirm URL mapping during Post-MCQMC work |
| https://qmcpy.org/publications/ | Publications | publish | TBD | Must preserve: new GitHub Pages target needed |
| https://qmcpy.org/references-for-python-and-mathematical-software-development/ | Dev Tools | publish |  | retire: standalone page is outside the must-preserve scope |
| https://qmcpy.org/videos/ | Videos | publish |  | retire: standalone page is outside the must-preserve scope |
| https://qmcpy.org/?page_id=689 | Tools | draft |  | retire: unpublished WordPress draft |

## Posts

| Old URL | Title | Status | Current/new location | Inventory note |
|---|---|---|---|---|
| https://qmcpy.org/2020/06/25/why_add_q_to_mc/ | Why Add Q to MC? | publish | blogs/why-add-q-to-mc/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2020/07/06/a-qmcpy-quick-start/ | A QMCPy Quick Start | publish | blogs/a-qmcpy-quick-start/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2020/07/08/what-makes-a-sequence-low-discrepancy/ | What Makes a Sequence "Low Discrepancy"? | publish | blogs/what-makes-a-sequence-low-discrepancy/index.md | already migrated: title/slug match; Source WordPress URL metadata missing |
| https://qmcpy.org/2020/07/19/qei-with-qmcpy/ | qEI with QMCPy | publish | blogs/qei-with-qmcpy/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2020/08/31/safe-handling-of-qmc-points/ | Safe Handling of QMC Points | publish | blogs/safe-handling-of-qmc-points/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2020/09/03/mcqmc-2020-tutorial/ | A Tutorial at MCQMC 2020 | publish |  | preserve; identify the current target during Post-MCQMC URL mapping |
| https://qmcpy.org/2020/09/03/posters/ | A Collection of QMCPy Posters | publish |  | retire: no confirmed current target |
| https://qmcpy.org/2020/09/03/pydata-chicago-talk/ | A Seminar at PyData Chicago | publish |  | retire: no confirmed current target |
| https://qmcpy.org/2021/02/12/qmcpy-version-1-0/ | QMCPy Version 1.0 | publish |  | preserve; identify the current target during Post-MCQMC URL mapping |
| https://qmcpy.org/2021/02/25/quasi-monte-carlo-software-article/ | Quasi-Monte Carlo Software Article | publish |  | preserve; identify the current target during Post-MCQMC URL mapping |
| https://qmcpy.org/2021/02/25/speeding-up-qmcpy-with-distributable-c-code/ | Speeding up QMCPy with Distributable C Code | publish | blogs/speeding-up-qmcpy-with-distributable-c-code/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2021/02/25/visualizing-the-internals-of-object-classes-in-qmcpy/ | Visualizing the Internals of Object Classes in QMCPy | publish | blogs/visualizing-the-internals-of-object-classes-in-qmcpy/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2021/03/11/monte-carlo-methods-2021/ | Monte Carlo Methods 2021 | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/04/27/a-presentation-at-iits-computational-mathematics-seminar/ | A Presentation at IIT's Computational Mathematics Seminar | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/04/27/a-talk-at-the-chicago-area-siam-student-conference/ | A Talk at the Chicago Area SIAM Student Conference | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/04/27/a-talk-at-the-great-lakes-siam-conference/ | A Talk at the Great Lakes SIAM Conference | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2021/06/04/digital-sequences-the-niederreiter-construction/ | Digital Sequences, the Niederreiter Construction | publish | blogs/digital-sequences-the-niederreiter-construction/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2021/06/04/elliptic-pde-demo/ | Elliptic PDE Demo | publish |  | preserve; identify the current target during Post-MCQMC URL mapping |
| https://qmcpy.org/2022/02/22/qmcpy-events-coming-soon/ | QMCPy Events Coming Soon | publish |  | retire: old event announcement without a confirmed current target |
| https://qmcpy.org/2022/05/19/bayesian-stopping-criteria/ | Bayesian Stopping Criteria | publish | blogs/bayesian-stopping-criteria/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2023/04/07/accelerating-rare-event-reliability-simulations-for-cerns-large-hadron-collider-using-qmcpy/ | Accelerating Rare-event Reliability Simulations for CERN's Large Hadron Collider using QMCPy | publish | blogs/accelerating-rare-event-reliability-simulations-for-cerns-large-hadron-collider-using-qmcpy/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2023/04/28/illinois-tech-receives-nsf-grant-to-offer-intensive-summer-research-program-in-computational-mathematics-and-data-science-for-undergraduates/ | Illinois Tech Receives NSF Grant to Offer Intensive Summer Research Program in Computational Mathematics and Data Science for Undergraduates | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/05/08/combining-the-expertise-of-the-stasasticians-and-a-commitment-to-esg-principles-to-deliver-comprehensive-wealth-management-and-investment/ | Combining the Expertise of the StaSASticians and a Commitment to ESG Principles to Deliver Comprehensive Wealth Management and Investment | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/05/13/boosting-numerical-efficiency-with-low-discrepancy-sampling-enhancing-estimation-and-integration-in-diverse-fields-from-fred-hickernell-and-the-qmcpy-library/ | Boosting Numerical Efficiency with Low Discrepancy Sampling: Enhancing Estimation and Integration in Diverse Fields from Fred Hickernell and the QMCPy Library | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/05/16/random-lattice-generators-are-not-bad/ | Random Lattice Generators are Not Bad | publish | blogs/random-lattice-generators-are-not-bad/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2023/07/04/open-source-an-important-tool-to-advancing-science/ | Open Source Software: An Important Tool for Advancing Science | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/07/05/qmcpy-talks-at-the-14th-international-conference-on-monte-carlo-methods-and-applications/ | QMCPY talks at the 14th International Conference on Monte Carlo Methods and Applications | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2023/08/06/innovative-investment-tool-qualifies-students-for-international-hackathon/ | Innovative Investment Tool Qualifies Students for International Hackathon | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/08/28/from-speedy-simulations-to-trustworthy-ai-undergraduates-take-on-research-challenges-at-illinois-tech-for-the-summer/ | From Speedy Simulations to Trustworthy AI: Undergraduates Take on Research Challenges at Illinois Tech for the Summer | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2023/10/04/simplifying-uncertainty-modeling-by-handing-the-backend-modeling-with-um-bridge/ | Simplifying and Improving Uncertainty Quantification with UM-Bridge | publish |  | preserve; identify the current target during Post-MCQMC URL mapping |
| https://qmcpy.org/2024/05/13/exploring-the-frontiers-the-15th-international-conference-on-monte-carlo-methods-and-applications-mcm-2025-at-illinois-institute-of-technology/ | Exploring the Frontiers: The 15th International Conference on Monte Carlo Methods and Applications (MCM 2025) at Illinois Institute of Technology | publish |  | retire: old event post without a confirmed current target |
| https://qmcpy.org/2024/05/13/karl-menger-graduate-teaching-assistant-award/ | Karl Menger Graduate Teaching Assistant Award | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2024/10/20/aleksei-secures-doe-research-fellowship/ | Aleksei Sorokin Awarded Prestigious DOE Research Fellowship | publish |  | retire: old news post without a confirmed current target |
| https://qmcpy.org/2025/07/15/analysis-of-quasi-monte-carlo-efficiency-for-asian-option-pricing/ | Analysis of Quasi-Monte Carlo Efficiency for Asian Option Pricing | publish | blogs/analysis-of-qmc-efficiency-for-asian-option-pricing/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2025/09/29/linear-matrix-scrambling-and-digital-shift-for-halton/ | Linear Matrix Scrambling and Digital Shift for Halton | publish | blogs/linear-matrix-scrambling-and-digital-shift-for-halton/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2025/10/22/highly-efficient-geometric-brownian-motion-modeling-with-qmcpy/ | Highly Efficient Geometric Brownian Motion Modeling with QMCPy | publish | blogs/gbm-qmcpy/index.md | already migrated: title/date match; Source WordPress URL metadata missing |
| https://qmcpy.org/2025/11/29/parsl-accelerated-qmcpy-notebook-tests/ | Parsl Accelerated QMCPy Notebook Tests | publish | blogs/accelerating-qmcpy-notebook-tests-with-parsl/index.md | already migrated: title/date match; Source WordPress URL metadata missing |
| https://qmcpy.org/2026/02/25/cubmccltvec-vectorizing-the-cubmcclt-algorithm/ | CubMCCLTVec: Vectorizing the CubMCCLT Algorithm | publish | blogs/cubmccltvec-vectorizing-the-cubmcclt-algorithm/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2026/02/25/visualizing-the-generated-samples-helps/ | Visualizing the Generated Samples Helps | publish | blogs/visualizing-the-generated-samples-helps/index.md | already migrated: exact Source WordPress URL match |
| https://qmcpy.org/2026/04/18/extending-scipywrapper-of-qmcpy-to-support-dependent-and-custom-distributions/ | Extending SciPyWrapper of QMCPy to Support Dependent and Custom Distributions | publish | blogs/scipywrapper/index.md | already migrated: title/date match; Source WordPress URL metadata missing |
| https://qmcpy.org/?p=1670 | Visualizing Discrete Distributions and True Measures: Unveiling Insights with QMCPy Plot Projections | draft |  | retire: unpublished WordPress draft |
| https://qmcpy.org/?p=1710 | Visualizing Discrete Distributions and True Measure Objects with QMCPy Plot Projections: Enhancing Understanding Through Graphical Representation | draft |  | retire: unpublished WordPress draft |

## Confirmed Before-MCQMC Decisions

- Preserve Home, Publications, Blogs, GitHub, Docs, PyPI, contributor content,
  and the five additional published posts marked `preserve` above.
- Retire Donation, Videos, Dev Tools, unpublished drafts, and unmatched old
  news, event, and announcement posts.
- External backlink importance is confirmed in this inventory.
- Exact target selection, redirect scope, and implementation remain
  Post-MCQMC work.
