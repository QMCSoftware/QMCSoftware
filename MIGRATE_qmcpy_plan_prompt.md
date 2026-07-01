# Multi-Agent Plan Mode Prompt for the Next QMCPy.org Migration Step

Use this from the repository root. It follows the Codex prompt structure of
Goal, Context, Constraints, and Done when. The first turn is planning-only;
implementation starts only after user approval.

```text
/plan Goal:
Plan and then, after my approval, execute the next independent QMCPy.org to GitHub Pages migration step using parallel subagents. Focus only on work that can be done without collaborator approval.

Context:
- Repository: /Users/kangjiangrui/Projects/QMCSoftware
- Branch: migrate_qmcpy
- Migration status files: MIGRATE_qmcpy_checklist.md and MIGRATE_qmcpy_inventory.md
- Current site target: https://qmcsoftware.github.io/QMCSoftware/
- Existing must-preserve scope: Home, Publications, Blogs, GitHub, Docs, PyPI
- Known independent work areas: minimum viable homepage, navigation cleanup, old URL mapping, missing source URL metadata, mechanical Markdown/link fixes, and generated QMC software ecosystem table hygiene.
- Persistent repo guidance: read AGENTS.md plus any closer AGENTS.md files under docs/, docs/blogs/, snippets/, and data/ before planning or editing those areas.

Constraints:
- Start read-only in Plan Mode. Inspect git status, the checklist, the inventory, README.md, mkdocs.yml, docs/, docs/blogs/, snippets/, and data/ as needed.
- Do not make edits during this planning turn. Ask for my approval before implementation.
- Put collaborator-dependent decisions aside. Do not decide retirement of Donation, Videos, Dev Tools, draft posts, old news/event posts, or external backlink importance.
- Do not change DNS, GitHub Pages settings, WordPress settings, domain forwarding, custom-domain settings, branches, commits, pushes, or PRs.
- Preserve scientific and mathematical meaning. Do not change runtime algorithms, public APIs, tests, or benchmarks.
- Prefer small, reviewable tasks with explicit validation commands.
- Avoid parallel write conflicts. If two tasks need the same file, serialize those edits through the main agent instead of assigning both agents to edit it.

Subagent plan:
- During planning, propose a small first batch of 2-3 independent workstreams and the exact files each workstream may edit.
- After I approve the plan, spawn one execution subagent per approved workstream.
- Execution subagents should use high reasoning: request model_reasoning_effort=high for each execution worker. They may edit only their assigned files.
- Wait for all execution subagents, then the main agent should reconcile their results and run the relevant checks.
- After implementation and before final handoff, spawn review subagents using model=gpt-5.4-mini. Use at least:
  1. a migration-scope reviewer checking that no collaborator-dependent decision was made;
  2. a docs/MkDocs reviewer checking links, nav impact, generated-file boundaries, and validation gaps.
- Review subagents are read-only. They should return concise findings with file references and severity. The main agent should address valid findings or explain why they are deferred.
- If prompt-level model pinning is unavailable for subagents in this session, state that limitation before review and use the closest available reviewer configuration.

Done when:
- First, produce a prioritized multi-agent execution plan and wait for my approval.
- After approval, complete the approved independent migration workstreams.
- Identify the exact files changed and which agent/workstream changed them.
- Run git diff --check and the relevant MkDocs build command, normally conda run -n qmcpy python -m mkdocs build.
- Run any narrower checks required by changed generated files, such as python scripts/make_qmc_software_page.py when data/qmc-software.yml changes.
- Summarize review-agent findings and how they were handled.
- List any assumptions and any decisions intentionally deferred to collaborators.
- Do not commit, push, create a PR, or change hosting/DNS state.
```
