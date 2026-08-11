# QMCSoftware Agent Guidance

## Purpose and authority

This file is a concise entry point for coding agents. It does not replace the
repository's contributor policies. Before substantive work, read and follow:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) for issues, branches, pull requests,
  setup, and validation;
- [`docs/good_practices.md`](docs/good_practices.md) for implementation,
  testing, documentation, and review expectations; and
- [`docs/ai-assisted-contributions.md`](docs/ai-assisted-contributions.md) for
  required verification and pull-request disclosure.

## Required workflow

- Inspect `git status` and preserve unrelated work before editing.
- Connect each change to a GitHub issue.
- Synchronize with `origin/develop`, create a focused feature branch from
  `develop`, and open a pull request back into `develop`.
- Do not commit directly to `develop` or `master`, and never force-push shared
  branches.
- Keep the change scoped to its issue. Do not modify neighboring repositories
  unless the user separately authorizes work there.
- Use the GitHub issue and pull request as the operational handoff for parallel
  work. Do not introduce repository-wide task or status files without
  maintainer agreement.
- Disclose substantive AI assistance in the pull-request template and state
  what was independently verified.

## Validation and generated content

- Run the smallest relevant checks described in the contributor guides, plus
  broader tests when the change affects shared behavior.
- Run `git diff --check` and review the complete diff before committing.
- When documentation is affected, build or render the relevant documentation
  and inspect the result.
- Do not edit generated files as if they were authoritative source. Keep source
  data, generators, committed outputs, and build instructions synchronized as
  required by the existing workflow.
- Never bypass failing tests or review safeguards merely to complete a task.

## Repository boundaries

QMCSoftware contains QMCPy source, tests, demos, papers, and technical
documentation. Organization-wide website content belongs in the separate
`qmcsoftware-website` repository. Coordinate cross-repository changes as
separate, explicitly scoped tasks and publish dependencies before updating
their consumers.
