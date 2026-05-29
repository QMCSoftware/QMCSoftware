# CI Testing

This page summarizes QMCPy's current GitHub Actions test layout and the cost-control rules behind it.

## Workflows

| Workflow | Trigger | Runner / Python | Main work |
|---|---|---|---|
| `alltests.yml` | Feature-branch `push` | `ubuntu-latest`, Python `3.13` | Non-Docker doctests, `unittests`, and coverage upload |
| `alltests.yml` | `push` to `develop` or `master`; PR into `develop` or `master`; `workflow_dispatch` | `ubuntu-latest`, `macos-latest`, `windows-latest`, Python `3.13` | Full sweep: doctests, `unittests`, coverage upload, booktests on all three OSes, and Linux-only UMBridge doctests when Docker is available |
| `unittests.yml` | `push` to `develop` or `master`; PR into `develop` or `master`; `workflow_dispatch` | `macos-latest`, `ubuntu-latest`, `windows-latest`; jobs are labeled with Python `3.5/3.8/3.11/3.14`, `3.6/3.9/3.12`, and `3.7/3.10/3.13` | Installs test and optional extras, then runs `unittests`; `test/booktests` stay excluded because the Makefile target ignores them |
| `docs.yml` | `push` to `master` | `ubuntu-latest`, Python `3.x` | `uml`, `copydocs`, and `mkdocs gh-deploy --force` |
| `pep8.yml` | `push` to `master`, `develop`; `workflow_dispatch` | `ubuntu-latest`, Python `3.13` | `check_pep8`, then open a badge-update pull request if badge assets change |
| `pypi-stats.yml` | Weekly schedule; `workflow_dispatch` | `ubuntu-latest`, Python `3.13` | Regenerate and publish PyPI download statistics |

Note: `unittests.yml` currently defines a Python-version matrix in job metadata, but the `setup-miniconda` step does not explicitly apply `matrix.python-version`.

There is no nightly CI schedule.

## Policy

- Linux is the default feedback path and runs on every push.
- macOS and Windows in `alltests.yml` are reserved for `develop`/`master` pushes, pull requests into those branches, and manual runs.
- Booktests are skipped on feature-branch pushes and run only in the full sweep.
- UMBridge doctests run only on Linux full sweeps with Docker available.
- `concurrency` cancels superseded runs for both `alltests.yml` and `unittests.yml`.

## Related Docs

- [tests.md](tests.md): local Makefile targets and coverage commands.
- [booktests.md](booktests.md): notebook-test mechanics and developer commands.

When workflow files change, update this page together with `mkdocs.yml`, `README.md`, and [tests.md](tests.md) if local guidance changes.
