# AI-Assisted Contributions

QMCPy welcomes AI assistance for drafting, refactoring, editing, test scaffolding, and similar support tasks. The human contributor remains fully responsible for the final change — numerical correctness, reproducibility, licensing, citations, and approval before merge.

## Core Policy

- Use AI in ways that help you understand and improve your change. Review and understand every AI-assisted change before committing it.
- Do not treat AI output as authoritative for mathematics, algorithms, references, benchmark claims, or API behavior. Independently verify equations, algorithm descriptions, stopping criteria, complexity claims, citations, and benchmark interpretations before merge.
- Hold AI-assisted changes to the same standards for tests, docstrings, notebooks, and validation evidence as hand-written changes.

## Prohibited Uses

- Do not commit unverified AI-generated citations, equations, benchmark claims, or other technical assertions.
- Do not paste secrets, credentials, private datasets, unpublished manuscripts, reviewer-confidential material, or other nonpublic information into external AI tools without prior approval from the maintainers ([qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com)).
- If a PR contains code, documentation, figures, or other content intended for a publication whose review rules prohibit AI use by reviewers, the PR creator must state that restriction clearly in the PR, and reviewers of that PR must not use AI tools.

## Required Pull Request Disclosure

If AI assistance substantively influenced code, tests, documentation, mathematical exposition, benchmarks, or the PR text itself:

- Disclose that use in the PR description via the pull request template checklist.
- Briefly summarize which parts were AI-assisted and what you independently verified.

Routine autocomplete and spelling or grammar fixes do not require disclosure.

## Filling Out the PR Template

The template gives reviewers a fast summary of scope, verification, and AI use. Keep entries short and concrete.

| Field | What to write | Example |
|---|---|---|
| `Issue` | Link the issue, or say why none was needed | Fixes `#742` |
| `Numerical or API impact` | Whether algorithms, numerical behavior, or public interfaces changed | Adds optional keyword `seed`; backward-compatible API expansion |
| `Commands run` | The exact checks you ran locally | `pytest test/fasttests/test_halton.py -q` |
| `AI tools and affected areas` | The tool and the parts of the PR it influenced | Copilot suggested a refactor in `qmcpy/stopping_criterion/foo.py` and a test skeleton in `test/foo/test_bar.py` |
| `Independent verification performed` | What you personally checked instead of trusting the AI output | Reviewed the refactor line by line, re-checked the equation against the cited paper, and ran the fast tests |
| `Publication-related AI review restriction` | Whether associated publication rules prohibit AI-assisted review | None — or: supports a manuscript under double-blind review at venue X; please review without AI tools |

If no substantive AI assistance was used, check the first box in the `AI Assistance` section and leave the rest blank or write `None`.

## Reproducibility and Provenance

- Regenerate plots, tables, examples, and derived outputs from committed source code rather than committing unverifiable AI-generated artifacts.
- Keep deterministic seeds, tolerances, and commands explicit when AI-assisted changes affect tests, demos, or performance claims.
- Check AI-assisted code and text for licensing or provenance concerns before including it in the repository.

## Review Expectations

- Reviewers may ask contributors to explain or remove AI-assisted content that is unverifiable, overly broad, or insufficiently understood.
- Public API changes, numerical method changes, dependency changes, and documentation claims receive the same scrutiny whether or not AI was used.
- When in doubt, prefer smaller PRs with clear tests and explicit rationale over large generated diffs.
