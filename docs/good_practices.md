# Good Practices for Contributors

This page collects the contribution expectations that help QMCPy stay numerically correct, reproducible, and reviewable. Use it together with the [contributing guide](https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/), the [test targets guide](tests.md), and the [notebook test guide](booktests.md).

## Start from an issue and keep the scope clear

- Connect every bug fix, feature, refactor, or documentation update to an issue.
- Keep each pull request focused on one topic so reviewers can reason about the mathematical and API impact.
- For architectural or mathematically subtle changes, open a draft PR early and schedule a review meeting before merge.

## Tests are required

We expect tests for every change that affects behavior, documentation, or user workflows.

- Add or update **unit tests** in `test/` for new logic, bug fixes, edge cases, invalid inputs, shapes, finite outputs, and meaningful invariants.
- Add or update **doctests** when public docstrings, examples, or usage patterns change.
- Add or update **notebook tests** when a demo or blog notebook changes.
- Use deterministic seeds or deterministic generators in tests and examples.
- Keep tests small enough to run locally and in CI.

Run the smallest relevant checks before requesting review:

```bash
make unittests
make doctests_no_docker
make booktests_no_docker
make tests_fast
```

Use the notebook-focused checks when you touch `demos/` or blog content backed by notebooks. If you add executable Python snippets to Markdown pages under `docs/`, keep those snippets runnable as well.

## Write Google-style docstrings

QMCPy documentation is built from docstrings, so public APIs should document their behavior clearly and consistently.

- Use **Google-style docstrings** for public classes, methods, and functions.
- Document parameters, return values, shapes, assumptions, and any stochastic behavior.
- Include short doctestable examples when they clarify expected use.
- Update docstrings at the same time as the implementation so the rendered API docs do not drift from the code.

## Extend the existing object model

New functionality should fit the existing QMCPy class hierarchy instead of introducing parallel designs without discussion.

- Inherit from the closest existing QMCPy abstract or base class rather than directly from `object`.
- Reuse established interfaces and field names where possible.
- Typical extension points include `DiscreteDistribution`, `TrueMeasure`, `Integrand`, `StoppingCriterion`, and `AccumulateData`.
- If a change does not fit the current hierarchy, raise that design question in an issue or draft PR before committing to a new abstraction.

The [components overview](components.md) and the blog post on [object classes in QMCPy](blogs/visualizing-the-internals-of-object-classes-in-qmcpy/index.md) provide useful background on the current architecture.

## Add demos or blogs as notebooks

User-facing methods, new workflows, and mathematically important additions should usually come with an executable notebook.

- Put demos and tutorials in `.ipynb` files under `demos/`.
- If a contribution is best explained as a blog post, keep the blog content backed by a notebook when practical.
- Keep notebooks lightweight, deterministic, and suitable for docs rendering and CI.
- Include the mathematical rationale, key assumptions, validation evidence, and a minimal example.

## Requesting review

Request review when the contribution is ready for technical evaluation, not while core pieces are still missing.

- Open a **draft PR** if you want early feedback on design, mathematics, or scope.
- Request formal review only after the relevant tests pass locally and the required docstrings, docs, and notebooks are in place.
- Summarize the numerical goal, API impact, issue link, and commands you ran in the PR description.
- Call out any remaining risks, approximations, or open questions explicitly.
- For complex mathematical changes, ask for a review meeting in addition to GitHub review comments.

## Requesting re-review

Re-request review when you have addressed prior comments and the branch is ready for another pass.

- After substantial updates, post a short summary of what changed since the last review.
- Re-run the relevant tests after addressing review feedback, especially if behavior or interfaces changed.
- Re-request review from the same reviewers when their previous concerns have been addressed.
- If new commits materially change the design or numerical behavior, mention that directly so reviewers know to re-check the affected areas.

## Before merge

- Ensure required reviews are complete.
- Ensure CI is green for the relevant jobs.
- Make sure docs, tests, and notebooks changed together when the contribution changed public behavior.
- Delete the feature branch after a successful merge.
