# data/AGENTS.md

## Scope

This guidance applies to structured data files under `data/`, especially
`data/qmc-software.yml`, which generates the QMC software ecosystem page.

## Independent subtask focus

Personal-developer work here should focus on:

- Correcting obvious formatting or URL issues in existing entries.
- Adding or updating QMC software entries only from verified public sources.
- Regenerating `docs/qmc-software.md` from the data source when the data changes.

Do not add private contact details, unverified project status, or subjective
rankings. Do not remove software entries unless the user explicitly asks or the
entry is a clear duplicate and the safer change is obvious.

## Data rules

- Keep YAML structure consistent with existing entries.
- Use plain public project URLs when possible.
- Use `mailto:` links only when the address is already present in the repository
  or explicitly supplied by the user.
- Keep descriptions factual and concise.

## Regeneration and verification

After changing `data/qmc-software.yml`, run:

- `python scripts/make_qmc_software_page.py`
- `git diff --check`
- `conda run -n qmcpy python -m mkdocs build`

If `python scripts/make_qmc_software_page.py` changes
`docs/qmc-software.md`, keep both files in the same change.
