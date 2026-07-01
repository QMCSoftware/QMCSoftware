# docs/AGENTS.md

## Scope

This guidance applies to MkDocs pages under `docs/` and to site navigation work
that publishes those pages through `mkdocs.yml`.

## Independent subtask focus

Personal-developer work in this subtree should focus on:

- Creating or improving GitHub Pages targets for already-confirmed
  must-preserve content.
- Improving navigation labels and grouping when the change is reversible and
  does not remove content.
- Fixing broken internal links, Markdown issues, and stale references to moved
  documentation.
- Adding a Publications or references target only from already-published,
  repository-visible sources such as `README.md`, `community.md`, `cite_qmcpy.bib`,
  existing docs, or recorded inventory notes.

Leave policy decisions for collaborators. Do not decide that a WordPress page,
post, section, or external backlink can be retired.

## MkDocs rules

- Keep page paths stable once published.
- Add new pages to `mkdocs.yml` only when they should be visible in the public
  site navigation.
- Do not hand-edit generated pages when a source data file or script owns them.
  In particular, `docs/qmc-software.md` is generated from
  `data/qmc-software.yml`.
- Keep headings concise and useful for navigation.
- Preserve mathematical notation and citations. Do not simplify technical
  statements by changing their meaning.

## Link rules

- Prefer relative links for pages inside the same documentation site.
- Use absolute links for external resources such as PyPI, GitHub, DOI, JOSS,
  YouTube, or old `qmcpy.org` source URLs.
- Old `qmcpy.org` links should either be intentional historical/source links or
  recorded redirect targets. Do not remove them just because they are old.

## Verification

After editing docs or `mkdocs.yml`, run:

- `git diff --check`
- `conda run -n qmcpy python -m mkdocs build`

If the conda environment is unavailable, try `python -m mkdocs build` and report
the failure clearly.
