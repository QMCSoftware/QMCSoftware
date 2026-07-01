# snippets/AGENTS.md

## Scope

This guidance applies to reusable Markdown snippets under `snippets/`, including
author and blog-author snippets.

## Independent subtask focus

Personal-developer work here should focus on:

- Keeping author snippets consistent with existing naming and slug conventions.
- Updating blog-author snippets when a migrated blog already has verified
  authorship.
- Removing duplication by referencing snippets from content pages when the
  existing project pattern supports it.

Do not add affiliations, roles, emails, biographies, or contributor status from
memory. Use only repository-visible sources or explicitly provided user input.

## Editing rules

- Preserve exact spelling of names already used in the repository.
- Keep snippets short and reusable.
- Do not use snippets to decide collaborator governance or retirement questions.
- If a snippet change affects a blog page, verify the page renders through
  MkDocs.

## Verification

Run:

- `git diff --check`
- `conda run -n qmcpy python -m mkdocs build`
