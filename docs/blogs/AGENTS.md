# docs/blogs/AGENTS.md

## Scope

This guidance applies to migrated blog posts under `docs/blogs/`.

## Independent subtask focus

Personal-developer work here should focus on:

- Preserving already-migrated WordPress blog posts.
- Adding missing `Source WordPress URL:` metadata when the exact source URL is
  already listed in `MIGRATE_qmcpy_inventory.md`.
- Fixing Markdown, image references, internal links, and navigation labels.
- Recording old-to-new URL mappings for posts that already have clear GitHub
  Pages targets.

Do not decide that old news, event, announcement, draft, Donation, Videos, or
Dev Tools content should be migrated or retired without explicit user or
collaborator direction.

## Blog migration rules

- Preserve the scientific and historical meaning of each post.
- Keep original WordPress URL metadata near the top of the file when present.
- When an original image URL is recorded in an HTML comment, keep it unless the
  task explicitly replaces it with a verified local asset.
- Do not rewrite author attribution unless the source is already verified in the
  repository or inventory.
- If a blog post needs shared author metadata, update the matching file under
  `snippets/blog-authors/` instead of duplicating large author blocks.

## URL preservation checks

When changing old-domain links or source metadata, compare against
`MIGRATE_qmcpy_inventory.md`. Prefer exact old URL matches over title-only
matches.

Useful checks:

- `rg -n "Source WordPress URL|qmcpy.org/" docs/blogs MIGRATE_qmcpy_inventory.md`
- `git diff --check`
- `conda run -n qmcpy python -m mkdocs build`
