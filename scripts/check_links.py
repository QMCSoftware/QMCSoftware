#!/usr/bin/env python3
"""Check for broken links in a built MkDocs site.

Crawls the static HTML output of `mkdocs build` and reports:
- internal links/anchors that don't resolve to a page or heading in the site
  (always checked; these are fully within our control and any failure here
  is a real build defect)
- external links that return an error or fail to connect (only checked with
  --external, since third-party sites rate-limit, gate behind auth walls, or
  have transient outages that should not fail a build)

Usage:
    mkdocs build -d site
    python scripts/check_links.py site
    python scripts/check_links.py site --external
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

HREF_RE = re.compile(r'href=["\']([^"\'#][^"\']*)["\']')
# Only <a href="...">, not <link href="..."> (stylesheets, preconnect resource
# hints, icons, ...) -- those aren't links a reader can click and follow.
A_HREF_RE = re.compile(r'<a\s[^>]*href=["\']([^"\'#][^"\']*)["\']', re.IGNORECASE)
ID_RE = re.compile(r'id=["\']([^"\']+)["\']')


def _path_for_url_target(site_dir: Path, target: str) -> Path | None:
    """Map a site-relative href to the file it should resolve to, mirroring
    MkDocs's `use_directory_urls` output layout (page.md -> page/index.html)."""
    clean = target.split("#", 1)[0].split("?", 1)[0]
    if not clean or clean == ".":
        return None
    # site_dir is already resolved by the caller, so .resolve() here only
    # normalizes "../" segments -- it does not re-translate any symlink.
    candidate = (site_dir / clean.lstrip("/")).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    elif candidate.suffix == "":
        alt = candidate.parent / (candidate.name + "/index.html")
        if alt.exists():
            candidate = alt
    return candidate


def check_internal(site_dir: Path) -> list[str]:
    # Resolve once so every path derived from site_dir (including each html
    # file's .parent used as the base for relative links) shares one
    # convention -- otherwise a symlinked path (e.g. macOS /tmp -> /private/tmp)
    # makes filesystem-equal paths compare unequal as dict keys.
    site_dir = site_dir.resolve()
    problems = []
    html_files = sorted(site_dir.rglob("*.html"))
    if not html_files:
        return [f"no .html files found under {site_dir} -- did `mkdocs build -d {site_dir}` run first?"]

    anchors_by_file: dict[Path, set[str]] = {}
    for f in html_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        anchors_by_file[f] = set(ID_RE.findall(text))

    for f in html_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for href in HREF_RE.findall(text):
            # Only same-origin, scheme-less hrefs are filesystem paths to check;
            # this also skips data:, mailto:, tel:, javascript:, blob:, etc.
            if urlparse(href).scheme != "" or href.startswith("//"):
                continue
            target_path = _path_for_url_target(f.parent, href)
            if target_path is None:
                continue
            try:
                exists = target_path.exists()
            except OSError as e:
                problems.append(f"{f.relative_to(site_dir)}: unresolvable link '{href[:80]}...' ({e})")
                continue
            if not exists:
                problems.append(f"{f.relative_to(site_dir)}: broken internal link '{href}'")
                continue
            if "#" in href:
                anchor = href.split("#", 1)[1]
                if anchor and anchor not in anchors_by_file.get(target_path, set()):
                    problems.append(
                        f"{f.relative_to(site_dir)}: link '{href}' has no matching "
                        f"id='{anchor}' on the target page"
                    )
    return problems


def check_external(site_dir: Path, delay: float = 0.1, timeout: float = 8.0) -> list[str]:
    links: dict[str, list[Path]] = {}
    for f in sorted(site_dir.rglob("*.html")):
        text = f.read_text(encoding="utf-8", errors="replace")
        for href in A_HREF_RE.findall(text):
            if urlparse(href).scheme in ("http", "https"):
                links.setdefault(href, []).append(f.relative_to(site_dir))

    problems = []
    for url in sorted(links):
        req = urllib.request.Request(url, headers={"User-Agent": "qmcpy-link-checker"}, method="HEAD")
        try:
            urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (405, 403):
                # Some hosts reject HEAD; retry with GET before giving up.
                try:
                    urllib.request.urlopen(
                        urllib.request.Request(url, headers={"User-Agent": "qmcpy-link-checker"}),
                        timeout=timeout,
                    )
                except Exception as e2:
                    problems.append(f"{url} -- {e2} (seen on {links[url][0]})")
            elif e.code >= 400:
                problems.append(f"{url} -- HTTP {e.code} (seen on {links[url][0]})")
        except Exception as e:
            problems.append(f"{url} -- {e} (seen on {links[url][0]})")
        time.sleep(delay)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", nargs="?", default="site", help="MkDocs build output directory")
    parser.add_argument("--external", action="store_true", help="also check external (http/https) links")
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    problems = check_internal(site_dir)
    print(f"Checked internal links under {site_dir}: {len(problems)} problem(s).")
    for p in problems:
        print(f"  {p}")

    if args.external:
        ext_problems = check_external(site_dir)
        print(f"\nChecked external links: {len(ext_problems)} problem(s).")
        for p in ext_problems:
            print(f"  {p}")
        problems += ext_problems

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
