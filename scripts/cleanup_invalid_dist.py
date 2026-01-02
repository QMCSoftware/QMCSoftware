#!/usr/bin/env python3
"""
Cleanup script to remove invalid distribution artifacts (e.g. '~eaborn') from site-packages.

Usage:
  python scripts/cleanup_invalid_dist.py        # dry run (lists candidates)
  python scripts/cleanup_invalid_dist.py --apply --reinstall   # remove candidates and reinstall seaborn
"""

import sys
import sysconfig
import argparse
import shutil
from pathlib import Path
import subprocess
import site


def get_site_paths():
    paths = set()
    try:
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            paths.add(Path(purelib))
    except Exception:
        # purelib unavailable in this environment; skip
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            paths.add(Path(user_site))
    except Exception:
        # user site-packages unavailable; skip
        pass
    for p in sys.path:
        try:
            pp = Path(p)
            if "site-packages" in str(pp):
                paths.add(pp)
        except Exception:
            # invalid path; skip
            pass
    return [p for p in paths if p.exists()]


def find_invalid_dists(site_paths):
    candidates = []
    for sp in site_paths:
        try:
            for entry in sp.iterdir():
                name = entry.name
                # Heuristic: entries that start with '~' are invalid artifacts like "~eaborn..."
                if name.startswith("~"):
                    candidates.append(entry)
                    continue
                # Also include entries that contain the corrupted fragment 'eaborn'
                if "eaborn" in name:
                    candidates.append(entry)
                    continue
                lower = name.lower()
                if (lower.endswith(".dist-info") or lower.endswith(".egg-info")) and ("~" in name or "eaborn" in name):
                    candidates.append(entry)
        except PermissionError:
            continue
    # deduplicate and sort
    unique = sorted(set(candidates), key=str)
    return unique


def remove_paths(paths):
    removed = []
    errors = []
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(p)
        except Exception as e:
            errors.append((p, str(e)))
    return removed, errors


def reinstall_seaborn():
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "seaborn"]
    return subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="Find and remove invalid distribution artifacts (e.g. ~eaborn) in this Python environment.")
    parser.add_argument("--apply", action="store_true", help="Actually remove files. Without this flag the script only lists candidates.")
    parser.add_argument("--reinstall", action="store_true", help="When used with --apply, reinstall seaborn after removal.")
    args = parser.parse_args()

    site_paths = get_site_paths()
    if not site_paths:
        print("No site-packages paths found for this interpreter.", file=sys.stderr)
        sys.exit(1)

    print("Scanning site-packages paths:")
    for p in site_paths:
        print("  -", p)

    candidates = find_invalid_dists(site_paths)
    if not candidates:
        print("No invalid-distribution candidates found.")
        return

    print("\nCandidates found:")
    for c in candidates:
        print("  -", c)

    if not args.apply:
        print("\nDry run. Re-run with --apply to remove these candidates.")
        return

    print("\nRemoving candidates...")
    removed, errors = remove_paths(candidates)
    for r in removed:
        print("Removed:", r)
    for p, err in errors:
        print("ERROR removing", p, ":", err, file=sys.stderr)

    if args.reinstall:
        print("\nReinstalling seaborn...")
        ret = reinstall_seaborn()
        if ret.returncode == 0:
            print("seaborn reinstalled successfully.")
        else:
            print("pip returned non-zero exit code:", ret.returncode, file=sys.stderr)
            sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
