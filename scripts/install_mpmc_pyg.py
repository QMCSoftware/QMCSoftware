#!/usr/bin/env python
"""Install the PyG runtime needed by QMCPy's MPMC tests."""

from __future__ import annotations

import re
import subprocess
import sys


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(list(args))


def torch_versions() -> list[str]:
    import torch

    match = re.match(r"(\d+\.\d+\.\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"Unable to parse torch version: {torch.__version__}")

    full = match.group(1)
    major, minor, _ = full.split(".")
    versions = [full]
    fallback = f"{major}.{minor}.0"
    if fallback != full:
        versions.append(fallback)
    return versions


def main() -> None:
    import torch

    print(f"Detected torch {torch.__version__}", flush=True)
    run(sys.executable, "-m", "pip", "install", "--prefer-binary", "torch-geometric>=2.6.1")

    last_error = None
    for torch_version in torch_versions():
        wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+cpu.html"
        print(f"Trying pyg_lib wheels from {wheel_url}", flush=True)
        try:
            run(
                sys.executable,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "--only-binary",
                "pyg_lib",
                "pyg_lib>=0.6.0",
                "-f",
                wheel_url,
            )
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to install pyg_lib for the current torch build. "
        "PyG wheels at data.pyg.org may not yet support this torch release. "
        "Pin torch to a supported version (for example, < 2.13) and retry."
    ) from last_error


if __name__ == "__main__":
    main()
