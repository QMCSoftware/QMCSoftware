#!/usr/bin/env python3
"""Emit pytest xdist argument if available.

Prints "-n auto" if the `xdist` plugin is importable, otherwise prints nothing.
This keeps Makefile recipes portable and avoids here-doc issues.
"""
import importlib.util
import sys

if importlib.util.find_spec("xdist"):
    sys.stdout.write("-n auto")
