#!/usr/bin/env python3
"""Generate badge files from pylint output."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


SCORE_RE = re.compile(r"rated at ([0-9]+\.[0-9]+)/10")


def score_to_color(score_value: float) -> str:
    if score_value >= 9:
        return "brightgreen"
    if score_value >= 8:
        return "yellowgreen"
    return "orange"


def color_to_hex(color: str) -> str:
    mapping = {
        "brightgreen": "#4c1",
        "yellowgreen": "#a4a61d",
        "orange": "#fe7d37",
    }
    return mapping[color]


def estimate_text_width(text: str) -> int:
    # Approximate DejaVu Sans width at 11px, close to shields.io visuals.
    return 10 + 7 * len(text)


def write_json_badge(path: Path, score_text: str, color: str) -> None:
    payload = {
        "schemaVersion": 1,
        "label": "pylint",
        "message": score_text,
        "color": color,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_svg_badge(path: Path, score_text: str, color: str) -> None:
    label = "pylint"
    left_w = estimate_text_width(label)
    right_w = estimate_text_width(score_text)
    total_w = left_w + right_w
    right_fill = color_to_hex(color)
    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_w}\" height=\"20\" role=\"img\" aria-label=\"{label}: {score_text}\">\n<title>{label}: {score_text}</title>\n<linearGradient id=\"s\" x2=\"0\" y2=\"100%\">\n  <stop offset=\"0\" stop-color=\"#fff\" stop-opacity=\".7\"/>\n  <stop offset=\".1\" stop-color=\"#aaa\" stop-opacity=\".1\"/>\n  <stop offset=\".9\" stop-opacity=\".3\"/>\n  <stop offset=\"1\" stop-opacity=\".5\"/>\n</linearGradient>\n<clipPath id=\"r\">\n  <rect width=\"{total_w}\" height=\"20\" rx=\"3\" fill=\"#fff\"/>\n</clipPath>\n<g clip-path=\"url(#r)\">\n  <rect width=\"{left_w}\" height=\"20\" fill=\"#555\"/>\n  <rect x=\"{left_w}\" width=\"{right_w}\" height=\"20\" fill=\"{right_fill}\"/>\n  <rect width=\"{total_w}\" height=\"20\" fill=\"url(#s)\"/>\n</g>\n<g fill=\"#fff\" text-anchor=\"middle\" font-family=\"DejaVu Sans,Verdana,Geneva,sans-serif\" font-size=\"11\">\n  <text x=\"{left_w // 2}\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{label}</text>\n  <text x=\"{left_w // 2}\" y=\"14\">{label}</text>\n  <text x=\"{left_w + right_w // 2}\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{score_text}</text>\n  <text x=\"{left_w + right_w // 2}\" y=\"14\">{score_text}</text>\n</g>\n</svg>\n"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: update_pep8_badge.py INPUT OUTPUT [OUTPUT ...]", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_paths = [Path(arg) for arg in sys.argv[2:]]
    text = input_path.read_text(encoding="utf-8")
    match = SCORE_RE.search(text)
    if not match:
        print(f"could not find pylint score in {input_path}", file=sys.stderr)
        return 1

    score = match.group(1)
    score_value = float(score)
    score_text = f"{score}/10"
    color = score_to_color(score_value)

    for output_path in output_paths:
        suffix = output_path.suffix.lower()
        if suffix == ".json":
            write_json_badge(output_path, score_text, color)
        elif suffix == ".svg":
            write_svg_badge(output_path, score_text, color)
        else:
            print(f"unsupported output extension: {output_path}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
