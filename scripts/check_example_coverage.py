#!/usr/bin/env python3
"""Fail when public plotting APIs are missing from examples/docs."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYTHON_EXAMPLE_PATHS = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "notebooks").glob("*.ipynb")),
]

WEB_EXAMPLE_PATHS = [
    ROOT / "README.md",
    ROOT / "packages/web/README.md",
    ROOT / "public/demo.js",
    ROOT / "app/page.jsx",
    ROOT / "docs/WEB.md",
    ROOT / "docs/WEBAPP_INTEGRATION.md",
]

PYTHON_IGNORE = {
    # Internal lifecycle is still documented, but requiring every notebook to
    # execute close() would hide the rendered widget.
}

WEB_IGNORE = {
    # Internal helper intentionally hidden from the public examples.
}


def read_text(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def python_plot_methods() -> list[str]:
    module = ast.parse((ROOT / "nbimplot/_plot.py").read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Plot":
            return [
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
            ]
    raise RuntimeError("Could not find nbimplot Plot class.")


def web_plot_methods() -> list[str]:
    source = (ROOT / "packages/web/src/index.js").read_text()
    start = source.index("export class WebPlot")
    end = source.index("\n}\n\nexport async function createPlot", start)
    block = source[start:end]
    return [
        match.group(1)
        for match in re.finditer(r"^  ([A-Za-z_$][\w$]*)\(", block, re.MULTILINE)
        if not match.group(1).startswith("_") and match.group(1) != "constructor"
    ]


def snake_to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def canonical_web_methods(methods: list[str]) -> list[str]:
    method_set = set(methods)
    canonical: list[str] = []
    for method in methods:
        if "_" in method and snake_to_camel(method) in method_set:
            continue
        canonical.append(method)
    return canonical


def is_covered(text: str, method: str) -> bool:
    pattern = re.compile(rf"\.\s*{re.escape(method)}\s*\(")
    return bool(pattern.search(text))


def report_missing(label: str, methods: list[str], ignore: set[str], text: str) -> list[str]:
    missing = [method for method in methods if method not in ignore and not is_covered(text, method)]
    if missing:
        print(f"{label} example coverage missing:")
        for method in missing:
            print(f"  - {method}")
    else:
        print(f"{label} example coverage OK ({len(methods) - len(ignore)} methods checked).")
    return missing


def main() -> int:
    python_text = read_text(PYTHON_EXAMPLE_PATHS)
    web_text = read_text(WEB_EXAMPLE_PATHS)

    py_missing = report_missing(
        "Python Plot",
        python_plot_methods(),
        PYTHON_IGNORE,
        python_text,
    )
    web_missing = report_missing(
        "WebPlot",
        canonical_web_methods(web_plot_methods()),
        WEB_IGNORE,
        web_text,
    )
    return 1 if py_missing or web_missing else 0


if __name__ == "__main__":
    sys.exit(main())
