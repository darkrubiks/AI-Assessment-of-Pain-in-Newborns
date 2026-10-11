#!/usr/bin/env python3
"""Check or explicitly clear saved notebook results without executing cells.

Run --check for tracked notebooks, --check --staged for the Git index, or
--strip to clear tracked working-tree notebooks. Optional paths limit a
working-tree operation. Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


class NotebookError(ValueError):
    """A notebook cannot safely be inspected or rewritten."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NotebookError("duplicate JSON keys")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise NotebookError("non-finite JSON value")


def parse_notebook(data: bytes) -> dict[str, Any]:
    """Parse a version-4 notebook; errors never expose notebook contents."""
    try:
        doc = json.loads(data, object_pairs_hook=_unique_object,
                         parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, NotebookError):
            raise
        raise NotebookError("invalid notebook JSON") from None
    if not isinstance(doc, dict) or doc.get("nbformat") != 4:
        raise NotebookError("unsupported notebook version; expected version 4")
    if (not isinstance(doc.get("nbformat_minor"), int)
            or not isinstance(doc.get("metadata"), dict)
            or not isinstance(doc.get("cells"), list)):
        raise NotebookError("invalid notebook structure")
    for cell in doc["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {
            "code", "markdown", "raw"
        }:
            raise NotebookError("invalid cell structure")
        source = cell.get("source")
        if not (isinstance(source, str) or (
            isinstance(source, list) and all(isinstance(s, str) for s in source)
        )):
            raise NotebookError("invalid cell source")
        if not isinstance(cell.get("metadata"), dict):
            raise NotebookError("invalid cell metadata")
        if "attachments" in cell and not isinstance(cell["attachments"], dict):
            raise NotebookError("invalid attachments")
        if "outputs" in cell and not isinstance(cell["outputs"], list):
            raise NotebookError("invalid outputs")
        if cell["cell_type"] == "code":
            if "outputs" not in cell or "execution_count" not in cell:
                raise NotebookError("missing code-cell results fields")
            count = cell["execution_count"]
            if count is not None and (not isinstance(count, int)
                                      or isinstance(count, bool)):
                raise NotebookError("invalid execution count")
    return doc


def result_fields(doc: dict[str, Any]) -> list[str]:
    """Return saved-result categories, never their values."""
    fields: set[str] = set()
    if "widgets" in doc["metadata"]:
        fields.add("widgets")
    for cell in doc["cells"]:
        if cell.get("outputs"):
            fields.add("outputs")
        if cell.get("execution_count") is not None:
            fields.add("execution_count")
        if "attachments" in cell:
            fields.add("attachments")
        if "widgets" in cell["metadata"]:
            fields.add("widgets")
    return sorted(fields)


def strip_notebook(data: bytes) -> bytes:
    """Clear result fields, preserving unchanged bytes and all source fields."""
    doc = parse_notebook(data)
    if not result_fields(doc):
        return data
    doc["metadata"].pop("widgets", None)
    for cell in doc["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        cell.pop("attachments", None)
        cell["metadata"].pop("widgets", None)
    return (json.dumps(doc, ensure_ascii=False, indent=1, allow_nan=False)
            + "\n").encode("utf-8")


def _git(*args: str, cwd: Path) -> bytes:
    proc = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise NotebookError("Git operation failed; inspect repository/index state")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--strip", action="store_true")
    parser.add_argument("--staged", action="store_true",
                        help="check changed notebook blobs in the index")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    if args.staged and (args.strip or args.paths):
        parser.error("--staged requires --check and no explicit paths")
    try:
        root = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd())
                    .decode().strip())
        if args.paths:
            paths = [Path(p).resolve() for p in args.paths]
        else:
            command = (["diff", "--cached", "--name-only", "-z",
                        "--diff-filter=ACMR"] if args.staged else ["ls-files", "-z"])
            names = _git(*command, "--", "*.ipynb", cwd=root).decode(
                "utf-8", errors="surrogateescape").split("\0")
            paths = [root / name for name in names if name]
        changes: list[tuple[Path, bytes]] = []
        failures = 0
        for path in paths:
            try:
                name = path.relative_to(root).as_posix()
                if path.suffix.lower() != ".ipynb":
                    raise NotebookError("expected an .ipynb path")
                data = (_git("show", ":" + name, cwd=root) if args.staged
                        else path.read_bytes())
                doc = parse_notebook(data)
                fields = result_fields(doc)
                if fields and args.check:
                    print(f"{name}: saved " + ", ".join(fields))
                    failures += 1
                elif fields:
                    changes.append((path, strip_notebook(data)))
            except (NotebookError, OSError, ValueError) as exc:
                reason = str(exc) if isinstance(exc, NotebookError) else "unreadable notebook path"
                print(f"{path.name}: {reason}")
                failures += 1
        if failures:
            return 1
        # Validate every input before changing any working-tree file.
        for path, data in changes:
            path.write_bytes(data)
            print(f"{path.relative_to(root).as_posix()}: cleared saved results")
        return 0
    except (NotebookError, OSError):
        print("Unable to inspect Git repository")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
