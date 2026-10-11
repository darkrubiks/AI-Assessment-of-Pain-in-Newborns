"""Synthetic, dataset-free checks of notebook clearing and staged-content guards."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("notebook_hygiene",
                                            ROOT / "scripts/notebook_hygiene.py")
hygiene = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hygiene)


def notebook() -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3"}, "widgets": {"synthetic": {}}},
        "cells": [
            {"cell_type": "code", "id": "synthetic-code", "metadata": {"keep": 1},
             "source": ["print('synthetic')\n"], "execution_count": 7,
             "outputs": [
                 {"output_type": "stream", "name": "stdout", "text": ["synthetic\n"]},
                 {"output_type": "display_data", "metadata": {},
                  "data": {"image/png": ["SYNTHETIC"], "text/plain": ["synthetic"]}}
             ]},
            {"cell_type": "markdown", "id": "synthetic-markdown",
             "metadata": {"keep": True, "widgets": {"synthetic": {}}},
             "source": ["![synthetic](attachment:figure.png)"],
             "attachments": {"figure.png": {"image/png": "SYNTHETIC"}}}
        ]
    }


def encoded(doc: dict) -> bytes:
    return json.dumps(doc).encode()


class NotebookHygieneTests(unittest.TestCase):
    def test_clears_all_results_and_preserves_source(self):
        original = notebook()
        expected = copy.deepcopy(original)
        expected["metadata"].pop("widgets")
        expected["cells"][0]["outputs"] = []
        expected["cells"][0]["execution_count"] = None
        expected["cells"][1].pop("attachments")
        expected["cells"][1]["metadata"].pop("widgets")
        cleaned = hygiene.strip_notebook(encoded(original))
        self.assertEqual(json.loads(cleaned), expected)
        self.assertEqual(hygiene.result_fields(json.loads(cleaned)), [])
        self.assertEqual(hygiene.strip_notebook(cleaned), cleaned)

    def test_clean_bytes_are_unchanged(self):
        clean = hygiene.strip_notebook(encoded(notebook()))
        unusual = json.dumps(json.loads(clean), indent=4).encode()
        self.assertEqual(hygiene.strip_notebook(unusual), unusual)

    def test_each_result_category_is_rejected(self):
        original = notebook()
        self.assertEqual(hygiene.result_fields(original),
                         ["attachments", "execution_count", "outputs", "widgets"])
        clean = json.loads(hygiene.strip_notebook(encoded(original)))
        for field in ["outputs", "execution_count", "attachments", "widgets"]:
            with self.subTest(field=field):
                changed = copy.deepcopy(clean)
                if field == "widgets":
                    changed["metadata"]["widgets"] = {}
                elif field == "attachments":
                    changed["cells"][1]["attachments"] = {}
                else:
                    changed["cells"][0][field] = original["cells"][0][field]
                self.assertEqual(hygiene.result_fields(changed), [field])

    def test_malformed_and_unsupported_inputs(self):
        for bad in [b"{", b'{"nbformat": 3}', b'{"nbformat":4,"nbformat":4}',
                    b'{"nbformat":NaN}']:
            with self.subTest(bad=bad):
                with self.assertRaises(hygiene.NotebookError):
                    hygiene.strip_notebook(bad)
        bad = notebook()
        bad["cells"][0]["outputs"] = "invalid"
        with self.assertRaises(hygiene.NotebookError):
            hygiene.strip_notebook(encoded(bad))

    def test_nested_partial_staging_and_deletion(self):
        # Repository instructions require all agent/test temporaries in .agents.
        base = ROOT / ".agents" / "scratch"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="notebook-hygiene-", dir=base) as directory:
            repo = Path(directory)
            def git(*args):
                return subprocess.run(["git", *args], cwd=repo, check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git("init", "-q")
            path = repo / "nested folder" / "synthetic.ipynb"
            path.parent.mkdir()
            dirty = encoded(notebook())
            clean = hygiene.strip_notebook(dirty)
            path.write_bytes(dirty)
            git("add", "--", "nested folder/synthetic.ipynb")
            path.write_bytes(clean)
            previous = Path.cwd()
            try:
                os.chdir(repo)
                self.assertEqual(hygiene.main(["--check", "--staged"]), 1)
                git("add", "--", "nested folder/synthetic.ipynb")
                path.write_bytes(dirty)
                self.assertEqual(hygiene.main(["--check", "--staged"]), 0)
                self.assertEqual(hygiene.main(["--check"]), 1)
                self.assertEqual(hygiene.main(["--strip"]), 0)
                self.assertEqual(path.read_bytes(), clean)
                git("rm", "--cached", "--", "nested folder/synthetic.ipynb")
                self.assertEqual(hygiene.main(["--check", "--staged"]), 0)
            finally:
                os.chdir(previous)

    def test_invalid_input_prevents_partial_strip(self):
        base = ROOT / ".agents" / "scratch"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="notebook-hygiene-", dir=base) as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            first, second = repo / "a.ipynb", repo / "b.ipynb"
            dirty = encoded(notebook())
            first.write_bytes(dirty)
            second.write_bytes(b"{")
            previous = Path.cwd()
            try:
                os.chdir(repo)
                self.assertEqual(hygiene.main(["--strip", str(first), str(second)]), 1)
                self.assertEqual(first.read_bytes(), dirty)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
