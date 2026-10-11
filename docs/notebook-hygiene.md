# Notebook results and repository privacy

Commit notebooks with empty outputs and null execution counts. Notebook attachments
and saved widget state are also excluded because they may contain dataset images.
The checker uses Python's standard library and never executes notebooks.

From the repository root:

```bash
python scripts/notebook_hygiene.py --check
python scripts/notebook_hygiene.py --strip
git add -- path/to/notebook.ipynb
python scripts/notebook_hygiene.py --check --staged
```

Optional paths limit working-tree checks or clearing, for example:

```bash
python scripts/notebook_hygiene.py --strip "path/to/notebook.ipynb"
```

Clearing preserves code, Markdown source, cell IDs and other metadata. References
to removed attachments can remain in Markdown; replace them with permitted local
resources if needed. Modified JSON may be reformatted. Unsupported versions or
malformed notebooks cause a failure before any working-tree notebook is changed.

To enable the supplied pre-commit hook:

```bash
git config core.hooksPath .githooks
```

The hook checks the staged notebook contents, including partially staged files.
It rejects saved results rather than clearing them automatically. Set the
`NOTEBOOK_HYGIENE_PYTHON` environment variable to your Python executable if needed.
Existing custom hooks should call the checker instead of replacing their hook setup.
GitHub Actions runs the same check and synthetic tests on pushes and pull requests.
CI detects violations after a push; enable the local hook to catch them before commits.

## After a history rewrite

Existing commit IDs change. Prefer a fresh clone of the rewritten repository.
Preserve uncommitted work privately before replacing an old clone. Do not merge
or push old branches into cleaned history: that can reintroduce removed images.
Transfer necessary changes only after checking their notebooks and ancestry.

A rewritten Git branch cannot remove copies held by other people, forks,
read-only pull-request references or GitHub caches. Sensitive historical images
may require GitHub Support assistance:
[Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

Dataset images, videos, checkpoints and restricted outputs must remain in
authorized local storage. Git ignore rules and output checks do not provide
access control or a complete scan of source text for sensitive information.
