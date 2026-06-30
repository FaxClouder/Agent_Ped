# Contributing

Ped-Agent uses a small, conventional Git workflow:

1. Create a branch from `main`.
2. Keep changes focused on one feature or fix.
3. Run `pytest -q` before opening a pull request.
4. Include tests for new behavior where practical.
5. Use the pull request template to summarize the change and validation.

## Development Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

Optional module dependencies are installed separately:

```bash
pip install -e ".[rag]"
pip install -e ".[vision]"
```

## Commit Style

Use concise imperative commit messages, for example:

```text
scaffold phase 1 project structure
add analysis pipeline smoke tests
```

