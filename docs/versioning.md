# Versioning And Release Management

Ped-Agent uses semantic versioning:

- `MAJOR` changes for breaking public APIs, configuration keys, or data schemas.
- `MINOR` changes for backward-compatible features.
- `PATCH` changes for fixes, documentation, and internal maintenance.

## Branches

- `main`: stable integration branch.
- `feature/<topic>`: feature development.
- `fix/<topic>`: bug fixes.
- `release/<version>`: release preparation when needed.

## Tags

Release tags use the `vMAJOR.MINOR.PATCH` format, for example:

```bash
git tag -a v0.1.0 -m "Ped-Agent v0.1.0"
git push origin v0.1.0
```

## Release Checklist

1. Update `CHANGELOG.md`.
2. Confirm `pyproject.toml` version.
3. Run `pytest -q`.
4. Create an annotated tag.
5. Publish GitHub release notes from the changelog entry.

