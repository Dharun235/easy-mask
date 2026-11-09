# Contributing to easy-mask

Thanks for helping. This file is the short version; [CLAUDE.md](CLAUDE.md) has the full
code guidelines and applies to humans too.

## Setup

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## Workflow

1. Open (or pick) an issue describing the change.
2. Branch from `main`: `feat/<name>`, `fix/<name>`, `docs/<name>` or `chore/<name>`.
3. Write the test first when you can; make it pass; keep the diff focused.
4. `uv run ruff check . && uv run mypy && uv run pytest --cov` must pass.
   Which tests to write: [docs/testing.md](docs/testing.md).
5. Open a PR into `main` using the template. Squash-merge once CI is green.

## Code style in brief

- Google-style docstrings on every public module, class and function.
- Type hints on every function signature.
- `core/` and `io/` never import Qt. GUI code stays in `gui/`.
- Small, obvious functions over clever ones. No speculative abstractions.
- Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), never by hand.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`test:`, `refactor:`, `chore:`. Subject in the imperative, under ~60 characters.
