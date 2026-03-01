# Contributing to distenum

Thanks for your interest in contributing. This document explains how to get set up and submit changes.

## Development setup

1. **Clone the repository** and go to the project root:
   ```bash
   git clone https://github.com/ktaneja6/distenum.git
   cd distenum
   ```

2. **Create a virtual environment, install the package with dev dependencies, and run tests**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   pytest tests/ -v
   ```
   The `dev` extra adds `pytest` and `ruff`. After this, you can edit the code and re-run `pytest tests/ -v` as needed.

   To also run the example script (uses the OpenAI API):
   ```bash
   pip install -e ".[dev,openai]"
   ```

## Running tests

From the project root (with the venv activated):

```bash
pytest tests/ -v
```

Add tests in `tests/` for new behavior. Use the helpers in `tests/conftest.py` to build mock logprobs (e.g. `make_logprobs_data`, `make_content_token`, `make_top_logprob`) so tests don’t call the API.

## Code style

- **Lint and format** with [ruff](https://docs.astral.sh/ruff/):
  ```bash
  ruff check distenum tests
  ruff format distenum tests
  ```
  Fix auto-fixable issues with:
  ```bash
  ruff check distenum tests --fix
  ```

- **Config:** Ruff is configured in `pyproject.toml` (`[tool.ruff]`). Use Python 3.9+ style; line length 100.

- Please run `ruff check` and `ruff format` before opening a pull request so CI stays green.

## How to contribute

1. **Open an issue** for bugs or feature ideas so we can align before you spend time on code.

2. **Fork the repo**, create a branch (e.g. `fix/parser-edge-case` or `feat/support-xyz`), and make your changes.

3. **Add or update tests** for new or changed behavior.

4. **Run tests and ruff** (see above).

5. **Open a pull request** against `main` with:
   - A short description of what changed and why
   - Any notes for reviewers (e.g. “first commit only adds tests; fix in next”)

6. **License:** By contributing, you agree that your contributions will be licensed under the same MIT License as the project.

If you have questions, open an issue and we’ll help.
