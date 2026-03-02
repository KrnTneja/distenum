# Publishing distenum to PyPI

Follow these steps to publish a new version to PyPI.

## Prerequisites

- PyPI account: https://pypi.org/account/register/
- Optional (for testing first): Test PyPI account: https://test.pypi.org/account/register/

## Step 1: Install build tools

```bash
pip install build twine
```

## Step 2: Bump the version

Edit `pyproject.toml` and set the release version, e.g.:

```toml
version = "0.1.0"
```

Commit the change. Optionally tag the release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Step 3: Build the distributions

From the **repository root** (where `pyproject.toml` is):

```bash
python -m build
```

This creates in `dist/`:

- `distenum-0.1.0.tar.gz` (source distribution)
- `distenum-0.1.0-py3-none-any.whl` (wheel)

## Step 4: Create a PyPI API token

1. Log in at https://pypi.org
2. Go to **Account settings** → **API tokens** → **Add API token**
3. Name it (e.g. `distenum`) and choose scope (project-specific or entire account)
4. Copy the token (starts with `pypi-`); you won’t see it again

## Step 5: Upload to PyPI

**Option A – Prompted for credentials**

```bash
twine upload dist/*
```

When prompted:

- **Username:** `__token__`
- **Password:** paste your PyPI API token

**Option B – Use Test PyPI first**

```bash
twine upload --repository testpypi dist/*
```

Use a Test PyPI token and username `__token__`. Then test install:

```bash
pip install --index-url https://test.pypi.org/simple/ distenum
```

**Option C – Save token in `~/.pypirc` (optional)**

Create or edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Then run:

```bash
twine upload dist/*
```

(Use `twine upload --repository testpypi dist/*` for Test PyPI.)

## Step 6: Verify

```bash
pip install distenum
python -c "from distenum import parse_using_schema_and_logprobs; print('OK')"
```

## Re-releasing

PyPI does not allow re-uploading the same version. For each release:

1. Bump `version` in `pyproject.toml` (e.g. to `0.1.1`)
2. Run `python -m build`
3. Run `twine upload dist/*`
