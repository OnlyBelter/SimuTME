# GitHub Actions tests

Run the complete SimuTME test suite automatically on pushes and pull requests.

## Workflow

Add a GitHub Actions workflow using an Ubuntu runner and Python 3.12. Install
NumPy, SciPy, and the required AnnData version from conda-forge, then install
SimuTME and pytest using the package's current `pyproject.toml` requirements.
The existing Poetry lockfile is not used because it is out of sync with the
current project requirements.

Run `python -m pip check` and `python -m pytest -q` with a headless Matplotlib
backend and limited numerical-library threads. Set repository permissions to
read-only. A failed installation or test must fail the workflow.

## Verification

Validate the workflow YAML and its triggers locally and run the full test
suite in the existing environment. GitHub-hosted execution can be verified
after the workflow is committed and pushed; no commit or push is implied.
