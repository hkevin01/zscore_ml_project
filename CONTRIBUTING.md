# Contributing Guide

Thank you for contributing to this project. The repository is designed to be clear, reproducible, and educational, so contributions should prioritize correctness, readability, and reproducible outputs.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run both scripts to validate outputs.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/zscore_demo.py
python src/compare_scalers.py
```

## Contribution Scope

Good contribution targets:

- Better experimental rigor (for example repeated CV improvements or additional metrics).
- Better documentation and interpretation of model/scaler behavior.
- Bug fixes in data splitting, leakage prevention, or plotting logic.
- New datasets that are tabular, numeric, and compatible with the project goals.

Out of scope unless discussed first:

- Large framework migrations.
- Serving/deployment infrastructure unrelated to the benchmark.
- Heavy dependencies that do not materially improve reproducibility.

## Coding Standards

- Keep functions focused and typed where practical.
- Prefer leakage-safe scikit-learn `Pipeline` usage.
- Keep random seeds explicit.
- Preserve compatibility with Python 3.11+.

## Validation Before PR

Run the following checks before opening a pull request:

1. `python src/zscore_demo.py`
2. `python src/compare_scalers.py`
3. Confirm updated CSV and plot files are generated under `results/`.
4. Confirm README sections are still aligned with current script behavior.

## Pull Request Expectations

Each pull request should include:

- A concise problem statement.
- A clear summary of what changed and why.
- Any metric or artifact deltas if experimental behavior changed.
- Notes about known limitations or follow-up work.

## Commit Message Guidance

Use clear, scoped commit messages, for example:

- `docs: expand README with CV methodology section`
- `feat: add repeated stratified CV summary table`
- `fix: prevent scaler fit leakage in benchmark path`

## Reporting Bugs

Use the bug issue template and include:

- Reproduction steps.
- Expected behavior.
- Actual behavior.
- Python version and OS.
- Relevant traceback or warning output.
