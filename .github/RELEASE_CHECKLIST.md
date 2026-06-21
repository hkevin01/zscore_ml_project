# Release Checklist

Use this checklist before tagging or publishing a release.

## 1. Version and Branch Hygiene

- [ ] Confirm you are on the correct release branch.
- [ ] Confirm local branch is up to date with remote.
- [ ] Confirm there are no uncommitted or unintended files.

## 2. Reproducibility Validation

- [ ] Create a clean virtual environment.
- [ ] Install dependencies from `requirements.txt`.
- [ ] Run `python src/zscore_demo.py` successfully.
- [ ] Run `python src/compare_scalers.py` successfully.
- [ ] Confirm expected outputs exist in `results/`.

## 3. Output Sanity Checks

- [ ] `results/scaler_model_scores.csv` generated.
- [ ] `results/scaler_comparison_table.csv` generated.
- [ ] `results/scaler_model_scores_cv.csv` generated.
- [ ] `results/scaler_dataset_summary_cv.csv` generated.
- [ ] `results/model_scores.png` generated.
- [ ] `results/pca_projections.png` generated.
- [ ] `results/confusion_matrices_*.png` generated.

## 4. Documentation Checks

- [ ] README matches current behavior and output names.
- [ ] Any methodology changes are described (CV folds/repeats, CI calculation).
- [ ] References and links are still valid.
- [ ] Contribution and template files reflect current workflow.

## 5. Quality and Risk Review

- [ ] Review warnings and decide if action is needed (for example MLP convergence warnings).
- [ ] Confirm no data leakage paths were introduced.
- [ ] Confirm random seed and split strategy are explicit.

## 6. Final Release Steps

- [ ] Create a release commit with clear message.
- [ ] Tag the release (for example `v1.1.0`).
- [ ] Publish release notes summarizing key changes.
- [ ] Announce known limitations and next planned improvements.
