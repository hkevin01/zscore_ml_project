"""Benchmark several scaling methods across multiple classifiers."""

from __future__ import annotations

from utils import (
    evaluate_scalers,
    evaluate_scalers_with_cv,
    format_console_table,
    plot_confusion_matrices,
    plot_model_scores,
    plot_pca_projections,
    save_benchmark_tables,
    save_cv_tables,
)


def main() -> None:
    """Run holdout and repeated-CV benchmarks and save all tabular/figure outputs."""

    artifacts = evaluate_scalers()
    detailed_path, summary_path = save_benchmark_tables(artifacts)
    figure_path = plot_model_scores(artifacts.results_table)
    pca_path = plot_pca_projections(artifacts)
    confusion_paths = plot_confusion_matrices(artifacts)
    cv_detailed, cv_summary = evaluate_scalers_with_cv(folds=5, repeats=5)
    cv_detailed_path, cv_summary_path = save_cv_tables(cv_detailed, cv_summary)

    print("Saved detailed benchmark table:", detailed_path)
    print("Saved summary benchmark table:", summary_path)
    print("Saved model comparison figure:", figure_path)
    print("Saved PCA projections figure:", pca_path)
    print("Saved repeated-CV detailed table:", cv_detailed_path)
    print("Saved repeated-CV dataset summary table:", cv_summary_path)
    for path in confusion_paths:
        print("Saved confusion matrices:", path)
    print("\nMean accuracy by scaler")
    print(format_console_table(artifacts.summary_table))
    print("\nPer-model accuracy breakdown")
    print(format_console_table(artifacts.results_table))
    print("\nRepeated-CV accuracy by dataset, scaler, and model")
    print(format_console_table(cv_detailed))
    print("\nRepeated-CV dataset-level scaler summary")
    print(format_console_table(cv_summary))


if __name__ == "__main__":
    main()
