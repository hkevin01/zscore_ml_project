"""Benchmark several scaling methods across multiple classifiers."""

from __future__ import annotations

from utils import (
    evaluate_scalers,
    format_console_table,
    plot_confusion_matrices,
    plot_model_scores,
    plot_pca_projections,
    save_benchmark_tables,
)


def main() -> None:
    """Run the benchmark and save CSV tables, accuracy chart, PCA plot, and confusion matrices."""

    artifacts = evaluate_scalers()
    detailed_path, summary_path = save_benchmark_tables(artifacts)
    figure_path = plot_model_scores(artifacts.results_table)
    pca_path = plot_pca_projections(artifacts)
    confusion_paths = plot_confusion_matrices(artifacts)

    print("Saved detailed benchmark table:", detailed_path)
    print("Saved summary benchmark table:", summary_path)
    print("Saved model comparison figure:", figure_path)
    print("Saved PCA projections figure:", pca_path)
    for path in confusion_paths:
        print("Saved confusion matrices:", path)
    print("\nMean accuracy by scaler")
    print(format_console_table(artifacts.summary_table))
    print("\nPer-model accuracy breakdown")
    print(format_console_table(artifacts.results_table))


if __name__ == "__main__":
    main()
