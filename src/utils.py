"""Utility functions for reproducible feature-scaling experiments.

This module centralizes data generation, scaler application, benchmarking,
and result serialization so the project scripts stay small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC


matplotlib.use("Agg")


SEED = 42


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Container for the primary benchmark outputs."""

    results_table: pd.DataFrame
    summary_table: pd.DataFrame
    predictions: dict  # {(scaler_name, model_name): np.ndarray of y_pred}
    y_test: np.ndarray
    scaled_test: dict  # {scaler_name: np.ndarray} for PCA


def project_root() -> Path:
    """Return the repository root from the src directory."""

    return Path(__file__).resolve().parents[1]


def ensure_directories() -> None:
    """Ensure the project directories exist before writing files."""

    root = project_root()
    for relative_path in ("data", "results", "src", "docs"):
        (root / relative_path).mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(samples: int = 150) -> pd.DataFrame:
    """Generate a synthetic numeric dataset with mixed scales and mild outliers."""

    rng = np.random.default_rng(SEED)
    heights = rng.normal(170, 10, samples)
    weights = rng.normal(70, 15, samples)
    ages = rng.normal(40, 12, samples)
    incomes = rng.normal(65000, 18000, samples)

    outlier_indices = rng.choice(samples, size=max(3, samples // 20), replace=False)
    incomes[outlier_indices] *= 1.9
    weights[outlier_indices] += 18

    return pd.DataFrame(
        {
            "height_cm": heights,
            "weight_kg": weights,
            "age_years": ages,
            "annual_income_usd": incomes,
        }
    )


def save_synthetic_data(data: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Persist the synthetic dataset to CSV and return the file path."""

    ensure_directories()
    destination = output_path or project_root() / "data" / "synthetic_data.csv"
    data.to_csv(destination, index=False)
    return destination


def zscore_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply z-score standardization and return the transformed frame and scaler."""

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    scaled_frame = pd.DataFrame(scaled, columns=data.columns)
    return scaled_frame, scaler


def summarize_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation summary statistics for each feature."""

    return pd.DataFrame(
        {
            "mean": data.mean(),
            "std": data.std(ddof=0),
            "min": data.min(),
            "max": data.max(),
        }
    )


def build_scalers() -> dict[str, object]:
    """Return the scaler collection compared in the project."""

    return {
        "zscore": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "l2_norm": Normalizer(),
    }


def build_models() -> dict[str, object]:
    """Return a small model suite covering scaling-sensitive and robust learners."""

    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "knn": KNeighborsClassifier(n_neighbors=9),
        "svm_rbf": SVC(kernel="rbf", gamma="scale", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=SEED),
    }


def _classification_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the benchmark classification dataset as pandas objects."""

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target
    return features, target


def evaluate_scalers() -> BenchmarkArtifacts:
    """Benchmark each scaler across multiple models with train-only fitting."""

    features, target = _classification_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=target,
    )

    rows: list[dict[str, float | str]] = []
    preds: dict[tuple[str, str], np.ndarray] = {}
    scaled_test: dict[str, np.ndarray] = {}
    scalers = build_scalers()
    models = build_models()

    for scaler_name, scaler in scalers.items():
        fitted_scaler = clone(scaler).fit(X_train)
        scaled_test[scaler_name] = fitted_scaler.transform(X_test)

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("scaler", clone(scaler)),
                    ("model", clone(model)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            preds[(scaler_name, model_name)] = y_pred
            rows.append(
                {
                    "scaler": scaler_name,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, y_pred),
                }
            )

    results = pd.DataFrame(rows).sort_values(["model", "accuracy"], ascending=[True, False])
    summary = (
        results.groupby("scaler", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "mean_accuracy"})
        .sort_values("mean_accuracy", ascending=False)
    )
    return BenchmarkArtifacts(
        results_table=results,
        summary_table=summary,
        predictions=preds,
        y_test=y_test.to_numpy(),
        scaled_test=scaled_test,
    )


def save_benchmark_tables(artifacts: BenchmarkArtifacts) -> tuple[Path, Path]:
    """Write the detailed and summary benchmark tables to CSV files."""

    ensure_directories()
    results_dir = project_root() / "results"
    detailed_path = results_dir / "scaler_model_scores.csv"
    summary_path = results_dir / "scaler_comparison_table.csv"
    artifacts.results_table.to_csv(detailed_path, index=False)
    artifacts.summary_table.to_csv(summary_path, index=False)
    return detailed_path, summary_path


def plot_model_scores(results: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Create a grouped bar chart of accuracy by model and scaler."""

    ensure_directories()
    figure_path = output_path or project_root() / "results" / "model_scores.png"

    pivot = results.pivot(index="model", columns="scaler", values="accuracy")
    ordered_scalers = [name for name in ("zscore", "minmax", "robust", "l2_norm") if name in pivot.columns]
    pivot = pivot[ordered_scalers]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.82, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    ax.set_title("Model Accuracy by Scaling Strategy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.85, 1.01)
    ax.legend(title="Scaler", ncol=2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_confusion_matrices(
    artifacts: BenchmarkArtifacts,
    output_dir: Path | None = None,
) -> list[Path]:
    """Save one confusion-matrix grid per scaler, with one panel per model."""

    ensure_directories()
    results_dir = output_dir or project_root() / "results"
    model_names = list(build_models().keys())
    scaler_names = list(build_scalers().keys())
    saved: list[Path] = []

    for scaler_name in scaler_names:
        n = len(model_names)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        fig.suptitle(f"Confusion Matrices - {scaler_name}", fontsize=13)
        for ax, model_name in zip(axes, model_names):
            y_pred = artifacts.predictions[(scaler_name, model_name)]
            ConfusionMatrixDisplay.from_predictions(
                artifacts.y_test,
                y_pred,
                ax=ax,
                colorbar=False,
            )
            ax.set_title(model_name.replace("_", "\n"), fontsize=9)
        fig.tight_layout()
        path = results_dir / f"confusion_matrices_{scaler_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def plot_pca_projections(
    artifacts: BenchmarkArtifacts,
    output_path: Path | None = None,
) -> Path:
    """Save 2D PCA scatter plots showing class separation under each scaler."""

    ensure_directories()
    figure_path = output_path or project_root() / "results" / "pca_projections.png"
    scaler_names = list(artifacts.scaled_test.keys())
    n = len(scaler_names)
    ncols = 2
    nrows = (n + 1) // 2
    class_colors = ["#d62728", "#1f77b4"]
    class_labels = ["malignant (0)", "benign (1)"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_flat = axes.flatten()

    pca = PCA(n_components=2, random_state=SEED)
    for idx, scaler_name in enumerate(scaler_names):
        ax = axes_flat[idx]
        components = pca.fit_transform(artifacts.scaled_test[scaler_name])
        for class_val, color, label in zip([0, 1], class_colors, class_labels):
            mask = artifacts.y_test == class_val
            ax.scatter(
                components[mask, 0],
                components[mask, 1],
                c=color,
                label=label,
                alpha=0.7,
                s=30,
                edgecolors="none",
            )
        var = pca.explained_variance_ratio_
        ax.set_title(f"{scaler_name}  (PC1 {var[0]:.1%}, PC2 {var[1]:.1%})")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8)

    for idx in range(len(scaler_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("PCA 2D Projection by Scaler", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def format_console_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Return a consistently formatted string representation for console output."""

    display_frame = frame.copy()
    for column in display_frame.select_dtypes(include=["float", "float64", "float32"]):
        display_frame[column] = display_frame[column].map(lambda value: f"{value:.{digits}f}")
    return display_frame.to_string(index=False)
