"""Run the z-score standardization demonstration for the synthetic dataset."""

from __future__ import annotations

from utils import format_console_table, generate_synthetic_data, save_synthetic_data, summarize_frame, zscore_frame


def main() -> None:
    """Generate synthetic data, apply z-score scaling, and print summaries."""

    data = generate_synthetic_data()
    output_path = save_synthetic_data(data)
    zscored, _ = zscore_frame(data)

    print("Saved synthetic dataset:", output_path)
    print("\nOriginal feature summary")
    print(format_console_table(summarize_frame(data)))
    print("\nZ-score standardized feature summary")
    print(format_console_table(summarize_frame(zscored)))
    print("\nFirst five standardized rows")
    print(zscored.head().to_string(index=False))


if __name__ == "__main__":
    main()
