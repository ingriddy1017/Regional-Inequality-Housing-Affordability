"""
main.py

End-to-end pipeline:
  1. Data Acquisition   -> src/data_acquisition.py
  2. Merging Datasets   -> src/data_merge.py
  3. Metric Engineering -> src/metrics.py
  4. Visualization      -> src/visualization.py
  (+ Urban/Rural confound analysis -> src/analysis.py)

Run with:  python main.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import analysis, data_acquisition, data_merge, metrics, visualization

ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"


def main(use_real_data: bool = False) -> None:
    # Create output folders up front so every later save has a destination.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Data Acquisition ---
    print("\n=== Phase 1: Data Acquisition ===")
    # Real downloads are opt-in; the default remains runnable offline.
    if use_real_data:
        try:
            from src import real_data
            usda_df, census_df = real_data.build_real_dataset()
        except Exception as exc:
            # A failed network request or changed source schema should not
            # prevent the local synthetic demonstration from running.
            print(f"[main] Real-data fetch failed ({exc}); falling back to synthetic demo data.")
            usda_df, census_df = data_acquisition.load_or_generate()
    else:
        usda_df, census_df = data_acquisition.load_or_generate()
    print(f"USDA education/poverty table: {usda_df.shape}")
    print(f"Census income/housing table: {census_df.shape}")

    # --- Phase 2: Merging Datasets ---
    print("\n=== Phase 2: Merging Datasets ===")
    # Analyze only counties represented in both source tables.
    merged = data_merge.merge_datasets(usda_df, census_df, how="inner")

    # --- Phase 3: Metric Engineering ---
    print("\n=== Phase 3: Metric Engineering ===")
    # Add all derived fields before any analysis or visualization consumes the
    # merged table.
    merged = metrics.engineer_all_metrics(merged)
    data_merge.save_merged(merged, TABLES_DIR / "merged_county_data.csv")

    # Save the matrix separately because it is useful apart from row-level
    # county data.
    corr = metrics.correlation_matrix(merged)
    corr.to_csv(TABLES_DIR / "correlation_matrix.csv")
    print("\nCorrelation matrix:\n", corr)

    corr_by_metro = metrics.correlation_by_group(
        merged, group_col="is_metro", x="pct_bachelors_or_higher", y="poverty_rate"
    )
    corr_by_metro.to_csv(TABLES_DIR / "correlation_by_metro_group.csv", index=False)
    print("\nEducation<->Poverty correlation by metro/nonmetro:\n", corr_by_metro)

    # --- Urban/Rural confound check (Pitfall #1) ---
    print("\n=== Urban/Rural Confound Analysis ===")
    # Produce broad metro/nonmetro summaries before the finer RUCC analysis.
    urbanicity_summary = analysis.summarize_by_urbanicity(merged)
    urbanicity_summary.to_csv(TABLES_DIR / "summary_by_urbanicity.csv")
    print(urbanicity_summary)

    rucc_summary = analysis.summarize_by_rucc_tier(merged)
    rucc_summary.to_csv(TABLES_DIR / "summary_by_rucc_tier.csv", index=False)

    confound_check = analysis.education_affordability_relationship(merged)
    confound_check.to_csv(TABLES_DIR / "education_affordability_relationship.csv", index=False)
    print("\nEducation <-> Price-to-Income Ratio correlation, overall vs. within-group:")
    print(confound_check)

    # --- Phase 4: Visualization ---
    print("\n=== Phase 4: Visualization ===")
    visualization.plot_correlation_heatmap(
        corr, FIGURES_DIR / "correlation_heatmap.png"
    )
    visualization.plot_choropleth(
        merged,
        value_column="price_to_income_ratio",
        save_path=FIGURES_DIR / "price_to_income_choropleth.html",
        title="Housing Price-to-Income Ratio by County",
    )
    visualization.plot_choropleth(
        merged,
        value_column="pct_bachelors_or_higher",
        save_path=FIGURES_DIR / "education_choropleth.html",
        title="% Population with Bachelor's Degree or Higher, by County",
    )

    print("\nDone. See outputs/tables/ and outputs/figures/.")


if __name__ == "__main__":
    # Keep argument parsing out of imports so another module can call main()
    # without accidentally starting the pipeline.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Fetch real USDA ERS + Census ACS data instead of using the synthetic demo dataset.",
    )
    args = parser.parse_args()
    main(use_real_data=args.real_data)
