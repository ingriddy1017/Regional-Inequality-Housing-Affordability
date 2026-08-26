"""
metrics.py

Phase 3: Metric Engineering
----------------------------
Custom derived metrics built on top of the merged county table, plus
correlation-matrix helpers. Always uses MEDIAN income/home value figures
(Pitfall #2) — never means — since a handful of extreme-wealth outliers
can badly distort an average at the county level.
"""

from __future__ import annotations

import pandas as pd


def add_price_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Price-to-Income Ratio = median_home_value / median_household_income.

    A standard housing-affordability metric (values above ~4-5 are
    generally considered 'severely unaffordable' by housing economists).
    """
    # Return a new DataFrame so composing metric functions does not mutate the
    # caller's original table unexpectedly.
    df = df.copy()
    # The quotient is unit-free, making home prices comparable relative to
    # annual income across counties.
    df["price_to_income_ratio"] = (
        df["median_home_value"] / df["median_household_income"]
    ).round(2)
    return df


def add_rent_burden_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Rent Burden % = (median_gross_rent * 12) / median_household_income * 100.

    The classic HUD "30% of income on rent" affordability threshold can
    be checked directly against this column.
    """
    df = df.copy()
    # Rent is monthly but income is annual, so annualize rent before dividing.
    df["rent_burden_pct"] = (
        (df["median_gross_rent"] * 12) / df["median_household_income"] * 100
    ).round(1)
    return df


def add_urban_flag(df: pd.DataFrame, rucc_column: str = "rucc_code") -> pd.DataFrame:
    """Add a boolean is_metro flag from the USDA RUCC code (1-3 = metro)."""
    df = df.copy()
    # USDA defines RUCC values 1-3 as metro counties; the comparison creates a
    # boolean column that is convenient for grouping and filtering.
    df["is_metro"] = df[rucc_column] <= 3
    return df


def engineer_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full metric-engineering pipeline in one call."""
    # Run the transformations in order so downstream analysis receives every
    # derived column it expects.
    df = add_price_to_income_ratio(df)
    df = add_rent_burden_pct(df)
    df = add_urban_flag(df)
    return df


def correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlation matrix across the key socioeconomic variables."""
    # Use the full project metric set by default, while allowing focused calls.
    if columns is None:
        columns = [
            "pct_bachelors_or_higher",
            "poverty_rate",
            "median_household_income",
            "median_home_value",
            "median_gross_rent",
            "price_to_income_ratio",
            "rent_burden_pct",
        ]
    # Partial DataFrames are valid inputs, so ignore requested fields that are
    # not present instead of raising a KeyError.
    columns = [c for c in columns if c in df.columns]
    return df[columns].corr(method=method).round(3)


def correlation_by_group(
    df: pd.DataFrame,
    group_col: str,
    x: str = "pct_bachelors_or_higher",
    y: str = "poverty_rate",
) -> pd.DataFrame:
    """Correlation between x and y computed separately within each group.

    Used to check whether an education<->poverty (or education<->housing)
    relationship holds *within* urban and *within* rural counties
    separately, rather than being an artifact of urban/rural mix
    (Pitfall #1: Confounding Variables).
    """
    # Compute the correlation independently inside each group rather than
    # mixing metro and nonmetro counties into one pooled value.
    out = (
        df.groupby(group_col)
        .apply(lambda g: g[x].corr(g[y]))
        .rename(f"corr({x}, {y})")
        .reset_index()
    )
    return out
