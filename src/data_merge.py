"""
data_merge.py

Phase 2: Merging Datasets
--------------------------
Joins the Education/Poverty (USDA) table and the Income/Housing (Census)
table on the 5-digit FIPS county code using pd.merge().

Directly addresses Pitfall #3 (FIPS formatting): every FIPS column is
forced to a zero-padded string before merging, so '01001' never silently
becomes the integer 1001 and drops out of the join.
"""

from __future__ import annotations

import pandas as pd


def normalize_fips(df: pd.DataFrame, column: str = "FIPS") -> pd.DataFrame:
    """Force a FIPS column to a clean, zero-padded 5-character string.

    Handles the common failure modes: FIPS read as int64 (drops leading
    zero), FIPS read as float (adds a trailing '.0'), or stray whitespace.
    """
    df = df.copy()
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(5)
    )
    return df


def merge_datasets(
    usda_df: pd.DataFrame,
    census_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge the Education/Poverty and Income/Housing tables on FIPS.

    Parameters
    ----------
    usda_df : DataFrame with columns including FIPS, pct_bachelors_or_higher,
        poverty_rate, rucc_code
    census_df : DataFrame with columns including FIPS, median_household_income,
        median_home_value, median_gross_rent
    how : join type passed to pd.merge (default 'inner' — only keep
        counties present in both source tables)

    Returns
    -------
    Merged DataFrame, one row per county.
    """
    usda_df = normalize_fips(usda_df)
    census_df = normalize_fips(census_df)

    before_usda, before_census = len(usda_df), len(census_df)

    merged = pd.merge(
        usda_df,
        census_df,
        on="FIPS",
        how=how,
        validate="one_to_one",
        suffixes=("_usda", "_census"),
    )

    dropped = (before_usda + before_census) - 2 * len(merged) if how == "inner" else None
    print(
        f"[data_merge] USDA rows: {before_usda} | Census rows: {before_census} "
        f"| Merged rows: {len(merged)} (how='{how}')"
    )
    unmatched_usda = set(usda_df["FIPS"]) - set(census_df["FIPS"])
    unmatched_census = set(census_df["FIPS"]) - set(usda_df["FIPS"])
    if unmatched_usda or unmatched_census:
        print(
            f"[data_merge] {len(unmatched_usda)} FIPS only in USDA table, "
            f"{len(unmatched_census)} FIPS only in Census table (dropped under how='{how}')."
        )

    return merged


def save_merged(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"[data_merge] Saved merged dataset -> {path} ({len(df)} rows)")
