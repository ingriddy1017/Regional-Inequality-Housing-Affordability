"""
analysis.py

Urban/Rural-aware analysis
---------------------------
Directly addresses Pitfall #1 (Confounding Variables: Urban vs. Rural).

Naively correlating education attainment with home prices nationwide
mostly captures "cities are expensive, and cities have more college
graduates" rather than isolating the effect of education itself. This
module stratifies by USDA Rural-Urban Continuum Code (RUCC) tier so we
can compare the education<->affordability relationship *within* metro
counties and *within* nonmetro counties separately.
"""

from __future__ import annotations

import pandas as pd

RUCC_LABELS = {
    1: "Metro >= 1M pop",
    2: "Metro 250K-1M pop",
    3: "Metro < 250K pop",
    4: "Nonmetro, urban >=20K, adj. metro",
    5: "Nonmetro, urban >=20K, not adj. metro",
    6: "Nonmetro, urban 2.5K-20K, adj. metro",
    7: "Nonmetro, urban 2.5K-20K, not adj. metro",
    8: "Nonmetro, rural, adj. metro",
    9: "Nonmetro, rural, not adj. metro",
}


def add_rucc_label(df: pd.DataFrame, rucc_column: str = "rucc_code") -> pd.DataFrame:
    df = df.copy()
    df["rucc_label"] = df[rucc_column].map(RUCC_LABELS)
    return df


def summarize_by_urbanicity(df: pd.DataFrame) -> pd.DataFrame:
    """Median metrics grouped by metro vs. nonmetro status.

    Uses medians throughout (Pitfall #2), and reports county counts so
    small groups aren't over-interpreted.
    """
    agg = df.groupby("is_metro").agg(
        n_counties=("FIPS", "count"),
        median_pct_bachelors=("pct_bachelors_or_higher", "median"),
        median_poverty_rate=("poverty_rate", "median"),
        median_household_income=("median_household_income", "median"),
        median_home_value=("median_home_value", "median"),
        median_price_to_income_ratio=("price_to_income_ratio", "median"),
        median_rent_burden_pct=("rent_burden_pct", "median"),
    )
    agg.index = agg.index.map({True: "Metro", False: "Nonmetro"})
    return agg.round(2)


def summarize_by_rucc_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Same as above but broken out into all 9 RUCC tiers, not just metro/nonmetro."""
    df = add_rucc_label(df)
    agg = df.groupby(["rucc_code", "rucc_label"]).agg(
        n_counties=("FIPS", "count"),
        median_pct_bachelors=("pct_bachelors_or_higher", "median"),
        median_poverty_rate=("poverty_rate", "median"),
        median_household_income=("median_household_income", "median"),
        median_home_value=("median_home_value", "median"),
        median_price_to_income_ratio=("price_to_income_ratio", "median"),
    ).round(2)
    return agg.reset_index()


def education_affordability_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """Compare education<->price-to-income-ratio correlation overall vs.
    within each urbanicity group, to surface confounding.

    If the "overall" correlation is much stronger than the within-group
    correlations, that's a signal the relationship is substantially
    driven by the urban/rural mix rather than education itself.
    """
    rows = []
    overall = df["pct_bachelors_or_higher"].corr(df["price_to_income_ratio"])
    rows.append({"group": "Overall (unadjusted)", "n": len(df), "correlation": round(overall, 3)})

    for is_metro, label in [(True, "Metro only"), (False, "Nonmetro only")]:
        sub = df[df["is_metro"] == is_metro]
        corr = sub["pct_bachelors_or_higher"].corr(sub["price_to_income_ratio"])
        rows.append({"group": label, "n": len(sub), "correlation": round(corr, 3)})

    return pd.DataFrame(rows)
