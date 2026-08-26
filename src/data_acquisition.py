"""
data_acquisition.py

Phase 1: Data Acquisition
--------------------------
Loads the two raw source tables (USDA education/poverty, Census
income/housing). If the real files aren't present in data/raw/, a
realistically-structured synthetic dataset is generated instead so the
rest of the pipeline is runnable end-to-end without any downloads.

Real-data column contracts (see data/README.md for full details):

USDA education/poverty file -> FIPS, State, County, pct_bachelors_or_higher,
                                poverty_rate, rucc_code

Census income/housing file  -> FIPS, median_household_income,
                                median_home_value, median_gross_rent
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"
USDA_FILENAME = "usda_education_poverty.csv"
CENSUS_FILENAME = "census_income_housing.csv"
COUNTY_REFERENCE_FILENAME = "county_fips_reference.csv"

# RUCC (Rural-Urban Continuum Code) reference:
#   1-3 = Metro counties (varying population size)
#   4-9 = Nonmetro counties (varying adjacency/urbanization)
RUCC_CODES = list(range(1, 10))


def _read_csv_with_string_fips(path: Path) -> pd.DataFrame:
    """Read a CSV always treating FIPS as a zero-padded string.

    This directly addresses Pitfall #3: FIPS codes are 5 digits, but
    leading zeros (e.g., Alabama = '01001') get silently dropped if
    pandas infers the column as an integer.
    """
    # Read FIPS as text from the start so pandas does not discard a state's
    # leading zero when it guesses that the column contains integers.
    df = pd.read_csv(path, dtype={"FIPS": str})
    # Counties always use five digits, so restore padding if the source file
    # contains values such as "1001" instead of "01001".
    df["FIPS"] = df["FIPS"].str.zfill(5)
    return df


def load_usda_education_poverty(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the USDA ERS education & poverty county file."""
    # Build the path from the caller-supplied directory so tests and users
    # can load data from somewhere other than the default project folder.
    path = raw_dir / USDA_FILENAME
    return _read_csv_with_string_fips(path)


def load_census_income_housing(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the Census Bureau income & housing county file."""
    path = raw_dir / CENSUS_FILENAME
    return _read_csv_with_string_fips(path)


def raw_files_available(raw_dir: Path = RAW_DIR) -> bool:
    """Whether both real source files are present on disk."""
    # Both files are required because the next pipeline phase joins the two
    # tables; having only one would still leave the pipeline incomplete.
    return (raw_dir / USDA_FILENAME).exists() and (raw_dir / CENSUS_FILENAME).exists()


def load_county_reference(reference_dir: Path = REFERENCE_DIR) -> pd.DataFrame:
    """Load the bundled reference table of all ~3,221 REAL US county FIPS
    codes, names, and states (derived from the Census Bureau county
    boundary set used by the plotly choropleth).

    This is what makes the synthetic demo dataset map *densely* -- every
    synthetic row is attached to a real, mappable FIPS code, instead of a
    fabricated code that happens not to match any actual county polygon.
    """
    path = reference_dir / COUNTY_REFERENCE_FILENAME
    # Keep both FIPS columns as strings because they may later be used as
    # identifiers rather than numbers in a merge or a map.
    df = pd.read_csv(path, dtype={"FIPS": str, "state_fips": str})
    df["FIPS"] = df["FIPS"].str.zfill(5)
    return df


def generate_synthetic_data(
    n_counties: int | None = None,
    seed: int = 42,
    save_to_raw: bool = True,
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic-but-realistic pair of county-level tables.

    Mimics the structure and rough statistical relationships of the real
    USDA/Census data (education correlates with income; urban counties
    trend toward higher education *and* higher housing costs) so the
    downstream merge/metrics/visualization/analysis code can be
    exercised without needing live downloads.

    Unlike a purely fabricated FIPS scheme, this draws counties from the
    bundled REAL county reference table (data/reference/county_fips_reference.csv,
    ~3,221 counties), so every synthetic row maps to an actual county
    polygon and the choropleth renders as a dense, fully-shaded map
    instead of scattered points. Pass n_counties=None (default) to use
    ALL real counties.

    Returns
    -------
    (usda_df, census_df) as a tuple, matching the real-data schemas.
    """
    # A local random generator makes output reproducible without changing
    # NumPy's global random state elsewhere in the application.
    rng = np.random.default_rng(seed)

    # Start with real county identifiers so synthetic values can still be
    # joined to the real county boundaries used by the map.
    ref = load_county_reference()
    if n_counties is not None and n_counties < len(ref):
        # Sampling allows quick small demos while using random_state keeps
        # the selected counties stable for a given seed.
        ref = ref.sample(n=n_counties, random_state=seed).reset_index(drop=True)
    n = len(ref)

    fips = ref["FIPS"].tolist()
    chosen_states = ref["State"].tolist()
    county_names = ref["County"].tolist()

    # RUCC code: skew toward more nonmetro counties (realistic - most US
    # counties by *count* are nonmetro, even though most people live metro)
    # Draw RUCC values with explicit probabilities so the demo resembles the
    # real distribution, where nonmetro counties are more numerous by count.
    rucc = rng.choice(RUCC_CODES, size=n, p=[
        0.12, 0.07, 0.06,   # metro tiers 1-3
        0.10, 0.11, 0.13, 0.14, 0.13, 0.14,  # nonmetro tiers 4-9
    ])
    # USDA defines RUCC 1-3 as metro; this boolean drives the urban premium.
    is_metro = rucc <= 3
    n_counties = n

    # Education: metro counties skew higher, with noise
    # Metro and nonmetro counties begin with different education baselines;
    # normal noise prevents every county in a category from being identical.
    base_educ = np.where(is_metro, 34, 19)
    pct_bachelors = np.clip(rng.normal(base_educ, 8, size=n_counties), 5, 75)

    # Income is driven by education (true underlying relationship) plus
    # an urban premium plus noise
    # Income combines a baseline, an education effect, an urban premium, and
    # random variation to imitate a plausible correlation structure.
    median_income = (
        32000
        + pct_bachelors * 900
        + is_metro * 9000
        + rng.normal(0, 6000, size=n_counties)
    )
    # Clip impossible or implausibly extreme demo values to a defined range.
    median_income = np.clip(median_income, 28000, 160000)

    # Poverty inversely related to both education and income, plus noise
    # Higher education and income reduce the simulated poverty rate; clipping
    # keeps the percentage within a realistic interval for this demo.
    poverty_rate = np.clip(
        28 - (pct_bachelors * 0.28) - (median_income - 45000) / 6000
        + rng.normal(0, 3.5, size=n_counties),
        2, 42,
    )

    # Home values: education + urban premium is the dominant driver
    # (this is the relationship the urban/rural confound pitfall is about)
    # Home values deliberately have a strong urban component because the
    # analysis later demonstrates how urbanicity can confound correlations.
    median_home_value = (
        70000
        + pct_bachelors * 3200
        + is_metro * 160000
        + rng.normal(0, 35000, size=n_counties)
    )
    median_home_value = np.clip(median_home_value, 45000, 950000)

    # Rent follows the same broad pattern as home values but remains monthly.
    median_gross_rent = np.clip(
        450 + pct_bachelors * 9 + is_metro * 480 + rng.normal(0, 130, size=n_counties),
        380, 3200,
    )

    # Keep USDA-like and Census-like fields in separate tables to match the
    # shape of the real downloads and exercise the merge step.
    usda_df = pd.DataFrame({
        "FIPS": fips,
        "State": chosen_states,
        "County": county_names,
        "pct_bachelors_or_higher": pct_bachelors.round(1),
        "poverty_rate": poverty_rate.round(1),
        "rucc_code": rucc,
    })

    census_df = pd.DataFrame({
        "FIPS": fips,
        "median_household_income": median_income.round(0).astype(int),
        "median_home_value": median_home_value.round(0).astype(int),
        "median_gross_rent": median_gross_rent.round(0).astype(int),
    })

    # Drop a small fraction of rows from each side at random to simulate
    # real-world partial coverage / mismatched county sets, which
    # pd.merge() must handle gracefully (this exercises the merge
    # diagnostics too).
    # Remove a small, reproducible sample from each table to mimic imperfect
    # source coverage and give merge diagnostics something meaningful to show.
    n_drop = max(1, round(n_counties * 0.01))
    usda_df = usda_df.drop(index=rng.choice(usda_df.index, size=n_drop, replace=False)).reset_index(drop=True)
    census_df = census_df.drop(index=rng.choice(census_df.index, size=n_drop, replace=False)).reset_index(drop=True)

    # Caching makes the next normal pipeline run use the generated tables
    # instead of generating them again.
    if save_to_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        usda_df.to_csv(raw_dir / USDA_FILENAME, index=False)
        census_df.to_csv(raw_dir / CENSUS_FILENAME, index=False)

    return usda_df, census_df


def load_or_generate(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience loader: use real files if present, else synthesize demo data."""
    # Prefer user-provided real files; synthetic data is only the fallback.
    if raw_files_available(raw_dir):
        return load_usda_education_poverty(raw_dir), load_census_income_housing(raw_dir)
    print(
        "[data_acquisition] Real CSVs not found in data/raw/ — "
        "generating a synthetic demo dataset instead. "
        "See data/README.md to plug in real USDA/Census data."
    )
    return generate_synthetic_data(raw_dir=raw_dir)
