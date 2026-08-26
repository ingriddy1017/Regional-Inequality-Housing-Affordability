"""
real_data.py

Fetches ACTUAL (non-synthetic) county-level data from free, public sources
that don't require any registration:

- USDA ERS County-level Data Sets (direct CSV downloads, updated annually):
    https://www.ers.usda.gov/data-products/county-level-data-sets/county-level-data-sets-download-data
  Used for: education attainment, poverty rate, median household income,
  and Rural-Urban Continuum Codes.

- US Census Bureau ACS 5-Year Estimates API (free, no key required for
  light/occasional use -- a free key is recommended for anything heavier,
  sign up at https://api.census.gov/data/key_signup.html):
    https://api.census.gov/data/{year}/acs/acs5
  Used for: median home value and median gross rent (not published by
  USDA ERS).

NOTE ON THIS SANDBOX: ers.usda.gov and api.census.gov are not reachable
from Claude's sandboxed code-execution environment (only a small
allow-list of domains, e.g. pypi.org/github.com, is reachable there).
That means these functions can't be exercised inside that sandbox -- but
they use plain https GET requests to public, unauthenticated endpoints,
so they will work normally in your own local environment or CI, where
outbound internet access isn't restricted the same way.

Run `python -m src.real_data` directly to fetch and cache everything into
data/raw/, then run `python main.py` as usual.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# --- USDA ERS direct file links (updated 2025; verify latest at the URL above) ---
USDA_POVERTY_CSV = "https://www.ers.usda.gov/media/5496/poverty-estimates-for-the-united-states-states-and-counties-2023.csv?v=72628"
USDA_EDUCATION_CSV = "https://www.ers.usda.gov/media/5495/educational-attainment-for-adults-age-25-and-older-for-the-united-states-states-and-counties-1970-2023.csv?v=69622"
USDA_INCOME_UNEMPLOYMENT_CSV = "https://www.ers.usda.gov/media/5497/unemployment-and-median-household-income-for-the-united-states-states-and-counties-2000-23.csv?v=82928"
USDA_RUCC_CSV = "https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv?v=30758"

# --- Census ACS 5-Year API ---
CENSUS_ACS_BASE = "https://api.census.gov/data/{year}/acs/acs5"
# B25077_001E = median home value, B25064_001E = median gross rent (monthly)
CENSUS_HOUSING_VARS = "B25077_001E,B25064_001E"

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (regional-socioeconomic-inequality research script)"}


def _get_csv(url: str, encoding: str = "latin-1", **read_csv_kwargs) -> pd.DataFrame:
    # A timeout prevents an unavailable public endpoint from blocking the
    # entire pipeline indefinitely.
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content), encoding=encoding, **read_csv_kwargs)


def _find_column(columns: list[str], *keyword_groups: list[str]) -> str | None:
    """Find the first column whose (lowercased) name contains every
    keyword in one of the given keyword groups. USDA ERS renames columns
    with the data year baked in (e.g. 'PCTPOVALL_2023'), so exact-name
    matching is brittle across years -- this matches on stable substrings
    instead.
    """
    # Match using lowercase text but return the original column spelling for
    # pandas selection.
    lowered = {c: c.lower() for c in columns}
    for group in keyword_groups:
        for col, low in lowered.items():
            if all(kw in low for kw in group):
                return col
    return None


def fetch_usda_poverty() -> pd.DataFrame:
    """Real county poverty rate from USDA ERS (SAIPE-based estimates)."""
    # USDA has used more than one FIPS column name across releases, so support
    # the known alternatives before standardizing the output.
    df = _get_csv(USDA_POVERTY_CSV, dtype={"FIPS_Code": str, "FIPStxt": str})
    fips_col = "FIPS_Code" if "FIPS_Code" in df.columns else "FIPStxt"
    poverty_col = _find_column(list(df.columns), ["pctpovall"], ["poverty", "percent"], ["poverty", "all", "ages"])
    if poverty_col is None:
        raise ValueError(f"Couldn't find a poverty-rate column in USDA poverty CSV. Columns: {list(df.columns)}")
    out = df[[fips_col, poverty_col]].rename(columns={fips_col: "FIPS", poverty_col: "poverty_rate"})
    # Normalize identifiers immediately so this table is merge-ready.
    out["FIPS"] = out["FIPS"].astype(str).str.zfill(5)
    return out.dropna(subset=["FIPS"])


def fetch_usda_education() -> pd.DataFrame:
    """Real % of adults 25+ with a bachelor's degree or higher, most recent year, from USDA ERS."""
    df = _get_csv(USDA_EDUCATION_CSV, dtype={"FIPS Code": str, "FIPS_Code": str, "FIPStxt": str})
    fips_col = next((c for c in ["FIPS Code", "FIPS_Code", "FIPStxt"] if c in df.columns), None)
    if fips_col is None:
        raise ValueError(f"Couldn't find a FIPS column in USDA education CSV. Columns: {list(df.columns)}")
    # Prefer the most recent 5-yr window's bachelor's-or-higher percentage column.
    # Multiple year windows may be present, so collect all bachelor's columns
    # and select the newest one below.
    candidates = [c for c in df.columns if "bachelor" in c.lower()]
    if not candidates:
        raise ValueError(f"Couldn't find a bachelor's-degree column. Columns: {list(df.columns)}")
    # Column names include the year range, e.g. "Percent of adults with a
    # bachelor's degree or higher, 2019-23" -- take the one with the latest
    # end-year found in its name.
    def end_year(col: str) -> int:
        years = re.findall(r"(\d{4})", col)
        return int(years[-1]) if years else -1
    # max(..., key=...) chooses the candidate with the latest ending year.
    educ_col = max(candidates, key=end_year)
    out = df[[fips_col, educ_col]].rename(columns={fips_col: "FIPS", educ_col: "pct_bachelors_or_higher"})
    out["FIPS"] = out["FIPS"].astype(str).str.zfill(5)
    return out.dropna(subset=["FIPS"])


def fetch_usda_income() -> pd.DataFrame:
    """Real median household income, most recent year, from USDA ERS."""
    df = _get_csv(USDA_INCOME_UNEMPLOYMENT_CSV, dtype={"FIPS_Code": str, "FIPStxt": str})
    fips_col = "FIPS_Code" if "FIPS_Code" in df.columns else "FIPStxt"
    # Accept both USDA's abbreviated and descriptive income field names.
    candidates = [c for c in df.columns if "medhhinc" in c.lower() or ("median" in c.lower() and "income" in c.lower())]
    if not candidates:
        raise ValueError(f"Couldn't find a median household income column. Columns: {list(df.columns)}")
    def end_year(col: str) -> int:
        years = re.findall(r"(\d{4})", col)
        return int(years[-1]) if years else -1
    # Prefer the most recent year when several income measures are included.
    income_col = max(candidates, key=end_year)
    out = df[[fips_col, income_col]].rename(columns={fips_col: "FIPS", income_col: "median_household_income"})
    out["FIPS"] = out["FIPS"].astype(str).str.zfill(5)
    return out.dropna(subset=["FIPS"])


def fetch_usda_rucc() -> pd.DataFrame:
    """Real 2023 Rural-Urban Continuum Codes from USDA ERS."""
    df = _get_csv(USDA_RUCC_CSV, dtype={"FIPS": str})
    fips_col = "FIPS" if "FIPS" in df.columns else _find_column(list(df.columns), ["fips"])
    rucc_col = _find_column(list(df.columns), ["rucc", "2023"], ["rucc"])
    if fips_col is None or rucc_col is None:
        raise ValueError(f"Couldn't find FIPS/RUCC columns in RUCC CSV. Columns: {list(df.columns)}")
    out = df[[fips_col, rucc_col]].rename(columns={fips_col: "FIPS", rucc_col: "rucc_code"})
    out["FIPS"] = out["FIPS"].astype(str).str.zfill(5)
    out["rucc_code"] = pd.to_numeric(out["rucc_code"], errors="coerce")
    return out.dropna(subset=["FIPS", "rucc_code"])


def fetch_census_housing(year: int = 2022, api_key: str | None = None) -> pd.DataFrame:
    """Real median home value & median gross rent for all counties, from
    the Census ACS 5-Year API. No key required for occasional light use;
    pass api_key= a free key (https://api.census.gov/data/key_signup.html)
    if you hit rate limits.
    """
    url = CENSUS_ACS_BASE.format(year=year)
    params = {"get": f"NAME,{CENSUS_HOUSING_VARS}", "for": "county:*"}
    # Supplying the key is optional for light use, so add it only when given.
    if api_key:
        params["key"] = api_key
    resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=60)
    resp.raise_for_status()
    # The API returns one header row followed by one row per county.
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["FIPS"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df = df.rename(columns={
        "B25077_001E": "median_home_value",
        "B25064_001E": "median_gross_rent",
    })
    # Convert text responses to numeric values; malformed values become NA.
    for col in ["median_home_value", "median_gross_rent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Census codes missing/suppressed values as large negative sentinels
    df.loc[df["median_home_value"] < 0, "median_home_value"] = pd.NA
    df.loc[df["median_gross_rent"] < 0, "median_gross_rent"] = pd.NA
    return df[["FIPS", "median_home_value", "median_gross_rent"]].dropna()


def build_real_dataset(
    census_year: int = 2022,
    census_api_key: str | None = None,
    save_to_raw: bool = True,
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and assemble the real USDA + Census tables into the same
    two-table schema the rest of the pipeline (data_merge, metrics,
    analysis, visualization) already expects, so no downstream code
    needs to change when you switch from synthetic to real data.
    """
    # Fetch each source separately, then assemble the same two-table contract
    # used by the synthetic-data path.
    print("[real_data] Fetching USDA ERS poverty data...")
    poverty = fetch_usda_poverty()
    print("[real_data] Fetching USDA ERS education data...")
    education = fetch_usda_education()
    print("[real_data] Fetching USDA ERS income data...")
    income = fetch_usda_income()
    print("[real_data] Fetching USDA ERS Rural-Urban Continuum Codes...")
    rucc = fetch_usda_rucc()
    print(f"[real_data] Fetching Census ACS {census_year} 5-Year housing data...")
    housing = fetch_census_housing(year=census_year, api_key=census_api_key)

    # Keep only counties with every required USDA measure so later metrics do
    # not silently operate on incomplete records.
    usda_df = education.merge(poverty, on="FIPS", how="inner").merge(rucc, on="FIPS", how="inner")
    census_df = income.merge(housing, on="FIPS", how="inner")

    # Attach clean State/County names from the bundled real-county
    # reference table (data/reference/county_fips_reference.csv) rather
    # than relying on whatever column layout each individual USDA file
    # happens to use -- keeps the schema consistent regardless of source
    # file quirks.
    try:
        from src.data_acquisition import load_county_reference
        ref = load_county_reference()[["FIPS", "State", "County"]]
        usda_df = ref.merge(usda_df, on="FIPS", how="inner")
    except Exception as exc:  # pragma: no cover - defensive, non-fatal
        print(f"[real_data] Warning: couldn't attach county names from reference table: {exc}")

    print(f"[real_data] Assembled USDA table: {usda_df.shape}, Census table: {census_df.shape}")

    # Cache successful downloads under the filenames expected by load_or_generate.
    if save_to_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        usda_df.to_csv(raw_dir / "usda_education_poverty.csv", index=False)
        census_df.to_csv(raw_dir / "census_income_housing.csv", index=False)
        print(f"[real_data] Cached real data -> {raw_dir}")

    return usda_df, census_df


if __name__ == "__main__":
    build_real_dataset()
