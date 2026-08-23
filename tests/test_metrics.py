"""
Unit tests covering the pitfalls this project is designed to avoid:
 - FIPS zero-padding survives round-tripping through pandas
 - merge doesn't silently duplicate or lose rows
 - engineered ratios use the correct (median-based) formulas
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_merge, metrics


def test_normalize_fips_pads_leading_zero():
    df = pd.DataFrame({"FIPS": [1001, 6037, "1003"]})
    out = data_merge.normalize_fips(df)
    assert out["FIPS"].tolist() == ["01001", "06037", "01003"]


def test_normalize_fips_handles_float_strings():
    df = pd.DataFrame({"FIPS": ["1001.0", "6037.0"]})
    out = data_merge.normalize_fips(df)
    assert out["FIPS"].tolist() == ["01001", "06037"]


def test_merge_datasets_inner_join_no_duplicates():
    usda = pd.DataFrame({
        "FIPS": ["01001", "06037", "48201"],
        "pct_bachelors_or_higher": [20.0, 45.0, 30.0],
        "poverty_rate": [18.0, 12.0, 15.0],
        "rucc_code": [6, 1, 2],
    })
    census = pd.DataFrame({
        "FIPS": ["01001", "06037"],  # 48201 intentionally missing
        "median_household_income": [42000, 75000],
        "median_home_value": [110000, 650000],
        "median_gross_rent": [700, 1800],
    })
    merged = data_merge.merge_datasets(usda, census, how="inner")
    assert len(merged) == 2
    assert set(merged["FIPS"]) == {"01001", "06037"}


def test_price_to_income_ratio():
    df = pd.DataFrame({
        "median_home_value": [200000],
        "median_household_income": [50000],
    })
    out = metrics.add_price_to_income_ratio(df)
    assert out["price_to_income_ratio"].iloc[0] == 4.0


def test_rent_burden_pct():
    df = pd.DataFrame({
        "median_gross_rent": [1000],
        "median_household_income": [48000],
    })
    out = metrics.add_rent_burden_pct(df)
    # (1000*12)/48000*100 = 25.0
    assert out["rent_burden_pct"].iloc[0] == pytest.approx(25.0)


def test_urban_flag():
    df = pd.DataFrame({"rucc_code": [1, 3, 4, 9]})
    out = metrics.add_urban_flag(df)
    assert out["is_metro"].tolist() == [True, True, False, False]


def test_correlation_matrix_only_uses_available_columns():
    df = pd.DataFrame({
        "pct_bachelors_or_higher": [10, 20, 30],
        "poverty_rate": [30, 20, 10],
    })
    corr = metrics.correlation_matrix(df)
    assert corr.shape == (2, 2)
    assert corr.loc["pct_bachelors_or_higher", "poverty_rate"] == pytest.approx(-1.0)
