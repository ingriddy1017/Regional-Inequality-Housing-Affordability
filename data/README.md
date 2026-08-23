# Data Sources

This project expects three county-level CSVs in `data/raw/`. If they are
absent, `main.py` will automatically generate a **synthetic but
realistically-structured** demo dataset (same schema, plausible value
ranges and correlations, and *real* county FIPS codes for full map
density) so the pipeline can be exercised without any downloads. Swap in
real data at any time — the code doesn't change.

**Fastest path to real data:** run `python main.py --real-data`, which
calls `src/real_data.py` to fetch and cache the real files below
automatically (USDA ERS direct downloads + the free Census ACS API, no
signup required). The manual instructions below are for when you want a
specific year/vintage or to inspect the files yourself.

## 1. Education & Poverty — USDA Economic Research Service

- Source: USDA ERS "County-level Data Sets" (Education, Poverty, Population,
  Unemployment): https://www.ers.usda.gov/data-products/county-level-data-sets/
- Direct downloads (as of this writing — check the page above for the latest):
  - Poverty: https://www.ers.usda.gov/media/5496/poverty-estimates-for-the-united-states-states-and-counties-2023.csv
  - Education: https://www.ers.usda.gov/media/5495/educational-attainment-for-adults-age-25-and-older-for-the-united-states-states-and-counties-1970-2023.csv
  - Income/Unemployment: https://www.ers.usda.gov/media/5497/unemployment-and-median-household-income-for-the-united-states-states-and-counties-2000-23.csv
  - Rural-Urban Continuum Codes (2023): https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv
- Save as: `data/raw/usda_education_poverty.csv`
- Expected columns (rename to these after download):
  - `FIPS` — 5-digit county FIPS code (string, zero-padded)
  - `State`
  - `County`
  - `pct_bachelors_or_higher` — % of adults 25+ with a bachelor's degree or higher
  - `poverty_rate` — % of population below the poverty line
  - `rucc_code` — USDA Rural-Urban Continuum Code (1–9; 1–3 = metro, 4–9 = nonmetro)

## 2. Median Household Income & Housing — US Census Bureau

- Source: Census Bureau QuickFacts / American Community Survey (ACS) 5-Year
  Estimates: https://data.census.gov/ (tables B19013, B25077, B25064), or
  the free ACS API directly: `https://api.census.gov/data/2022/acs/acs5?get=NAME,B25077_001E,B25064_001E&for=county:*`
- Save as: `data/raw/census_income_housing.csv`
- Expected columns:
  - `FIPS`
  - `median_household_income`
  - `median_home_value`
  - `median_gross_rent` (monthly)

## 3. (Optional) County Boundary / Geometry data for `geopandas`

- Source: US Census Bureau Cartographic Boundary Files:
  https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
- Save as: `data/raw/cb_counties.zip` (shapefile) — only needed if you
  swap the `plotly` choropleth for a `geopandas` static map.

## Why FIPS as a string?

Pandas will infer FIPS as an integer by default, which silently drops
leading zeros (`01001` → `1001`). Always read with:

```python
pd.read_csv(path, dtype={"FIPS": str})
```

and normalize with `.str.zfill(5)` — see `src/data_merge.py`.
