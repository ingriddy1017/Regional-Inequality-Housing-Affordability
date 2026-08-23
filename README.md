# Regional Socioeconomic Inequality & Housing Affordability

**Core Question:** How do regional education levels impact local housing affordability and median household income?

This repo analyzes county-level socioeconomic data to explore relationships between
educational attainment, poverty, median household income, and housing costs (rent /
home value) across U.S. counties — while explicitly controlling for the
urban-vs-rural confound described below.

---

## 1. Project Structure

```
regional-socioeconomic-inequality/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # Raw downloaded CSVs go here (see data/README.md)
│   ├── reference/           # Bundled real ~3,221-county FIPS/name/state table
│   ├── processed/           # Cleaned/merged output lands here
│   └── README.md            # Data sources & download instructions
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py  # Load real CSVs, or generate a dense synthetic demo dataset
│   ├── real_data.py         # Live fetchers for real USDA ERS + Census ACS data (no signup)
│   ├── data_merge.py        # pd.merge() of Education / Income / Housing tables on FIPS
│   ├── metrics.py           # Price-to-Income Ratio, Rent Burden %, correlation matrices
│   ├── analysis.py          # Urban/Rural grouped analysis (RUCC), regression helpers
│   └── visualization.py     # seaborn heatmaps + plotly choropleth maps
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── outputs/
│   ├── figures/             # Saved PNG/HTML charts and maps
│   └── tables/              # Saved CSV summary tables
├── tests/
│   └── test_metrics.py      # Unit tests (FIPS padding, ratio math, merge integrity)
└── main.py                  # End-to-end pipeline entry point
```

## 2. Pipeline Phases

| Phase | Difficulty | Key Tools | What happens |
|---|---|---|---|
| 1. Data Acquisition | Low | CSV / API | Load County-level CSVs from USDA ERS or Census `data.census.gov` (or generate a synthetic demo set) |
| 2. Merging Datasets | Low | `pandas.merge()` | Join Income, Education, and Housing tables on the 5-digit FIPS code |
| 3. Metric Engineering | Easy | `pandas` math | Compute Price-to-Income Ratio, % Rent Burdened, etc. |
| 4. Visualization | Medium | `seaborn` / `plotly` | Correlation heatmaps + interactive US choropleth map |

## 3. Quickstart

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Option A: synthetic demo data (default). If data/raw/*.csv are missing,
# a dense, realistic synthetic dataset covering all ~3,200 REAL US
# county FIPS codes is generated automatically -- so the choropleth maps
# render fully shaded, not scattered dots, right out of the box.
python main.py

# Option B: real data. Fetches actual county-level education/poverty/
# income from USDA ERS and actual home value/rent from the Census ACS
# API -- both free, no signup required for light use.
python main.py --real-data
```

Outputs land in `outputs/figures/` (correlation heatmap PNG + interactive
choropleth HTML) and `outputs/tables/` (merged & summary CSVs).

### Why the map used to look sparse

Two things had to be true for a choropleth to render as filled counties
instead of scattered dots: (1) enough counties, and (2) FIPS codes that
match real county polygons. The original synthetic generator only
created ~500 counties with fabricated FIPS codes, so only a handful
happened to collide with real ones. It now draws all ~3,200 counties
from a bundled reference table of **real** FIPS codes
(`data/reference/county_fips_reference.csv`, derived from the Census
county boundary set), so every row -- synthetic values or real -- maps
to an actual county and the choropleth is fully dense.

### Using real data

`python main.py --real-data` (or `python -m src.real_data` to just fetch
and cache the CSVs) pulls:
- Education, poverty, and median household income from **USDA ERS**'s
  direct CSV downloads (updated annually, no signup)
- 2023 Rural-Urban Continuum Codes, also from USDA ERS
- Median home value and median gross rent from the **Census ACS 5-Year
  API** (free; a key is recommended for anything beyond light/occasional
  use -- get one at https://api.census.gov/data/key_signup.html and pass
  it to `real_data.build_real_dataset(census_api_key=...)`)

USDA occasionally renames columns to bake in the data year (e.g.
`PCTPOVALL_2023`), so `real_data.py` matches on stable substrings rather
than exact column names -- if a fetch ever fails after USDA reorganizes
a file, the error message lists the actual columns found so you can
adjust the keyword match in `_find_column()`.

## 4. Key Pitfalls This Project Explicitly Addresses

1. **Confounding Variables (Urban vs. Rural).** High-density urban counties
   tend to have both high degree attainment *and* high housing costs. Naively
   correlating education with home prices nationwide mostly just proves
   "cities are expensive." `src/analysis.py` groups/filters by USDA
   **Rural-Urban Continuum Codes (RUCC)** and reports correlations *within*
   each urban/rural tier so the education-affordability relationship isn't
   confounded with urbanicity.

2. **Medians vs. Means.** All income and home-value figures used are
   **medians** (`median_household_income`, `median_home_value`,
   `median_gross_rent`), never means, since a handful of extreme-wealth
   outliers can badly distort an average.

3. **FIPS Code Formatting.** FIPS codes are always read and merged as
   zero-padded 5-character **strings** (`dtype={'FIPS': str}` +
   `.str.zfill(5)`), so a county like Alabama's `01001` doesn't silently
   become the integer `1001` and break the join. See
   `src/data_merge.py::normalize_fips`.

## 5. Metrics Engineered

- **Price-to-Income Ratio** = `median_home_value / median_household_income`
- **Rent Burden %** = `(median_gross_rent * 12) / median_household_income * 100`
- **Poverty–Education correlation** (overall and grouped by RUCC tier)

## 6. License

MIT — see `LICENSE` (add one appropriate to your use case).
