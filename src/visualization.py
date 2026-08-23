"""
visualization.py

Phase 4: Visualization
------------------------
- seaborn correlation heatmap (static PNG)
- plotly interactive US county choropleth map (HTML), using FIPS as the
  location key (kept as zero-padded strings the whole way through, per
  Pitfall #3)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    save_path: str | Path,
    title: str = "Correlation Matrix: Education, Poverty, Income & Housing",
) -> None:
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title(title, fontsize=13, pad=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualization] Saved heatmap -> {save_path}")


def plot_choropleth(
    df: pd.DataFrame,
    value_column: str,
    save_path: str | Path,
    fips_column: str = "FIPS",
    title: str = "County-Level Metric Map",
    color_scale: str = "Viridis",
) -> None:
    """Interactive choropleth map of a metric by county, via Plotly.

    Requires the `plotly.express` built-in county GeoJSON, matched to
    counties by their (string, zero-padded) FIPS code.
    """
    import plotly.express as px

    df = df.copy()
    df[fips_column] = df[fips_column].astype(str).str.zfill(5)

    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations=fips_column,
        color=value_column,
        color_continuous_scale=color_scale,
        scope="usa",
        hover_data=["State", "County"] if {"State", "County"}.issubset(df.columns) else None,
        labels={value_column: value_column.replace("_", " ").title()},
    )
    fig.update_layout(
        title=title,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"[visualization] Saved choropleth -> {save_path}")


def plot_choropleth_static_fallback(
    df: pd.DataFrame,
    value_column: str,
    save_path: str | Path,
    fips_column: str = "FIPS",
    title: str = "County-Level Metric Map",
) -> None:
    """Static geopandas-based alternative to the Plotly choropleth.

    Only used if you'd rather render a static PNG map (e.g., for a
    printed report) instead of an interactive HTML map. Requires
    geopandas + a county boundary shapefile — see data/README.md,
    section 3. Left as an opt-in alternative since geopandas adds a
    heavier dependency (GDAL) than the default plotly path.
    """
    import geopandas as gpd

    counties = gpd.read_file(Path(__file__).resolve().parents[1] / "data" / "raw" / "cb_counties.zip")
    counties["FIPS"] = counties["STATEFP"] + counties["COUNTYFP"]

    df = df.copy()
    df[fips_column] = df[fips_column].astype(str).str.zfill(5)
    merged = counties.merge(df, left_on="FIPS", right_on=fips_column, how="inner")

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    merged.plot(column=value_column, cmap="viridis", legend=True, ax=ax, edgecolor="white", linewidth=0.1)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualization] Saved static geopandas map -> {save_path}")
