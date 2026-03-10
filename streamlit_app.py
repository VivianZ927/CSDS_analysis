import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Test Trends Dashboard", layout="wide")


def load_data(xlsx_path: str, sheet_name: str | None = None) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # Standardize column names (trim spaces/newlines)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    # Drop rows with no timestamp
    if "SAMPLE DATE/TIME" in df.columns:
        df = df.dropna(subset=["SAMPLE DATE/TIME"]).copy()
        df["SAMPLE DATE/TIME"] = pd.to_datetime(df["SAMPLE DATE/TIME"], errors="coerce")
        df = df.dropna(subset=["SAMPLE DATE/TIME"])
    else:
        raise ValueError("Expected a 'SAMPLE DATE/TIME' column in the dataset.")

    # Ensure numeric
    value_cols = [c for c in df.columns if c != "SAMPLE DATE/TIME"]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["SAMPLE DATE/TIME"]
    out["Date"] = dt.dt.date
    out["Year"] = dt.dt.year
    out["Month"] = dt.dt.month
    out["MonthName"] = dt.dt.strftime("%b")


    # Meteorological seasons (Northern Hemisphere)
    def season_from_month(m: int) -> str:
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"

    out["Season"] = out["Month"].map(season_from_month)
    # Useful for ordering in charts
    out["SeasonOrder"] = pd.Categorical(out["Season"], categories=["Spring", "Summer", "Autumn","Winter"], ordered=True)
    return out

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["SAMPLE DATE/TIME", "Date", "Year", "Month", "MonthName", "Season", "SeasonOrder"]
    value_cols = [c for c in df.columns if c not in id_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="Test", value_name="Value")
    long_df = long_df.dropna(subset=["Value"])
    return long_df

# ---------- Sidebar ----------
st.title("Test Trends & Seasonality Dashboard")

with st.sidebar:
    default_path = "CS DS.xlsx"
    st.header("Filters")

try:
    df_raw = load_data(default_path, 'Sheet1')
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

df = add_time_features(df_raw)
long_df = to_long(df)

with st.sidebar:
    min_dt = pd.to_datetime(df["SAMPLE DATE/TIME"].min()).to_pydatetime()
    max_dt = pd.to_datetime(df["SAMPLE DATE/TIME"].max()).to_pydatetime()
    st.subheader("Date range")
    st.caption(f"{min_dt.strftime('%d %b %Y')} → {max_dt.strftime('%d %b %Y')}")
    # date_range = st.date_input("Date range", value=(min_dt.date(), max_dt.date()), min_value=min_dt.date(), max_value=max_dt.date())

    tests = sorted(long_df["Test"].unique().tolist())
    default_tests = tests[: min(5, len(tests))]
    selected_tests = st.multiselect("Tests", options=tests, default=default_tests)

    agg = st.selectbox("Aggregation", options=["Raw", "Weekly", "Monthly"], index=2)

if not selected_tests:
    st.warning("Select at least one test from the sidebar.")
    st.stop()

start_date, end_date = min_dt.date(), max_dt.date()
mask = (df["SAMPLE DATE/TIME"].dt.date >= start_date) & (df["SAMPLE DATE/TIME"].dt.date <= end_date)
df_f = df.loc[mask].copy()
long_f = long_df.loc[long_df["Test"].isin(selected_tests)].copy()
long_f = long_f.loc[(long_f["SAMPLE DATE/TIME"].dt.date >= start_date) & (long_f["SAMPLE DATE/TIME"].dt.date <= end_date)].copy()

# Aggregation
def agg_frame(long_df_in: pd.DataFrame, how: str) -> pd.DataFrame:
    out = long_df_in.copy()
    if how == "Raw":
        out["Bucket"] = out["SAMPLE DATE/TIME"]
        out["BucketLabel"] = out["SAMPLE DATE/TIME"].dt.strftime("%Y-%m-%d")
        return out
    if how == "Weekly":
        out["Bucket"] = out["SAMPLE DATE/TIME"].dt.to_period("W").dt.start_time
        out["BucketLabel"] = out["Bucket"].dt.strftime("%Y-%m-%d")
    elif how == "Monthly":
        out["Bucket"] = out["SAMPLE DATE/TIME"].dt.to_period("M").dt.start_time
        out["BucketLabel"] = out["Bucket"].dt.strftime("%Y-%m")


    g = out.groupby(["Test", "Bucket", "BucketLabel"], as_index=False)["Value"].mean()
    return g

trend_df = agg_frame(long_f, agg)

# ---------- Layout ----------
tab1, tab2 = st.tabs(["Trends by Test", "Seasonal Trends"])

with tab1:
    st.subheader("Trend lines over time")

    # If multiple tests have vastly different magnitudes, let user choose normalization
    col_a, col_b = st.columns([1, 1])
    with col_a:
        normalize = st.checkbox("Normalize per test (z-score)", value=False)
    with col_b:
        show_points = st.checkbox("Show points", value=True)

    plot_df = trend_df.copy()

    if normalize:
        # z-score within each test
        plot_df["Value"] = plot_df.groupby("Test")["Value"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0))

    base = alt.Chart(plot_df).encode(
        x=alt.X("Bucket:T", title="Time"),
        y=alt.Y("Value:Q", title="Value" + (" (z-score)" if normalize else "")),
        color=alt.Color("Test:N", legend=alt.Legend(title="Test")),
        tooltip=[
            alt.Tooltip("Test:N"),
            alt.Tooltip("Bucket:T", title="Time"),
            alt.Tooltip("Value:Q", format=".4g"),
        ],
    )

    line = base.mark_line()
    chart = line

    if show_points:
        chart = (line + base.mark_circle(size=60, opacity=0.65))

    st.altair_chart(chart.interactive(), use_container_width=True)

    st.divider()
    st.subheader("Summary stats (selected range)")
    stats = (
        long_f.groupby("Test")["Value"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .sort_values("Test")
    )
    st.dataframe(stats, use_container_width=True)

with tab2:
    st.subheader("Seasonality")

    # Choose a single test for deeper seasonal analysis
    single_test = st.selectbox("Pick one test for seasonality details", options=tests, index=tests.index(selected_tests[0]) if selected_tests else 0)
    season_df = long_df.loc[long_df["Test"] == single_test].copy()
    season_df = season_df.loc[(season_df["SAMPLE DATE/TIME"].dt.date >= start_date) & (season_df["SAMPLE DATE/TIME"].dt.date <= end_date)].copy()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Distribution by season**")
        box = (
            alt.Chart(season_df)
            .mark_boxplot()
            .encode(
                x=alt.X("SeasonOrder:N", title="Season", sort=["Spring", "Summer", "Autumn","Winter"]),
                y=alt.Y("Value:Q", title="Value"),
                tooltip=[alt.Tooltip("Season:N"), alt.Tooltip("Value:Q", format=".4g")],
            )
        )
        st.altair_chart(box, use_container_width=True)

    with c2:
        st.markdown("**Seasonal mean by year**")
        seasonal_mean = (
            season_df.groupby(["Year", "Season", "SeasonOrder"], as_index=False)["Value"].mean()
        )
        line = (
            alt.Chart(seasonal_mean)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Value:Q", title="Mean value"),
                color=alt.Color("SeasonOrder:N", title="Season", sort=[ "Spring", "Summer", "Autumn","Winter"]),
                tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Season:N"), alt.Tooltip("Value:Q", format=".4g")],
            )
        )
        st.altair_chart(line, use_container_width=True)

    st.divider()
    st.markdown("**Seasonal comparison for the selected test (mean)**")

    comp = long_df.loc[long_df["Test"] == single_test].copy()
    comp = comp.loc[
        (comp["SAMPLE DATE/TIME"].dt.date >= start_date) & (comp["SAMPLE DATE/TIME"].dt.date <= end_date)].copy()
    comp_mean = comp.groupby(["Season", "SeasonOrder"], as_index=False)["Value"].mean()

    bar = (
        alt.Chart(comp_mean)
        .mark_bar()
        .encode(
            x=alt.X("SeasonOrder:N", title="Season", sort=["Spring", "Summer", "Autumn", "Winter"]),
            y=alt.Y("Value:Q", title="Mean value"),
            tooltip=[alt.Tooltip("Season:N"), alt.Tooltip("Value:Q", format=".4g")],
        )
    )
    st.altair_chart(bar, use_container_width=True)



st.caption("Seasons use the meteorological definition for the Northern Hemisphere: Winter (Dec–Feb), Spring (Mar–May), Summer (Jun–Aug), Autumn (Sep–Nov).")
