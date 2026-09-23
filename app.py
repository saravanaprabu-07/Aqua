"""
AquaTrend - Groundwater Level Forecasting & Depletion Trend Platform
Streamlit dashboard (app.py)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import load_raw_data, clean_data, add_time_features
from model import train_and_evaluate
from forecasting import historical_and_forecast
from trend_analysis import run_trend_analysis

st.set_page_config(page_title="AquaTrend", page_icon="💧", layout="wide")

# ----------------------------------------------------------------------
# Cached data / model loading (so we don't retrain on every interaction)
# ----------------------------------------------------------------------

@st.cache_data
def get_clean_data():
    raw = load_raw_data()
    df = clean_data(raw)
    df = add_time_features(df)
    return df


@st.cache_resource
def get_trained_model():
    model, feature_names, results, metrics = train_and_evaluate()
    return model, feature_names, results, metrics


@st.cache_data
def get_trends():
    trends, thresholds = run_trend_analysis()
    return trends, thresholds


df = get_clean_data()
model, feature_names, test_results, metrics = get_trained_model()
trends, thresholds = get_trends()

locations = sorted(df["location"].unique())

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("💧 AquaTrend")
st.caption("Groundwater Level Forecasting & Depletion Trend Platform — built on real CGWB (India) monitoring data")

tab_overview, tab_forecast, tab_trend, tab_seasonal, tab_compare, tab_dataset = st.tabs(
    ["📊 Overview", "🔮 Forecast", "📉 Depletion Trend", "🌦️ Seasonal Analysis", "🔀 Region Comparison", "🗂️ Dataset & Model"]
)

# ----------------------------------------------------------------------
# 1. OVERVIEW
# ----------------------------------------------------------------------
with tab_overview:
    st.subheader("Overview")

    n_locations = df["location"].nunique()
    n_records = len(df)
    date_min, date_max = df["date"].min().date(), df["date"].max().date()
    n_declining = (trends["trend_category"].isin(["Mild Decline", "Significant Decline"])).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Locations (Districts)", n_locations)
    c2.metric("Total Records", f"{n_records:,}")
    c3.metric("Date Range", f"{date_min.year} - {date_max.year}")
    c4.metric("Declining Locations", int(n_declining), delta=None)

    st.markdown(
        "Each **location** is a district-level average of multiple CGWB "
        "monitoring wells. The core measurement is **Depth to Water Level "
        "(metres below ground level)** — a *higher* number means the water "
        "table is *deeper* (i.e. groundwater has declined)."
    )

    st.markdown("#### Trend category distribution across all locations")
    cat_counts = trends["trend_category"].value_counts().reindex(
        ["Improving", "Stable", "Mild Decline", "Significant Decline"]
    ).fillna(0)
    fig = px.bar(
        cat_counts, x=cat_counts.index, y=cat_counts.values,
        color=cat_counts.index,
        color_discrete_map={
            "Improving": "#2E86AB", "Stable": "#8FBF8F",
            "Mild Decline": "#F2A65A", "Significant Decline": "#D64545"
        },
        labels={"x": "Trend Category", "y": "Number of Locations"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Locations by monitoring priority")
    st.dataframe(
        trends[["location", "trend_category", "monitoring_priority", "trend_slope_m_per_quarter"]]
        .sort_values("trend_slope_m_per_quarter", ascending=False)
        .rename(columns={"trend_slope_m_per_quarter": "slope (m/quarter)"})
        .reset_index(drop=True),
        use_container_width=True, height=300,
    )

# ----------------------------------------------------------------------
# 2. GROUNDWATER FORECAST
# ----------------------------------------------------------------------
with tab_forecast:
    st.subheader("Groundwater Forecast")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        location = st.selectbox("Select a location", locations, index=0)
    with col_b:
        n_periods = st.slider("Forecast horizon (quarters ahead)", 1, 8, 4)

    combined = historical_and_forecast(model, feature_names, location, n_periods=n_periods)

    hist_part = combined[combined["type"] == "Historical"]
    fc_part = combined[combined["type"] == "Forecast"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_part["date"], y=hist_part["depth_to_water_level_m"],
        mode="lines+markers", name="Historical (Actual)",
        line=dict(color="#2E86AB", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=fc_part["date"], y=fc_part["depth_to_water_level_m"],
        mode="lines+markers", name="Forecast (ML Predicted)",
        line=dict(color="#D64545", width=2, dash="dash"),
        marker=dict(symbol="diamond"),
    ))
    if len(hist_part) and len(fc_part):
        boundary_date = hist_part["date"].max()
        fig.add_vline(x=boundary_date, line_width=1, line_dash="dot", line_color="gray",
                       annotation_text="Historical ends / Forecast begins", annotation_position="top")

    fig.update_layout(
        title=f"Depth to Water Level — {location}",
        xaxis_title="Date", yaxis_title="Depth to Water Level (m bgl)",
        yaxis=dict(autorange="reversed"),  # deeper depth shown lower on chart = intuitive
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Y-axis is reversed so that 'up' on the chart visually means groundwater getting better (shallower depth), matching intuition.")

    st.markdown("#### Forecast values")
    st.dataframe(
        fc_part[["date", "depth_to_water_level_m"]]
        .rename(columns={"depth_to_water_level_m": "predicted depth (m bgl)"})
        .reset_index(drop=True),
        use_container_width=True,
    )
    st.info(
        "Forecasts are generated recursively by the trained Random Forest model "
        "and are estimates, not guarantees. Uncertainty grows further into the future."
    )

# ----------------------------------------------------------------------
# 3. DEPLETION TREND
# ----------------------------------------------------------------------
with tab_trend:
    st.subheader("Depletion Trend Analysis")

    st.markdown(
        f"""
Trend category is based on a **linear regression slope** (m per quarter) of
depth-to-water-level fitted over each location's full history. Thresholds
are **data-driven**, computed from this dataset's own slope distribution:

- **Improving**: slope < 0
- **Stable**: 0 ≤ slope < {thresholds['mild_threshold']:.4f} (dataset mean)
- **Mild Decline**: {thresholds['mild_threshold']:.4f} ≤ slope < {thresholds['significant_threshold']:.4f} (mean + 1 std dev)
- **Significant Decline**: slope ≥ {thresholds['significant_threshold']:.4f}

A location is additionally flagged **Persistent Decline** if depth worsened
in at least 75% of its last 8 quarter-over-quarter changes.
        """
    )

    sel_location = st.selectbox("Inspect a specific location", locations, key="trend_loc")
    row = trends[trends["location"] == sel_location].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trend Category", row["trend_category"])
    c2.metric("Trend Slope (m/quarter)", f"{row['trend_slope_m_per_quarter']:.4f}")
    c3.metric("Average Depth (m bgl)", f"{row['avg_depth_m']:.2f}")
    c4.metric("Recent Depth (m bgl)", f"{row['recent_depth_m']:.2f}")

    if row["persistent_decline"]:
        st.error(f"⚠️ Persistent decline detected. Monitoring Priority: **{row['monitoring_priority']}**")
    elif row["trend_category"] == "Improving":
        st.success(f"✅ Trend: Improving. Monitoring Priority: **{row['monitoring_priority']}**")
    else:
        st.info(f"Monitoring Priority: **{row['monitoring_priority']}**")

    st.caption("Recommendation: Continue monitoring this location and review extraction/recharge conditions. "
               "This system supports monitoring decisions — it does not control extraction directly.")

    st.markdown("#### Persistent decline locations")
    persistent = trends[trends["persistent_decline"]].sort_values("trend_slope_m_per_quarter", ascending=False)
    st.dataframe(
        persistent[["location", "trend_category", "trend_slope_m_per_quarter", "monitoring_priority"]],
        use_container_width=True,
    )

    st.markdown("#### Full trend table (all locations)")
    st.dataframe(
        trends[["location", "trend_category", "trend_slope_m_per_quarter", "avg_depth_m",
                "recent_depth_m", "persistent_decline", "monitoring_priority"]]
        .sort_values("trend_slope_m_per_quarter", ascending=False),
        use_container_width=True, height=350,
    )

# ----------------------------------------------------------------------
# 4. SEASONAL ANALYSIS
# ----------------------------------------------------------------------
with tab_seasonal:
    st.subheader("Seasonal Analysis")

    seasonal_scope = st.radio("Scope", ["All locations (national average)", "Single location"], horizontal=True)

    if seasonal_scope == "Single location":
        loc_season = st.selectbox("Select location", locations, key="season_loc")
        season_df = df[df["location"] == loc_season]
    else:
        season_df = df

    season_avg = (
        season_df.groupby("season")["depth_to_water_level_m"].mean()
        .reindex(["Winter", "Summer", "Monsoon", "Post-Monsoon"])
    )
    fig_season = px.bar(
        season_avg, x=season_avg.index, y=season_avg.values,
        labels={"x": "Season", "y": "Avg Depth to Water Level (m bgl)"},
        color=season_avg.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_season.update_layout(showlegend=False, title="Average Groundwater Depth by Season")
    st.plotly_chart(fig_season, use_container_width=True)
    st.caption("Lower bars = shallower water table (better). Summer (May) typically shows the deepest water table; "
               "Post-Monsoon (November) typically shows recovery after monsoon recharge.")

    st.markdown("#### Monthly pattern")
    month_avg = season_df.groupby("month")["depth_to_water_level_m"].mean().sort_index()
    fig_month = px.line(month_avg, x=month_avg.index, y=month_avg.values, markers=True,
                         labels={"x": "Month (reading taken)", "y": "Avg Depth (m bgl)"})
    st.plotly_chart(fig_month, use_container_width=True)

    st.info("Rainfall data was not available in the source dataset, so rainfall-vs-groundwater "
            "comparison is not shown. Seasonal patterns above are derived directly from the "
            "quarterly reading months (Jan/May/Aug/Nov), which act as a proxy for dry vs monsoon periods.")

# ----------------------------------------------------------------------
# 5. REGION COMPARISON
# ----------------------------------------------------------------------
with tab_compare:
    st.subheader("Region Comparison")

    default_sel = locations[:3]
    sel_locations = st.multiselect("Select locations to compare", locations, default=default_sel)

    if sel_locations:
        comp_hist = df[df["location"].isin(sel_locations)]
        fig_cmp = px.line(
            comp_hist, x="date", y="depth_to_water_level_m", color="location",
            labels={"depth_to_water_level_m": "Depth to Water Level (m bgl)", "date": "Date"},
        )
        fig_cmp.update_yaxes(autorange="reversed")
        fig_cmp.update_layout(title="Historical Groundwater Depth — Region Comparison")
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("#### Comparison table")
        comp_table = trends[trends["location"].isin(sel_locations)][
            ["location", "avg_depth_m", "recent_depth_m", "trend_slope_m_per_quarter", "trend_category"]
        ].rename(columns={"trend_slope_m_per_quarter": "slope (m/quarter)"})
        st.dataframe(comp_table, use_container_width=True)

        fig_bar = px.bar(
            comp_table, x="location", y="slope (m/quarter)", color="trend_category",
            color_discrete_map={
                "Improving": "#2E86AB", "Stable": "#8FBF8F",
                "Mild Decline": "#F2A65A", "Significant Decline": "#D64545"
            },
            title="Trend Slope Comparison",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Select at least one location to compare.")

# ----------------------------------------------------------------------
# 6. DATASET & MODEL
# ----------------------------------------------------------------------
with tab_dataset:
    st.subheader("Dataset & Model")

    st.markdown("""
**Dataset:** Central Ground Water Board (CGWB) India — Quarterly Depth to Water Level
**Source:** https://github.com/craigdsouza/cgwb (compiled from official CGWB records)
**Processed to:** district-level quarterly time series, 60 districts, 1996–2017
(see `dataset_source.txt` for the full processing description)
    """)

    st.markdown("#### Features used")
    st.markdown("""
- `lag_1`, `lag_2`, `lag_4` — depth values from 1, 2, and 4 quarters ago
- `rolling_avg_4` — rolling mean depth over the trailing 4 quarters (~1 year)
- `quarter_index` — sequential time counter for the location
- `year`, `month`, `season` — calendar/seasonal position
- `location` (one-hot) — district identity
- `num_wells_reporting` — number of CGWB wells behind each district-quarter reading
    """)

    st.markdown("#### Model")
    st.markdown("""
**Algorithm:** Random Forest Regressor (scikit-learn)
**Train/Test split:** Chronological — for every location, the most recent 8 quarters
(2 years) are held out as the test set; all earlier quarters are used for training.
This is **not** a random split, because random splitting would leak future
information into training for a time-series problem.
    """)

    st.markdown("#### Evaluation results (on the held-out chronological test set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.3f} m")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f} m")
    m3.metric("R²", f"{metrics['R2']:.3f}")
    st.caption(f"Trained on {metrics['n_train']:,} rows, tested on {metrics['n_test']:,} rows "
               f"({metrics['test_quarters_per_location']} most recent quarters per location).")

    st.markdown("#### Actual vs. predicted (test set sample)")
    st.dataframe(test_results.head(30), use_container_width=True)

    fig_scatter = px.scatter(
        test_results, x="actual_depth_m", y="predicted_depth_m",
        opacity=0.5, labels={"actual_depth_m": "Actual Depth (m bgl)", "predicted_depth_m": "Predicted Depth (m bgl)"},
        title="Actual vs. Predicted Depth (Test Set)",
    )
    min_v = min(test_results["actual_depth_m"].min(), test_results["predicted_depth_m"].min())
    max_v = max(test_results["actual_depth_m"].max(), test_results["predicted_depth_m"].max())
    fig_scatter.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                                      name="Perfect prediction", line=dict(dash="dash", color="gray")))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.caption("AquaTrend — built for a Level 1 ML Hackathon. Predictions are model estimates for monitoring "
            "support only, not guarantees or real-world control actions.")
