"""WeatherGuard Streamlit dashboard.

This dashboard provides a practical reliability layer over NWP forecasts by
combining generated forecast features with a trained Random Forest classifier.
It is intentionally designed to visualize risk on a national map and present a
clear safety-oriented forecast passport for operational usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_engine import generate_historical_dataset, get_live_inference_grid
from src.model_engine import WeatherGuardEngine


st.set_page_config(
    page_title="WeatherGuard",
    page_icon="🌦️",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0f172a 100%);
        color: #e2e8f0;
    }
    .topbar {
        background: linear-gradient(135deg, #1E3A8A, #0F172A);
        padding: 1.2rem 1.5rem;
        border-radius: 0.9rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.22);
    }
    .topbar h1 {
        color: #F8FAFC !important;
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
    }
    .topbar .subtitle {
        color: #DDE8FF !important;
        margin-top: 0.35rem;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.92));
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 0.9rem;
        padding: 0.8rem 1rem;
        box-shadow: 0 6px 18px rgba(14, 116, 144, 0.15);
    }
    .kpi-card .label {
        color: #cbd5e1;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-card .value {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.92);
        color: #dfeafc;
        border-radius: 0.7rem 0.7rem 0 0;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #1E3A8A, #0F172A);
        color: white;
    }
    .stDataFrame {
        background: rgba(15, 23, 42, 0.75);
        border-radius: 0.8rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 0.8rem;
        padding: 0.5rem 0.75rem;
    }
    .stPlotlyChart > div {
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 0.9rem;
        overflow: hidden;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="topbar">
        <h1>WeatherGuard</h1>
        <div class="subtitle">AI-Based Forecast Reliability Layer · SIH Problem Statement 26079</div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_engine() -> WeatherGuardEngine:
    """Load a fitted WeatherGuard model once per session."""
    engine = WeatherGuardEngine()
    engine.fit()
    return engine


@st.cache_data(show_spinner=False)
def get_forecast_snapshot(lead_time: int, region: str) -> pd.DataFrame:
    """Generate a region/lead-time-specific inference snapshot."""
    return get_live_inference_grid(lead_time=lead_time, selected_region=region)


@st.cache_data(show_spinner=False)
def compute_decay_curve(region: str) -> pd.DataFrame:
    """Compute average reliability by lead time for a selected region."""
    engine = load_engine()
    rows = []
    for lead in range(1, 11):
        df = get_forecast_snapshot(lead, region)
        predictions = engine.predict_live(df)
        reliability = predictions["reliability_score"].mean()
        rows.append({"lead_time": lead, "average_reliability": round(float(reliability), 2)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_backtest_snapshot(lead_time: int, region: str) -> pd.DataFrame:
    """Return historical ERA5 verification rows for a given lead time and region."""
    historical = generate_historical_dataset(n_samples=3000)
    if region != "ALL":
        historical = historical[historical["region"] == region].copy()
    backtest = historical[historical["lead_time"] == lead_time].copy()
    if backtest.empty:
        return pd.DataFrame()
    return backtest.sort_values("absolute_error_mm", ascending=False).head(10).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_demo_row() -> pd.Series:
    """Select one reproducible generated D5 Andhra-region bust for the demo."""
    historical = generate_historical_dataset(n_samples=2500)
    candidates = historical[
        (historical["lead_time"] == 5)
        & historical["lat"].between(12.0, 20.0)
        & historical["lon"].between(76.0, 85.0)
        & (historical["is_bust"] == 1)
    ].copy()
    candidates["demo_distance"] = (
        (candidates["forecast_rain_mm"] - 28.0).abs()
        + (candidates["era5_reference_rain_mm"] - 7.0).abs()
    )
    return candidates.sort_values("demo_distance").iloc[0].drop(labels=["demo_distance"])


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table for each region in a grid-level result set."""
    summary = df.groupby("region", as_index=False).agg(
        avg_reliability=("reliability_score", "mean"),
        bust_probability=("bust_probability", "mean"),
        high_risk_cells=("bust_alert", "sum"),
        avg_error=("estimated_error_mm", "mean"),
    )
    summary["avg_reliability"] = summary["avg_reliability"].round(2)
    summary["bust_probability"] = summary["bust_probability"].round(3)
    summary["avg_error"] = summary["avg_error"].round(2)
    return summary.sort_values("avg_reliability")


def render_evidence_tab(engine: WeatherGuardEngine) -> None:
    """Render one reproducible forecast-to-alert example and held-out metrics."""
    st.subheader("🔎 End-to-End Evidence")
    st.caption("All values below are generated by the current prototype; no model numbers are hand-entered.")

    demo = get_demo_row()
    demo_input = demo[engine.feature_columns + ["grid_id", "region"]].to_frame().T
    prediction = engine.predict_live(demo_input).iloc[0]
    expected_error = float(demo["absolute_error_mm"])
    reference_rain = float(demo["era5_reference_rain_mm"])

    st.markdown("### Forecast → WeatherGuard → Alert → Why")
    input_cols = st.columns(4)
    input_cols[0].metric("Location", "Andhra region")
    input_cols[1].metric("Lead time", f"D{int(demo['lead_time'])}")
    input_cols[2].metric("Forecast rainfall", f"{demo['forecast_rain_mm']:.2f} mm")
    input_cols[3].metric("Reference rainfall", f"{reference_rain:.2f} mm")

    result_cols = st.columns(5)
    result_cols[0].metric("Expected error", f"{expected_error:.2f} mm")
    result_cols[1].metric("Bust threshold", ">15 mm")
    result_cols[2].metric("Bust", "YES" if int(demo["is_bust"]) else "NO")
    result_cols[3].metric("Model bust probability", f"{prediction['bust_probability']:.2%}")
    result_cols[4].metric("Reliability", f"{prediction['reliability_score']:.1f}%")

    st.info(
        f"Alert: {'TRIGGERED' if bool(prediction['bust_alert']) else 'not triggered'} | "
        f"Why: {prediction['why_unreliable']}"
    )
    st.caption(
        f"Generated cell: {demo['lat']:.2f}°N, {demo['lon']:.2f}°E | "
        f"Rainfall rule: forecast >10 mm and error >15 mm | "
        f"Model threshold: {engine.threshold:.3f}"
    )

    st.markdown("### Held-out model results")
    metrics = engine.evaluate()
    metric_cols = st.columns(5)
    metric_cols[0].metric("Precision", f"{metrics['precision']:.3f}")
    metric_cols[1].metric("Recall", f"{metrics['recall']:.3f}")
    metric_cols[2].metric("F1", f"{metrics['f1']:.3f}")
    metric_cols[3].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    metric_cols[4].metric("Brier score", f"{metrics['brier_score']:.3f}")

    counts = pd.DataFrame(
        {
            "Class": ["Bust", "Non-bust"],
            "Samples": [metrics["bust_samples"], metrics["non_bust_samples"]],
        }
    )
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual non-bust", "Actual bust"],
        columns=["Predicted non-bust", "Predicted bust"],
    )
    count_col, matrix_col = st.columns(2)
    with count_col:
        st.dataframe(counts, use_container_width=True, hide_index=True)
    with matrix_col:
        st.dataframe(matrix, use_container_width=True)
    st.caption("Brier score measures probability calibration; lower is better. Metrics use a held-out test partition.")


def render_map_tab(engine: WeatherGuardEngine, lead_time: int, region: str) -> None:
    """Render the national map tab."""
    st.subheader("🗺️ National Reliability Map")
    col_a, col_b = st.columns([2.5, 1])

    with col_a:
        current_df = get_forecast_snapshot(lead_time, region)
        predictions = engine.predict_live(current_df)
        fig = go.Figure(
            go.Scattergeo(
                lat=predictions["lat"],
                lon=predictions["lon"],
                mode="markers",
                marker=dict(
                    size=np.clip(predictions["bust_probability"] * 18 + 6, 6, 22),
                    color=predictions["reliability_score"],
                    colorscale="RdYlGn_r",
                    cmin=0,
                    cmax=100,
                    opacity=0.86,
                    line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                    colorbar=dict(title="Reliability (%)"),
                ),
                text=predictions["region"],
                hovertemplate=(
                    "Region: %{text}<br>"
                    "Grid ID: %{customdata[0]}<br>"
                    "Lat: %{lat}<br>"
                    "Lon: %{lon}<br>"
                    "Lead Time: %{customdata[1]} days<br>"
                    "Bust Probability: %{customdata[2]:.3f}<br>"
                    "Reliability: %{customdata[3]:.1f}%<br>"
                    "Expected Error: %{customdata[4]:.1f} mm<br>"
                    "Diagnostic: %{customdata[5]}<extra></extra>"
                ),
                customdata=predictions[
                    [
                        "grid_id",
                        "lead_time",
                        "bust_probability",
                        "reliability_score",
                        "estimated_error_mm",
                        "why_unreliable",
                    ]
                ].values,
            )
        )
        fig.update_geos(
            scope="asia",
            projection_type="mercator",
            center=dict(lat=22.5, lon=78.5),
            lataxis_range=[6, 38],
            lonaxis_range=[68, 98],
            showcountries=True,
            countrycolor="rgb(45, 81, 143)",
            countrywidth=1.1,
            showcoastlines=True,
            coastlinecolor="rgb(67, 90, 117)",
            showland=True,
            landcolor="rgb(237, 242, 245)",
            showocean=True,
            oceancolor="rgb(255, 255, 255)",
            showframe=False,
            bgcolor="rgba(255,255,255,0.9)",
            resolution=110,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            geo=dict(
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Regional Drill-down")
        summary = region_summary(predictions)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### System Health")
        avg_rel = predictions["reliability_score"].mean()
        risk_cells = int(predictions["bust_alert"].sum())
        worst = predictions.sort_values("bust_probability", ascending=False).head(1)

        st.metric("National Reliability", f"{avg_rel:.1f}%")
        st.metric("High-Risk Grid Cells", f"{risk_cells}")
        if not worst.empty:
            st.metric(
                "Peak Bust Risk",
                f"{worst['bust_probability'].iloc[0]:.2%}",
            )
            st.caption(
                f"Lat {worst['lat'].iloc[0]:.2f}, Lon {worst['lon'].iloc[0]:.2f} | "
                f"{worst['why_unreliable'].iloc[0]}"
            )


def render_decay_tab() -> None:
    """Render lead-time decay curve."""
    st.subheader("📈 Lead-Time Decay Curve")
    region = st.selectbox("Compare region", ["ALL", "North", "South", "East", "West", "Central"], index=0)
    curve = compute_decay_curve(region)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["lead_time"],
            y=curve["average_reliability"],
            mode="lines+markers",
            line=dict(color="#0F766E", width=3),
            marker=dict(size=8, color=curve["average_reliability"], colorscale="RdYlGn_r", showscale=True),
            name=f"{region} reliability",
        )
    )
    fig.update_layout(
        title="Forecast reliability vs. lead time",
        xaxis_title="Lead Time (days)",
        yaxis_title="Average Reliability (%)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(range=[0, 100]),
        coloraxis_colorbar=dict(title="Reliability (%)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add a small performance table for interpretation.
    st.dataframe(curve, use_container_width=True, hide_index=True)


def _normalize_passport_columns(passport: pd.DataFrame) -> pd.DataFrame:
    """Normalize optional historical/live field names so the passport can render safely."""
    renamed = passport.copy()

    if "forecast_rain_mm" not in renamed.columns:
        if "forecast_rain" in renamed.columns:
            renamed = renamed.rename(columns={"forecast_rain": "forecast_rain_mm"})
        else:
            renamed["forecast_rain_mm"] = 0.0

    if "absolute_error_mm" not in renamed.columns:
        if "abs_error" in renamed.columns:
            renamed = renamed.rename(columns={"abs_error": "absolute_error_mm"})
        else:
            renamed["absolute_error_mm"] = 0.0

    if "era5_reference_rain_mm" not in renamed.columns:
        if "era5_actual_rain" in renamed.columns:
            renamed = renamed.rename(columns={"era5_actual_rain": "era5_reference_rain_mm"})
        else:
            renamed["era5_reference_rain_mm"] = 0.0

    if "is_bust" not in renamed.columns:
        renamed["is_bust"] = 0

    return renamed


def render_forecast_passport(engine: WeatherGuardEngine, lead_time: int, region: str, mode: str) -> None:
    """Render a risk table and high-risk alert card.

    mode controls whether the passport is built from live NWP forecast outputs or from
    historical ERA5 backtest records. In backtest mode, the visible fields include the
    observed ERA5 reference rain and absolute error for scientific verification.
    """
    st.subheader("📋 Forecast Passport & Alerts")

    if mode == "historical":
        historical = get_backtest_snapshot(lead_time, region)
        if historical.empty:
            st.warning("No ERA5 historical verification rows are available for this lead time and region.")
            return

        passport = _normalize_passport_columns(historical)
        passport["source"] = "ERA5 Backtest"
        passport["run"] = "Historical verification"
        passport["value"] = passport["forecast_rain_mm"].apply(lambda x: f"{x:.1f} mm")
        passport["reliability"] = passport["is_bust"].apply(lambda x: "Low" if x == 1 else "High")
        passport["bust_risk"] = passport["is_bust"].apply(lambda x: "High Risk" if x == 1 else "Low Risk")
        passport["expected_error_range"] = passport["absolute_error_mm"].apply(
            lambda x: f"{max(0.0, x - 5.0):.1f}–{x + 5.0:.1f} mm"
        )
        passport["diagnostic_factors"] = passport.apply(
            lambda row: (
                "High rainfall signal; lead-time decay; moisture-rich atmosphere"
                if row["is_bust"] == 1 else
                "Near-climatology forecast; low bust probability"
            ),
            axis=1,
        )
        passport["observed_era5_rain_mm"] = passport["era5_reference_rain_mm"].apply(lambda x: f"{x:.1f} mm")
        passport["observed_error_mm"] = passport["absolute_error_mm"].apply(lambda x: f"{x:.1f} mm")

        top_row = passport.iloc[0]
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #E0F2FE, #E2E8F0); border-left: 6px solid #1D4ED8; padding: 18px; border-radius: 12px; margin-bottom: 18px;">
                <h3 style="margin:0; color:#1E3A8A;">📊 Historical Verification Summary</h3>
                <p style="margin:8px 0 0 0; font-size: 1.0rem; color:#1F2937;">
                    ERA5 observed rainfall: <strong>{top_row['era5_reference_rain_mm']:.1f} mm</strong> |
                    Absolute error: <strong>{top_row['absolute_error_mm']:.1f} mm</strong> |
                    Bust flag: <strong>{'Yes' if top_row['is_bust'] else 'No'}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display = passport[
            [
                "source",
                "run",
                "lead_time",
                "value",
                "reliability",
                "bust_risk",
                "expected_error_range",
                "observed_era5_rain_mm",
                "observed_error_mm",
                "diagnostic_factors",
            ]
        ].rename(
            columns={
                "source": "Source",
                "run": "Run",
                "lead_time": "Lead Time",
                "value": "Value",
                "reliability": "Reliability",
                "bust_risk": "Bust Risk",
                "expected_error_range": "Expected Error Range",
                "observed_era5_rain_mm": "Observed ERA5 Rain",
                "observed_error_mm": "Observed Error (mm)",
                "diagnostic_factors": "Diagnostic Factors",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        return

    current_df = get_forecast_snapshot(lead_time, region)
    predictions = engine.predict_live(current_df)
    high_risk = predictions[predictions["bust_alert"] == True].sort_values(
        "bust_probability",
        ascending=False,
    ).head(10)

    if high_risk.empty:
        st.warning("No high-risk cells triggered for this lead time and region.")
        return

    top_row = high_risk.iloc[0]
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #fff3cd, #f8d7da); border-left: 6px solid #b22222; padding: 18px; border-radius: 12px; margin-bottom: 18px;">
            <h3 style="margin:0; color:#7a1c1c;">🚨 High-Risk Alert</h3>
            <p style="margin:8px 0 0 0; font-size: 1.05rem; color:#3d3d3d;">
                Grid {int(top_row['grid_id'])} in {top_row['region']} is forecast to become unreliable.<br>
                Bust probability: <strong>{top_row['bust_probability']:.2%}</strong> | Reliability: <strong>{top_row['reliability_score']:.1f}%</strong> |
                Expected error: <strong>{top_row['estimated_error_mm']:.1f} mm</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    passport = _normalize_passport_columns(high_risk)
    passport["source"] = "NWP Forecast"
    passport["run"] = "00Z / 12Z cycle"
    passport["value"] = passport["forecast_rain_mm"].apply(lambda x: f"{x:.1f} mm")
    passport["expected_error_range"] = passport["estimated_error_mm"].apply(
        lambda x: f"{max(0.0, x - 8.0):.1f}–{x + 8.0:.1f} mm"
    )
    passport["bust_risk"] = passport["bust_probability"].apply(lambda x: f"{x:.2%}")
    passport["reliability"] = passport["reliability_score"].apply(lambda x: f"{x:.1f}%")
    passport["diagnostic_factors"] = passport["why_unreliable"]

    display = passport[
        [
            "source",
            "run",
            "lead_time",
            "value",
            "reliability",
            "bust_risk",
            "expected_error_range",
            "diagnostic_factors",
        ]
    ].rename(
        columns={
            "source": "Source",
            "run": "Run",
            "lead_time": "Lead Time",
            "value": "Value",
            "reliability": "Reliability",
            "bust_risk": "Bust Risk",
            "expected_error_range": "Expected Error Range",
            "diagnostic_factors": "Diagnostic Factors",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    """Main application entrypoint."""
    engine = load_engine()

    with st.sidebar:
        st.header("Forecast Controls")
        mode = st.radio(
            "Operational Mode",
            ["Live Reliability Mode", "Historical Verification Mode (ERA5 Backtest)"],
            index=0,
            horizontal=False,
        )
        lead_time = st.slider("Lead Time (Days)", min_value=1, max_value=10, value=3)
        region = st.selectbox("Region Filter", ["ALL", "North", "South", "East", "West", "Central"], index=0)

        st.markdown("---")
        st.caption("Scientific Rule")
        st.write(
            "ERA5 is only used in historical dataset generation to create ground truth labels. "
            "Live inference uses forecast features only."
        )

    metric_cols = st.columns(4)
    grid_resolution = "0.25° × 0.25°"
    with metric_cols[0]:
        st.markdown('<div class="kpi-card"><div class="label">Analysis Grid Resolution</div><div class="value">0.25° × 0.25°</div></div>', unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown(f'<div class="kpi-card"><div class="label">Active Lead Time</div><div class="value">Day {lead_time}</div></div>', unsafe_allow_html=True)
    with metric_cols[2]:
        if mode == "Live Reliability Mode":
            current_df = get_forecast_snapshot(lead_time, region)
            predictions = engine.predict_live(current_df)
            high_risk_cells = int(predictions["bust_alert"].sum())
        else:
            historical_df = get_backtest_snapshot(lead_time, region)
            high_risk_cells = int(historical_df["is_bust"].sum()) if not historical_df.empty else 0
        st.markdown(f'<div class="kpi-card"><div class="label">High Bust Risk Grids</div><div class="value">{high_risk_cells}</div></div>', unsafe_allow_html=True)
    with metric_cols[3]:
        if mode == "Live Reliability Mode":
            current_df = get_forecast_snapshot(lead_time, region)
            predictions = engine.predict_live(current_df)
            avg_rel = predictions["reliability_score"].mean()
        else:
            historical_df = get_backtest_snapshot(lead_time, region)
            avg_rel = 100.0 * (1.0 - historical_df["is_bust"].mean()) if not historical_df.empty else 100.0
        st.markdown(f'<div class="kpi-card"><div class="label">Mean Regional Reliability</div><div class="value">{avg_rel:.1f}%</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🗺️ National Reliability Map",
            "📈 Lead-Time Decay Curve",
            "📋 Forecast Passport & Alerts",
            "🔎 Model Evidence",
        ]
    )

    with tab1:
        render_map_tab(engine, lead_time, region)
    with tab2:
        render_decay_tab()
    with tab3:
        render_forecast_passport(engine, lead_time, region, mode="historical" if "Historical" in mode else "live")
    with tab4:
        render_evidence_tab(engine)


if __name__ == "__main__":
    main()
