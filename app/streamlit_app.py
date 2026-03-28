"""
app/streamlit_app.py
=====================
Streamlit dashboard for the BI-101 Geospatial Profitability Predictor.

Refactored to match the "Tomato Intelligence BI-101" dark-mode UI design.
"""

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
GRID_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "grid.geojson",
)

st.set_page_config(
    page_title="X-Pand Profitability Predictor",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS for Tomato Intel UI
# ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* Base Dark Theme */
[data-testid="stAppViewContainer"] {
    background-color: #0b1120;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Sidebar Titles */
.sidebar-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: -10px;
}
.sidebar-subtitle {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: -10px;
}
.sidebar-link {
    font-size: 0.9rem;
    color: #38bdf8;
    margin-bottom: 2rem;
}
.sidebar-divider {
    border-bottom: 1px solid #1e293b;
    margin: 1.5rem 0;
}

/* System Status */
.status-dot {
    height: 8px;
    width: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}
.status-text {
    font-size: 0.85rem;
    color: #94a3b8;
}

/* Top Header */
.top-header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}
.top-header-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Metric Cards */
.metric-card {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 15px;
    display: flex;
    flex-direction: column;
}
.metric-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
}
.val-white { color: #ffffff; }
.val-green { color: #10b981; }
.val-yellow { color: #f59e0b; }
.val-red { color: #ef4444; }

/* Legend */
.legend-container {
    display: flex;
    gap: 20px;
    align-items: center;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 15px;
    margin-bottom: 5px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.legend-dot-g { height: 10px; width: 10px; border-radius: 50%; background-color: #10b981; }
.legend-dot-y { height: 10px; width: 10px; border-radius: 50%; background-color: #f59e0b; }
.legend-dot-r { height: 10px; width: 10px; border-radius: 50%; background-color: #ef4444; }
.legend-dot-b { height: 12px; width: 12px; border-radius: 50%; background-color: #06b6d4; border: 2px solid #0b1120; }

/* Override Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 0;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Button override */
div.stButton > button:first-child {
    background-color: transparent;
    color: #38bdf8;
    border: 1px solid #38bdf8;
    border-radius: 4px;
    width: 100%;
    font-weight: 600;
    letter-spacing: 0.05em;
}
div.stButton > button:hover {
    background-color: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid #38bdf8;
}

/* Result Card */
.result-card {
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    background-color: rgba(16, 185, 129, 0.05);
}
.result-title { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
.result-val { font-size: 1.5rem; font-weight: 700; color: #10b981; margin-bottom: 5px; }
.result-sub { font-size: 0.85rem; color: #94a3b8; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_grid():
    """Load the grid GeoDataFrame from disk and prep coordinates for PolygonLayer."""
    try:
        if not os.path.exists(GRID_PATH):
            st.error(f"Grid file not found at {GRID_PATH}. Run notebooks first.")
            st.stop()
            
        gdf = gpd.read_file(GRID_PATH)
        if "centroid_lat" not in gdf.columns:
            gdf["centroid_lat"] = gdf.geometry.centroid.y
        if "centroid_lon" not in gdf.columns:
            gdf["centroid_lon"] = gdf.geometry.centroid.x
            
        # Extract polygon coordinates for PyDeck PolygonLayer
        # format: [[[lon, lat], [lon, lat], ...]]
        gdf["coordinates"] = gdf.geometry.apply(
            lambda geom: [list(c) for c in geom.exterior.coords]
        )
        return gdf
    except Exception as exc:
        st.error(f"Failed to load grid: {exc}")
        st.stop()


grid_gdf = load_grid()

def fetch_predictions(locations_dict):
    """Call POST /batch with every grid-cell centroid."""
    try:
        resp = requests.post(
            f"{API_BASE}/batch",
            json={"locations": locations_dict},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["predictions"]
    except requests.exceptions.ConnectionError:
        st.sidebar.error(f"API unreachable ({API_BASE}). Start uvicorn.")
        st.stop()
    except Exception as exc:
        st.sidebar.error(f"Prediction error: {exc}")
        st.stop()

# Build location payload once (vectorised — no iterrows)
if "locs_payload" not in st.session_state:
    _payload_df = grid_gdf[["grid_id", "centroid_lat", "centroid_lon"]].copy()
    _payload_df = _payload_df.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
    _payload_df["grid_id"] = _payload_df["grid_id"].astype(str)
    _payload_df["lat"] = _payload_df["lat"].astype(float)
    _payload_df["lon"] = _payload_df["lon"].astype(float)
    st.session_state["locs_payload"] = _payload_df.to_dict(orient="records")

if "predictions" not in st.session_state:
    with st.spinner("Connecting to Intelligence Core..."):
        st.session_state["predictions"] = fetch_predictions(st.session_state["locs_payload"])

predictions = st.session_state["predictions"]


# ──────────────────────────────────────────────────────────────────────
# Sidebar Layout
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">TOMATO INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">BI-101</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-link">Geospatial Profitability</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title" style="margin-bottom: 15px;">OPTIMIZER CONTROLS</div>', unsafe_allow_html=True)
    
    min_prob = st.slider("MIN PROBABILITY", 0.0, 1.0, 0.50, 0.05)
    max_hubs = st.slider("MAX HUBS", 1, 50, 10)
    min_sep = st.slider("MIN SEPARATION KM", 0.5, 10.0, 2.0, 0.5)
    
    st.write("") # spacer
    run_optimizer = st.button("RUN OPTIMIZER")
    
    if run_optimizer:
        with st.spinner("Optimizing..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/optimize",
                    json={"max_hubs": max_hubs, "min_separation_km": min_sep, "min_prob_threshold": min_prob},
                    timeout=120,
                )
                resp.raise_for_status()
                res = resp.json()
                st.session_state["selected_hubs"] = res["selected_hubs"]
                st.session_state["opt_score"] = res["total_score"]
                st.session_state["sep_met"] = res["separation_constraint_met"]
            except Exception as exc:
                st.error(f"Optimizer failed: {exc}")

    # Display Last Result
    if "selected_hubs" in st.session_state:
        n_hubs = len(st.session_state["selected_hubs"])
        sc = st.session_state["opt_score"]
        sep_ok = "OK" if st.session_state.get("sep_met") else "VIOLATED"
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">LAST RESULT</div>
            <div class="result-val">{n_hubs} hubs</div>
            <div class="result-sub">Score: {sc:.4f}<br/>Separation: {sep_ok}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="sidebar-divider" style="margin-top: 40px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title" style="margin-bottom: 15px;">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div><span class="status-dot"></span><span class="status-text">API Connected</span></div>', unsafe_allow_html=True)
    st.markdown('<div><span class="status-dot"></span><span class="status-text">Models Loaded</span></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Main Layout - Header & Tabs
# ──────────────────────────────────────────────────────────────────────

# Convert predictions to DF and join with coordinates geometry
pred_df = pd.DataFrame(predictions)
merged_df = pred_df.merge(grid_gdf[["grid_id", "coordinates"]], on="grid_id", how="inner")

h_cnt = int((merged_df["p_profit"] > 0.7).sum())
m_cnt = int(((merged_df["p_profit"] >= 0.4) & (merged_df["p_profit"] <= 0.7)).sum())
s_cnt = int((merged_df["p_profit"] < 0.4).sum())
t_cnt = len(merged_df)

tab1, tab2 = st.tabs(["PREDICTION MAP", "CELL DETAIL"])

with tab1:
    # ── Header row
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown('<div class="top-header-title">Profitability<br/>Prediction Map</div>', unsafe_allow_html=True)
    with col_t2:
        st.markdown('<div class="top-header-subtitle" style="text-align: right; margin-top: 25px;">DELHI NCR — 500M GRID</div>', unsafe_allow_html=True)
        
    # ── Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-title">TOTAL CELLS</div><div class="metric-value val-white">{t_cnt:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-title">HIGH POTENTIAL</div><div class="metric-value val-green">{h_cnt:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-title">MONITOR</div><div class="metric-value val-yellow">{m_cnt:,}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-title">SKIP</div><div class="metric-value val-red">{s_cnt:,}</div></div>', unsafe_allow_html=True)

    # ── Legend row
    st.markdown("""
    <div class="legend-container">
        <div style="margin-right: 10px; letter-spacing: 0.05em; transform: scale(0.9);">LEGEND</div>
        <div class="legend-item"><span class="legend-dot-g"></span> High > 0.7</div>
        <div class="legend-item"><span class="legend-dot-y"></span> Monitor 0.4–0.7</div>
        <div class="legend-item"><span class="legend-dot-r"></span> Skip < 0.4</div>
        <div class="legend-item"><span class="legend-dot-b"></span> Selected hub</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Map
    # Colors matching the target UI exactly
    def _color_for_p(p):
        if p > 0.7:
            return [16, 185, 129]  # Green
        if p >= 0.4:
            return [245, 158, 11]  # Yellow/Orange
        return [239, 68, 68]       # Red
        
    merged_df["fill_color"] = merged_df["p_profit"].apply(_color_for_p)
    
    grid_layer = pdk.Layer(
        "PolygonLayer",
        data=merged_df,
        get_polygon="coordinates",
        get_fill_color="fill_color",
        get_line_color=[30, 41, 59, 200], # dark slate border for grid cells
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        extruded=False,
        pickable=True,
        opacity=0.85,
    )
    
    layers = [grid_layer]
    
    # Formatting for tooltips
    merged_df["p_profit_fmt"] = merged_df["p_profit"].apply(lambda p: f"{p*100:.1f}%")

    # Hub overlay
    if "selected_hubs" in st.session_state:
        hub_ids = set(st.session_state["selected_hubs"])
        hub_df = merged_df[merged_df["grid_id"].isin(hub_ids)].copy()
        
        hub_layer = pdk.Layer(
            "ScatterplotLayer",
            data=hub_df,
            get_position=["lon", "lat"],
            get_radius=550,
            get_fill_color=[6, 182, 212], # Cyan
            pickable=False,
            opacity=1.0,
        )
        layers.append(hub_layer)

    tooltip = {
        "html": "<b>Grid Cell:</b> {grid_id}<br/><b>P(profit):</b> {p_profit_fmt}<br/><b>Status:</b> {recommendation}",
        "style": {
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "border": "1px solid #1e293b",
            "fontSize": "13px",
            "fontFamily": "sans-serif",
            "padding": "10px",
            "borderRadius": "6px"
        },
    }

    view = pdk.ViewState(
        longitude=77.10,
        latitude=28.65,
        zoom=10.2,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip=tooltip,
        map_style=pdk.map_styles.DARK,
        map_provider="carto"
    )

    # Force a specific height for the map so it fits nicely
    st.pydeck_chart(deck, use_container_width=True)


with tab2:
    st.markdown('<div class="top-header-title" style="margin-bottom: 20px;">Grid Cell Detail View</div>', unsafe_allow_html=True)

    grid_ids = sorted([p["grid_id"] for p in predictions])
    selected_id = st.selectbox("Select Grid Cell Identifier", options=grid_ids)

    cell_pred = next((p for p in predictions if p["grid_id"] == selected_id), None)

    if cell_pred:
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        
        p_pct = f"{cell_pred['p_profit'] * 100:.1f}%"
        mcol1.markdown(f'<div class="metric-card"><div class="metric-title">P(PROFIT)</div><div class="metric-value val-white">{p_pct}</div></div>', unsafe_allow_html=True)

        ci_lo = cell_pred.get("ci_lower")
        lo_str = f"{ci_lo:.3f}" if ci_lo is not None else "N/A"
        mcol2.markdown(f'<div class="metric-card"><div class="metric-title">CI LOWER</div><div class="metric-value val-white">{lo_str}</div></div>', unsafe_allow_html=True)

        ci_hi = cell_pred.get("ci_upper")
        hi_str = f"{ci_hi:.3f}" if ci_hi is not None else "N/A"
        mcol3.markdown(f'<div class="metric-card"><div class="metric-title">CI UPPER</div><div class="metric-value val-white">{hi_str}</div></div>', unsafe_allow_html=True)

        rec = cell_pred["recommendation"].upper()
        # Color status
        rc_color = "val-green" if rec=="OPEN" else "val-yellow" if rec=="MONITOR" else "val-red"
        mcol4.markdown(f'<div class="metric-card"><div class="metric-title">STATUS</div><div class="metric-value {rc_color}">{rec}</div></div>', unsafe_allow_html=True)

        cld = "YES" if cell_pred["is_cold_start"] else "NO"
        cld_color = "val-red" if cld=="YES" else "val-white"
        mcol5.markdown(f'<div class="metric-card"><div class="metric-title">COLD START</div><div class="metric-value {cld_color}">{cld}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title" style="margin-top: 30px; margin-bottom: 15px;">TOP FEATURE IMPACTS (SHAP)</div>', unsafe_allow_html=True)

        drivers = cell_pred.get("shap_drivers", [])
        if not drivers and not cell_pred["is_cold_start"]:
            with st.spinner("Analyzing ML drivers..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/predict",
                        json={"locations": [{"lat": cell_pred["lat"], "lon": cell_pred["lon"], "grid_id": selected_id}]},
                        timeout=5,
                    )
                    if r.status_code == 200:
                        drivers = r.json()["predictions"][0].get("shap_drivers", [])
                except Exception:
                    pass

        if drivers:
            driver_df = pd.DataFrame(drivers[:5]).set_index("feature")
            st.bar_chart(driver_df["impact"], color="#38bdf8")
        else:
            st.info("No SHAP drivers available (cold-start cells use Thompson Sampling).")
