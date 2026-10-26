import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date

st.set_page_config(
    page_title="AirScope | NO₂ Downscaling",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Theme --------------------
st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e8eef9; }
    [data-testid="stSidebar"] { background: #111b2d; }
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #123b4a 60%, #174b45);
        border: 1px solid #28546a; margin-bottom: 1rem;
    }
    .eyebrow { color: #78e0c1; text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; font-weight: 700; }
    .hero h1 { margin: .25rem 0 .35rem 0; color: #f4f8ff; font-size: 2.2rem; }
    .hero p { color: #c6d5e6; margin: 0; }
    .metric-card {
        padding: 1rem 1.1rem; border: 1px solid #243650;
        border-radius: 14px; background: #111c2d;
    }
    .small-note { color: #9eb0c8; font-size: .85rem; }
    div[data-testid="stMetric"] { background: #111c2d; border: 1px solid #243650; padding: 14px; border-radius: 12px; }
    h2, h3 { color: #eaf2ff; }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.markdown("## 🌍 AirScope")
st.sidebar.caption("Satellite air-quality intelligence")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Explore satellite data", "Downscaling lab", "Model evaluation"],
)
st.sidebar.divider()
st.sidebar.markdown("**Study area**")
region_name = st.sidebar.selectbox("Region", ["Chennai, Tamil Nadu", "Custom study area (coming soon)"])
start_date = st.sidebar.date_input("Start date", value=date(2025, 1, 1))
end_date = st.sidebar.date_input("End date", value=date(2025, 1, 31))
st.sidebar.caption("Satellite source: Sentinel-5P OFFL NO₂")
st.sidebar.caption("Project: SIH 2024 problem statement SIH1734")

# -------------------- Demo data --------------------
# A visible demo is used until the real Earth Engine raster is exported and loaded.
@st.cache_data
def make_demo(seed=7, n=72):
    rng = np.random.default_rng(seed)
    lat = np.linspace(12.7, 13.3, n)
    lon = np.linspace(80.0, 80.5, n)
    xx, yy = np.meshgrid(lon, lat)
    # Smooth illustrative spatial field, not measured observations.
    field = (
        0.000025
        + 0.000075 * np.exp(-(((xx - 80.22) / 0.10) ** 2 + ((yy - 13.05) / 0.09) ** 2))
        + 0.000035 * np.exp(-(((xx - 80.36) / 0.07) ** 2 + ((yy - 12.91) / 0.08) ** 2))
        + rng.normal(0, 0.000004, size=xx.shape)
    )
    field = np.clip(field, 0, None)
    return pd.DataFrame({"longitude": xx.ravel(), "latitude": yy.ravel(), "no2": field.ravel()})

import rasterio
from rasterio.transform import xy

RASTER_PATH = "chennai_no2_jan2025.tif"

try:
    with rasterio.open(RASTER_PATH) as src:
        band = src.read(1, masked=True).astype("float64")
        rows, cols = np.indices(band.shape)
        lon, lat = xy(src.transform, rows, cols)

        df = pd.DataFrame({
            "longitude": np.asarray(lon).ravel(),
            "latitude": np.asarray(lat).ravel(),
            "no2": np.asarray(band.filled(np.nan)).ravel()
        }).dropna()

    st.sidebar.success("Loaded Earth Engine GeoTIFF")

except Exception as e:
    df = make_demo()
    st.sidebar.warning(f"Using demo data: {e}")

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI • Earth observation • Environmental analytics</div>
  <h1>AirScope</h1>
  <p>Satellite-Based Air Quality Map Downscaling with AI/ML</p>
</div>
""", unsafe_allow_html=True)

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

if page == "Overview":
    st.markdown("### Project overview")
    st.write("Explore tropospheric nitrogen dioxide (NO₂) and compare satellite observations with model-based downscaling outputs.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satellite scenes retrieved", "440", help="Count returned by the Earth Engine query for the selected study window.")
    c2.metric("Target pollutant", "NO₂")
    c3.metric("Study area", "Chennai")
    c4.metric("Models", "FNO + AM-FNO")
    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("### Illustrative NO₂ spatial pattern")
        fig = px.density_map(
            df, lat="latitude", lon="longitude", z="no2",
            radius=18, zoom=8, center={"lat": 13.0, "lon": 80.25},
            map_style="carto-darkmatter",
            color_continuous_scale=["#14213d", "#00b4d8", "#90e0ef", "#ffd166", "#ef476f"],
            labels={"no2": "NO₂ (mol/m²)"}
        )
        fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#0b1220", font_color="#e8eef9")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Workflow")
        st.markdown("""
        1. **Acquire** Sentinel-5P NO₂ observations
        2. **Prepare** a consistent spatial grid
        3. **Downscale** with Fourier Neural Operator models
        4. **Evaluate** against suitable reference data
        """)
        st.warning("The map currently uses illustrative demo values, not the real satellite raster. Do not report these values as measured NO₂.")
        st.markdown("### Current progress")
        st.progress(0.55, text="UI scaffold + Earth Engine connection")

elif page == "Explore satellite data":
    st.markdown("### Explore satellite data")
    st.write(f"**Region:** {region_name}  ·  **Date window:** {start_date} to {end_date}")
    st.info("Earth Engine is connected in your Colab notebook. To show measured satellite values here, export a small raster/CSV from that notebook and upload it below.")
    uploaded = st.file_uploader("Upload exported CSV with longitude, latitude, and no2 columns", type=["csv"])
    plot_df = df
    measured = False
    if uploaded is not None:
        candidate = pd.read_csv(uploaded)
        required = {"longitude", "latitude", "no2"}
        if required.issubset(candidate.columns):
            plot_df = candidate.dropna(subset=["longitude", "latitude", "no2"])
            measured = True
            st.success(f"Loaded {len(plot_df):,} spatial samples from your CSV.")
        else:
            st.error("CSV needs these columns: longitude, latitude, no2.")
    st.caption("Showing uploaded Earth Engine export." if measured else "Showing illustrative demo values.")
    fig = px.scatter_map(
        plot_df, lat="latitude", lon="longitude", color="no2",
        color_continuous_scale=["#14213d", "#00b4d8", "#90e0ef", "#ffd166", "#ef476f"],
        zoom=8, center={"lat": 13.0, "lon": 80.25}, map_style="carto-darkmatter",
        labels={"no2": "NO₂ (mol/m²)"}, opacity=0.8
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#0b1220", font_color="#e8eef9")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(plot_df.describe().T, use_container_width=True)

elif page == "Downscaling lab":
    st.markdown("### Downscaling lab")
    st.write("Compare a coarse input grid with model outputs once the trained checkpoints and compatible inputs are connected.")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### Coarse input")
        st.image("https://placehold.co/480x300/14213d/e8eef9?text=Coarse+satellite+grid", use_container_width=True)
    with b:
        st.markdown("#### FNO output")
        st.image("https://placehold.co/480x300/164e63/e8eef9?text=FNO+prediction", use_container_width=True)
    with c:
        st.markdown("#### AM-FNO output")
        st.image("https://placehold.co/480x300/3f315b/e8eef9?text=AM-FNO+prediction", use_container_width=True)
    st.warning("Model outputs are not connected in this UI scaffold yet. The placeholders are not predictions. We will connect your trained model and exported input next.")
    st.markdown("#### Run settings")
    resolution = st.select_slider("Target grid size", options=["16 × 16", "32 × 32", "64 × 64"], value="32 × 32")
    model_choice = st.multiselect("Models to compare", ["FNO", "AM-FNO"], default=["FNO", "AM-FNO"])
    if st.button("Prepare comparison", type="primary"):
        st.info(f"Comparison settings saved for {resolution}: {', '.join(model_choice) if model_choice else 'no models selected'}. Connect model weights to generate actual outputs.")

elif page == "Model evaluation":
    st.markdown("### Model evaluation")
    st.write("The metrics below are from the earlier synthetic-data experiment and are shown as a development reference only.")
    metrics = pd.DataFrame({
        "Method": ["Bilinear baseline", "FNO", "AM-FNO"],
        "Validation MSE (synthetic)": [0.000421, 0.000576, 0.001039],
    })
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    fig = px.bar(metrics, x="Method", y="Validation MSE (synthetic)", color="Method",
                 color_discrete_sequence=["#00b4d8", "#06d6a0", "#ffd166"])
    fig.update_layout(height=360, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", font_color="#e8eef9", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.warning("These are synthetic experiment metrics, not validation results on real satellite or ground-monitor measurements. Lower MSE is better; the bilinear baseline performed best in that experiment.")

st.divider()
st.caption("AirScope • Academic prototype • Sentinel-5P NO₂ • Results must be validated before real-world interpretation.")
