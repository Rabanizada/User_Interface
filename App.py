import streamlit as st

# App Title
st.title('Evaluation of Global Hydrological Model')

# --- Sidebar ---
st.sidebar.title("Hydrology Research Group")  # Title at the very top
st.sidebar.header("Home")

# Top-level menu selection
main_selection = st.sidebar.radio(
    "Select Section:",
    ("Variable", "Case Study", "Contact Us")
)

# Show sub-options in sidebar, only for selected main section
if main_selection == "Variable":
    sub_selection = st.sidebar.radio(
        "Variable:",
        ("Streamflow", "Terrestrial Water Storage Anomaly", "Snow Cover Fraction", "Reservoir Storage")
    )
    # Main Page Content
    st.header(sub_selection)
    st.write(f"Displaying information about **{sub_selection}** here...")

# streamlit_app.py
import os
import json
import warnings
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ==============================
# PAGE CHROME
# ==============================
st.set_page_config(page_title="Evaluation GHM", page_icon=":globe_with_meridians:", layout="wide")
st.title(" Evaluation of Global Hydrological model")
st.subheader("Terrestrial Water Storage Anomaly")

# ==============================
# CONFIG — adjust these paths
# ==============================
DEFAULT_CSV = r"C:\Users\rbaniza\Documents\TWSA\User_interface\twsa_hydrosat_results06062025.csv"
SHP_PATH = Path(r"C:\Users\rbaniza\Documents\TWSA\Basins_classification\WaterGAP22e_TWSA__classified.shp")

# ==============================
# SINGLE FILE UPLOADER (UNIQUE KEY)
# ==============================
fl = st.file_uploader("📂 Upload TWSA metrics (CSV/XLSX)", type=["csv", "txt", "xlsx", "xls"], key="twsa_metrics_uploader_final")

# ---------- Load table ----------
def read_table(file_like_or_path):
    if hasattr(file_like_or_path, "name"):  # uploaded file
        name = file_like_or_path.name.lower()
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_like_or_path)
        try:
            return pd.read_csv(file_like_or_path, sep=None, engine="python")
        except UnicodeDecodeError:
            file_like_or_path.seek(0)
            return pd.read_csv(file_like_or_path, sep=None, engine="python", encoding="ISO-8859-1")
    else:  # local path
        path = str(file_like_or_path)
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=None, engine="python", encoding="ISO-8859-1")

if fl is not None:
    df = read_table(fl)
    st.caption(f"Loaded: {fl.name}")
else:
    if os.path.exists(DEFAULT_CSV):
        df = read_table(DEFAULT_CSV)
        st.caption(f"Loaded default: {DEFAULT_CSV}")
    else:
        st.error("No file uploaded and default CSV not found.")
        st.stop()

st.markdown("### Performance of TWSA: WaterGAP22e against Observational data (Hydrosat)")

# ==============================
# HARMONIZE COLUMNS & GEOMETRY
# ==============================
# Accept either KGEα or KGEa
vr_col = "KGEα" if "KGEα" in df.columns else ("KGEa" if "KGEa" in df.columns else None)
needed = ["ID_1", "NSE", "KGE", "KGEr"]
if (vr_col is None) or any(c not in df.columns for c in needed):
    st.error("Uploaded table must include: ID_1, NSE, KGE, KGEr and KGEα/KGEa.")
    st.stop()

# Build GeoDataFrame: use WKT 'geometry' if present; else merge with shapefile by ID_1
if "geometry" in df.columns:
    # Parse WKT geometry robustly
    try:
        geom = gpd.GeoSeries.from_wkt(df["geometry"])
    except Exception:
        geom = gpd.GeoSeries.from_wkt(df["geometry"].astype(str).str.strip('"').str.strip("'"))
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
else:
    if not SHP_PATH.exists():
        st.error("CSV has no 'geometry' column and shapefile is missing. Provide one of them.")
        st.stop()
    basins = gpd.read_file(SHP_PATH)
    if basins.crs is None or str(basins.crs).lower() not in ("epsg:4326", "wgs84"):
        basins = basins.to_crs(epsg=4326)
    left = df[["ID_1", "NSE", "KGE", "KGEr", vr_col]].copy()
    left["ID_1"] = left["ID_1"].astype(str)
    basins["ID_1"] = basins["ID_1"].astype(str)
    gdf = basins.merge(left, on="ID_1", how="left")

# Ensure ID_1 is string in properties for matching
gdf["ID_1"] = gdf["ID_1"].astype(str)

# --- Complete drop-in code (equal-height colorbar segments, black labels, boundary ticks) ---

# ==============================
# HELPERS
# ==============================
def pick_name_column(df_: pd.DataFrame) -> str:
    for c in ["rivr_nm", "sttn_nm", "basin_name", "name"]:
        if c in df_.columns:
            return c
    return "ID_1"

def first_existing_column(df_: pd.DataFrame, candidates, default=None) -> str:
    for c in candidates:
        if c in df_.columns:
            return c
    return default if default is not None else candidates[0]

def equal_step_colorscale(colors):
    """Equal-height step colorscale: divides the bar into len(colors) equal boxes."""
    n = len(colors)
    cs = []
    for i, c in enumerate(colors):
        lo = i / n
        hi = (i + 1) / n
        cs.append([lo, c])
        cs.append([hi, c])
    return cs

def metric_config(metric, values):
    """Per-metric thresholds, colors, ticks, and filters (filters unchanged)."""
    col_bad   = "#D55E00"
    col_mid   = "#E69F00"
    col_good  = "#56B4E9"
    col_great = "#009E73"
    col_yel   = "#F0E442"
    col_blue  = "#0072B2"

    v = pd.to_numeric(values, errors="coerce")
    finite_vals = v[np.isfinite(v)]
    vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0

    if metric == "NSE":
        zmin, zmax = min(vmin, -1.0), 1.0
        thresholds = [zmin, 0.0, 0.5, 0.7, 1.0]
        colors     = [col_bad, col_mid, col_good, col_great]
        ticks      = [0.0, 0.5, 0.7]
        ticktext   = ["0.00", "0.50", "0.70"]
        filter_opts = ["All ranges", "≥ 0.70", "0.50 – 0.70", "0.00 – 0.50", "< 0.00"]

    elif metric == "KGE":
        zmin, zmax = min(vmin, -1.0), 1.0
        thresholds = [zmin, -0.41, 0.5, 0.7, 1.0]
        colors     = [col_bad, col_mid, col_good, col_great]
        ticks      = [-0.41, 0.5, 0.7]
        ticktext   = ["-0.41", "0.50", "0.70"]
        filter_opts = ["All ranges", "≥ 0.70", "0.50 – 0.70", "-0.41 – 0.50", "< -0.41"]

    elif metric == "KGEr":
        zmin, zmax = -1.0, 1.0
        thresholds = [zmin, 0.0, 0.5, 0.8, 1.0]
        colors     = [col_bad, col_mid, col_good, col_great]
        ticks      = [0.0, 0.5, 0.8]
        ticktext   = ["0.00", "0.50", "0.80"]
        filter_opts = ["All ranges", "≥ 0.80", "0.50 – 0.80", "0.00 – 0.50", "< 0.00"]

    else:  # KGEα
        zmin = min(vmin, 0.0)
        zmax = max(1.8, vmax, 1.5)  # keep at least standard upper bound
        thresholds = [zmin, 0.5, 0.9, 1.1, 1.5, zmax]  # 6 bounds -> 5 colors
        colors     = [col_bad, col_mid, col_yel, col_good, col_blue]
        ticks      = [0.5, 0.9, 1.1, 1.5]
        ticktext   = ["0.50", "0.90", "1.10", "1.50"]
        filter_opts = ["All ranges", "0.90 – 1.10 (ideal)", "1.10 – 1.50", "> 1.50", "0.50 – 0.90", "< 0.50"]

    return dict(
        zmin=zmin, zmax=zmax,
        thresholds=thresholds, colors=colors,
        ticks=ticks, ticktext=ticktext,
        filter_options=filter_opts
    )

def parse_filter(metric, label):
    if label == "All ranges": return -np.inf, np.inf
    if metric == "NSE":
        return {"≥ 0.70": (0.70, 1.0), "0.50 – 0.70": (0.50, 0.70),
                "0.00 – 0.50": (0.00, 0.50), "< 0.00": (-np.inf, 0.00)}[label]
    if metric == "KGE":
        return {"≥ 0.70": (0.70, 1.0), "0.50 – 0.70": (0.50, 0.70),
                "-0.41 – 0.50": (-0.41, 0.50), "< -0.41": (-np.inf, -0.41)}[label]
    if metric == "KGEr":
        return {"≥ 0.80": (0.80, 1.0), "0.50 – 0.80": (0.50, 0.80),
                "0.00 – 0.50": (0.00, 0.50), "< 0.00": (-np.inf, 0.00)}[label]
    if metric == "KGEα":
        return {"0.90 – 1.10 (ideal)": (0.90, 1.10), "1.10 – 1.50": (1.10, 1.50),
                "> 1.50": (1.50, np.inf), "0.50 – 0.90": (0.50, 0.90), "< 0.50": (-np.inf, 0.50)}[label]
    return -np.inf, np.inf

# ==============================
# UI (left) – metric + filter; right stays blank
# ==============================
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    c1, c2 = st.columns([1, 1])
    metric = c1.selectbox("Metric", ["NSE", "KGE", "KGEr", "KGEα"], index=0, key="metric_select_plotly")

    # Column mapping
    nse_col  = first_existing_column(gdf, ["NSE", "nse"], default="NSE")
    kge_col  = first_existing_column(gdf, ["KGE", "kge"], default="KGE")
    kger_col = first_existing_column(gdf, ["KGEr", "r", "KGE_r"], default="KGEr")
    vr_col   = first_existing_column(gdf, ["KGEα", "KGEa", "alpha", "vr", "VR"], default="vr")

    metric_to_column = {"NSE": nse_col, "KGE": kge_col, "KGEr": kger_col, "KGEα": vr_col}
    mcol = metric_to_column[metric]
    vals = pd.to_numeric(gdf[mcol], errors="coerce").values

    cfg = metric_config(metric, vals)
    perf_filter = c2.selectbox("Performance filter", cfg["filter_options"], index=0, key="perf_filter_plotly")

    # Filter rows
    lo, hi = parse_filter(metric, perf_filter)
    mask = pd.to_numeric(gdf[mcol], errors="coerce").between(lo, hi, inclusive="both")
    gdf_f = gdf.loc[mask].copy()

    # GeoJSON + data for plotted features
    geojson = json.loads(gdf.to_json())
    locations = gdf_f["ID_1"].astype(str).tolist()
    z_true = pd.to_numeric(gdf_f[mcol], errors="coerce").values

    # Equal-height colorbar logic
    n_colors = len(cfg["colors"])
    edges = cfg["thresholds"][1:-1]                 # interior boundaries
    z_idx = np.digitize(z_true, edges, right=False) # 0..n_colors-1
    colorscale = equal_step_colorscale(cfg["colors"])

    # Dynamic extremes from *displayed* data
    finite = z_true[np.isfinite(z_true)]
    zmin_data = float(np.nanmin(finite)) if finite.size else None
    zmax_data = float(np.nanmax(finite)) if finite.size else None
    fmt2 = lambda x: (f"{x:.2f}" if x is not None else "")

    # Colorbar ticks at category boundaries (0..n_colors)
    if metric == "NSE":
        tickvals = list(range(1, n_colors)) + [n_colors]
        ticktext = cfg["ticktext"] + [fmt2(zmax_data)]
    elif metric == "KGE":
        tickvals = list(range(1, n_colors)) + [n_colors]
        ticktext = cfg["ticktext"] + [fmt2(zmax_data)]
    elif metric == "KGEr":
        tickvals = [0] + list(range(1, n_colors)) + [n_colors]
        ticktext = [fmt2(zmin_data)] + cfg["ticktext"] + [fmt2(zmax_data)]
    else:  # KGEα  <-- UPDATED: now shows dynamic TOP value too
        tickvals = [0] + list(range(1, n_colors)) + [n_colors]
        ticktext = [fmt2(zmin_data)] + cfg["ticktext"] + [fmt2(zmax_data)]

    name_col = pick_name_column(gdf_f)
    custom = np.column_stack([
        gdf_f["ID_1"].astype(str).fillna(""),
        gdf_f[name_col].astype(str).fillna(""),
        z_true,
    ])

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.ID_1",
            locations=locations,
            z=z_idx.astype(float),
            zmin=0,
            zmax=float(n_colors),
            colorscale=colorscale,
            marker_line_color="black",
            marker_line_width=0.4,
            colorbar=dict(
                title=metric,
                ticks="outside",
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                len=0.85,
                thickness=14,
                tickfont=dict(color="black"),
                titlefont=dict(color="black"),
            ),
            hovertemplate=(
                "<b>Basin ID</b>: %{customdata[0]}<br>"
                "<b>Basin name</b>: %{customdata[1]}<br>"
                f"<b>{metric}</b>: " + "%{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=custom,
        )
    )

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="black",
        showcountries=False,
        showframe=False,
        projection_type="robinson",
        lonaxis_range=[-220, 220],
        lataxis_range=[-100, 140],
        showocean=False,
        showland=False,
    )
    fig.update_layout(
        margin=dict(l=50, r=50, t=70, b=50),
        height=540,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )

    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("&nbsp;", unsafe_allow_html=True)


# ==============================
# RIGHT COLUMN — Time series (interactive)
# ==============================
with right_col:
    st.subheader("Time series of TWSA by basin")

    # --- Default file (falls back to this if user doesn't upload) ---
    DEFAULT_TS_CSV = r"C:/Users/rbaniza/Documents/TWSA/User_interface\Hydrosat_Model_twsa_all_basins.csv"

    # --- Dedicated uploader for time series (unique key) ---
    ts_file = st.file_uploader(
        "📎 Upload time‑series file (CSV/TXT/XLSX/XLS)",
        type=["csv", "txt", "xlsx", "xls"],
        key="twsa_timeseries_uploader",
        help="Upload a file containing columns: Date, Observed_HydroSat_TWSA_mm, Model_WaterGAP_TWSA_mm, ID_1, rivr_nm",
    )

    # --- Reader (reuses your read_table() if present; otherwise uses a local fallback) ---
    def _read_ts(file_or_path):
        if "read_table" in globals() and callable(globals()["read_table"]):
            return read_table(file_or_path)  # use your existing robust reader
        # Fallback reader
        if hasattr(file_or_path, "name"):  # uploaded file-like
            name = file_or_path.name.lower()
            if name.endswith((".xlsx", ".xls")):
                return pd.read_excel(file_or_path)
            try:
                return pd.read_csv(file_or_path, sep=None, engine="python")
            except UnicodeDecodeError:
                file_or_path.seek(0)
                return pd.read_csv(file_or_path, sep=None, engine="python", encoding="ISO-8859-1")
        else:  # local path string
            path = str(file_or_path)
            if path.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(path)
            try:
                return pd.read_csv(path, sep=None, engine="python")
            except UnicodeDecodeError:
                return pd.read_csv(path, sep=None, engine="python", encoding="ISO-8859-1")

    # Load time-series table
    if ts_file is not None:
        ts_df = _read_ts(ts_file)
        st.caption(f"Loaded time-series: {ts_file.name}")
    else:
        if os.path.exists(DEFAULT_TS_CSV):
            ts_df = _read_ts(DEFAULT_TS_CSV)
            st.caption(f"Loaded default time-series: {DEFAULT_TS_CSV}")
        else:
            st.error("No time-series file uploaded and the default file was not found.")
            st.stop()

    # --- Ensure required columns / harmonize types ---
    if "first_existing_column" not in globals():
        def first_existing_column(df_, candidates, default=None):
            for c in candidates:
                if c in df_.columns:
                    return c
            return default if default is not None else candidates[0]

    required = ["ID_1", "rivr_nm"]
    missing = [c for c in required if c not in ts_df.columns]
    if missing:
        st.error(f"Missing required column(s) in time-series file: {', '.join(missing)}")
        st.stop()

    date_col = "Date" if "Date" in ts_df.columns else ("time" if "time" in ts_df.columns else None)
    if date_col is None:
        st.error("Time-series file must include a 'Date' (or 'time') column.")
        st.stop()

    obs_col = first_existing_column(
        ts_df, ["Observed_HydroSat_TWSA_mm", "Observed_TWSA_mm", "HydroSat_TWSA_mm", "Observed", "obs"], default=None
    )
    mod_col = first_existing_column(
        ts_df, ["Model_WaterGAP_TWSA_mm", "Model_TWSA_mm", "WaterGAP_TWSA_mm", "Model", "sim"], default=None
    )
    if obs_col not in ts_df.columns or mod_col not in ts_df.columns:
        st.error(
            "Time-series file must include the TWSA columns "
            "'Observed_HydroSat_TWSA_mm' and 'Model_WaterGAP_TWSA_mm' (or recognized alternatives)."
        )
        st.stop()

    # Clean types
    ts_df = ts_df.copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df["ID_1"] = ts_df["ID_1"].astype(str).str.strip()
    ts_df["rivr_nm"] = ts_df["rivr_nm"].astype(str).str.strip()
    ts_df = ts_df.dropna(subset=[date_col])

    # --- Selection widgets (top-of-graph filter) ---
    csel1, csel2 = st.columns([1, 2])
    with csel1:
        select_mode = st.radio(
            "Select basin by",
            options=["ID_1", "rivr_nm"],
            index=0,
            horizontal=True,
            key="ts_select_mode",
        )
    with csel2:
        if select_mode == "ID_1":
            id_options = sorted(ts_df["ID_1"].dropna().astype(str).unique(), key=lambda x: (len(x), x))
            chosen_id = st.selectbox("Basin ID (ID_1)", id_options, key="ts_id_select")
            sel_df = ts_df.loc[ts_df["ID_1"] == chosen_id].sort_values(date_col)
            title = f"Basin ID {chosen_id}"
            export_name = f"TWSA_{chosen_id}"
        else:
            name_options = sorted(ts_df["rivr_nm"].dropna().astype(str).unique())
            chosen_name = st.selectbox("Basin name (rivr_nm)", name_options, key="ts_name_select")
            sel_df = ts_df.loc[ts_df["rivr_nm"] == chosen_name].sort_values(date_col)
            ids_here = ", ".join(sorted(sel_df["ID_1"].astype(str).unique()))
            title = f"{chosen_name} (ID_1: {ids_here})"
            export_name = f"TWSA_{chosen_name.replace(' ', '_').replace(',', '')}"

    if sel_df.empty:
        st.info("No data for the selected basin.")
        st.stop()

    # --- Interactive Plotly chart (no range slider, centered title, black fonts, legend moved down) ---
    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=sel_df[date_col],
            y=sel_df[obs_col],
            name="Observed (HydroSat)",
            mode="lines",
            line=dict(color="black", width=2.5),   # observed = black
            hovertemplate="%{x|%Y-%m-%d}<br>Observed: %{y:.2f} mm<extra></extra>",
            showlegend=True,
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=sel_df[date_col],
            y=sel_df[mod_col],
            name="Model (WaterGAP)",
            mode="lines",
            line=dict(color="blue", width=2.5),    # model = blue
            hovertemplate="%{x|%Y-%m-%d}<br>Model: %{y:.2f} mm<extra></extra>",
            showlegend=True,
        )
    )

    fig_ts.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color="black", size=16)),
        xaxis=dict(
            title=dict(text="Date", font=dict(color="black")),
            type="date",
            showgrid=True,
            showline=False,         
            zeroline=False,         
            rangeslider=dict(visible=False),  
        ),
        yaxis=dict(
            title=dict(text="TWSA (mm)", font=dict(color="black")),
            showgrid=True,
            showline=False,
            zeroline=False,
        ),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=70, b=50),
        height=540,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,         
            xanchor="left",
            x=0.0,
            font=dict(color="black"),
        ),
        hoverlabel=dict(font=dict(color="black")),
    )

    st.plotly_chart(
        fig_ts,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"filename": export_name},  # keep download
            # default modebar keeps zoom in/out/reset etc.
        },
    )

    # --- Download the filtered table as CSV ---
    dl_cols = [date_col, obs_col, mod_col, "ID_1", "rivr_nm"]
    csv_bytes = sel_df[dl_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download displayed data (CSV)",
        data=csv_bytes,
        file_name=f"{export_name}.csv",
        mime="text/csv",
        help="Download the filtered time-series for this basin",
        key="ts_download_btn",
    )


elif main_selection == "Case Study":
    sub_selection = st.sidebar.radio(
        "Case Study:",
        ("WaterGAP",)
    )
    st.header(sub_selection)
    st.write(f"Details for case study: **{sub_selection}**.")

elif main_selection == "Contact Us":
    st.header("Contact Us")
    st.write("📧 Email: hydrology@example.com")
    st.write("🏢 Address: Hydrology Research Group, Example University")

