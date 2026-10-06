# ============================================================
# IEEE C57.91 Dynamic Transformer Rating – Streamlit Dashboard
# ============================================================

import io
import os
import requests
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
# from geopy.geocoders import Nominatim
import datetime
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json

# Folder where this script lives — used to locate the bundled CSV
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# FUTURE CMIP6 DTR DATABASE
# ============================================================

FUTURE_DATA_DIR = os.path.join(
    SCRIPT_DIR,
    "future_dtr_data"
)

FUTURE_CASE_FILE = os.path.join(
    FUTURE_DATA_DIR,
    "ACCESS-CM2_US50_DTR_Case_Summary.csv"
)

FUTURE_SEASONAL_FILE = os.path.join(
    FUTURE_DATA_DIR,
    "ACCESS-CM2_US50_DTR_Seasonal.csv"
)

FUTURE_ANNUAL_FILE = os.path.join(
    FUTURE_DATA_DIR,
    "ACCESS-CM2_US50_DTR_Annual.csv"
)

FUTURE_HISTOGRAM_FILE = os.path.join(
    FUTURE_DATA_DIR,
    "ACCESS-CM2_US50_DTR_Histogram.csv"
)

FUTURE_MAPPING_FILE = os.path.join(
    FUTURE_DATA_DIR,
    "ACCESS-CM2_US50_Capital_Grid_Mapping.csv"
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Distribution Asset Rating Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Dynamic Asset Rating Dashboard ⚡")
st.caption("IEEE C57.91 /IEEE 738 thermal model · NASA POWER Temperature data")

# Independent scroll for sidebar and main content
st.markdown("""
<style>
    /* Full-height flex layout so both panels sit side by side */
    .stApp {
        display: flex;
        flex-direction: row;
        height: 100vh;
        overflow: hidden;
    }

    /* Sidebar outer: clip overflow so only inner div scrolls */
    section[data-testid="stSidebar"] {
        height: 100vh;
        overflow: hidden;
        flex-shrink: 0;
    }

    /* Sidebar inner div: single scrollbar, always visible, blue */
    section[data-testid="stSidebar"] > div {
        height: 100%;
        overflow-y: scroll;
        overflow-x: hidden;
        scrollbar-width: thin;
        scrollbar-color: #1a73e8 #dce8fc;
    }

    section[data-testid="stSidebar"] > div::-webkit-scrollbar {
        width: 6px;
        display: block;
    }
    section[data-testid="stSidebar"] > div::-webkit-scrollbar-track {
        background: #dce8fc;
        border-radius: 4px;
    }
    section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
        background-color: #1a73e8;
        border-radius: 4px;
    }
    section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb:hover {
        background-color: #1558b0;
    }

    /* Main panel: fills remaining width, always-visible scrollbar on right */
    section.main {
        height: 100vh;
        overflow-y: scroll;
        overflow-x: hidden;
        flex: 1;
        scrollbar-width: thin;
        scrollbar-color: #1a73e8 #dce8fc;
    }

    /* Webkit browsers (Chrome, Edge) – always show main scrollbar */
    section.main::-webkit-scrollbar {
        width: 6px;
        display: block;
    }
    section.main::-webkit-scrollbar-track {
        background: #dce8fc;
        border-radius: 4px;
    }
    section.main::-webkit-scrollbar-thumb {
        background-color: #1a73e8;
        border-radius: 4px;
    }
    section.main::-webkit-scrollbar-thumb:hover {
        background-color: #1558b0;
    }

    /* Remove any max-height constraints on inner block */
    section.main > div.block-container {
        max-height: none !important;
        padding-bottom: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GEOAPIFY GEOCODING FUNCTIONS
# Cloud-safe replacement for public Nominatim
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city):
    """
    Convert a city/address into coordinates using Geoapify.
    Results are cached for 24 hours.
    """

    api_key = st.secrets["GEOAPIFY_API_KEY"]

    url = "https://api.geoapify.com/v1/geocode/search"

    params = {
        "text": city,
        "format": "json",
        "limit": 1,
        "filter": "countrycode:us",
        "apiKey": api_key
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        return None

    result = results[0]

    return {
        "lat": result.get("lat"),
        "lon": result.get("lon"),
        "address": result.get(
            "formatted",
            city
        ),
        "state": result.get("state"),
        "state_code": result.get("state_code"),
        "city": result.get(
            "city",
            result.get("county", city)
        )
    }


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode(lat, lon):
    """
    Convert coordinates into address/state using Geoapify.
    Results are cached for 24 hours.
    """

    api_key = st.secrets["GEOAPIFY_API_KEY"]

    url = "https://api.geoapify.com/v1/geocode/reverse"

    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "limit": 1,
        "apiKey": api_key
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        return None

    result = results[0]

    return {
        "address": result.get(
            "formatted",
            f"{lat}, {lon}"
        ),
        "state": result.get("state"),
        "state_code": result.get("state_code"),
        "city": result.get(
            "city",
            result.get("county")
        )
    }

# ─────────────────────────────────────────────────────────────
# SIDEBAR – INPUTS
# ─────────────────────────────────────────────────────────────
fixed_wind_speed = 0.6

with st.sidebar:
    st.header("🔧 Configuration")

    st.subheader("⚙️ Calculation Mode")

    calc_mode = st.radio(
        "What do you want to calculate?",
        [
            "Dynamic Transformer Rating",
            "Dynamic Conductor Rating",
            "Future U.S. DTR Projection"
        ],
        key="calculation_mode"
    )

    # --- Location ---
    # ─────────────────────────────────────────────────────────────
    # 📍 PROFESSIONAL LOCATION MODULE
    # ─────────────────────────────────────────────────────────────
    st.subheader("📍 Location Selection")

    DEFAULT_LAT = 42.8864
    DEFAULT_LON = -78.8784

    # Initialize session state
    if "lat" not in st.session_state:
        st.session_state.lat = DEFAULT_LAT
    if "lon" not in st.session_state:
        st.session_state.lon = DEFAULT_LON
    if "city" not in st.session_state:
        st.session_state.city = ""
    if "address" not in st.session_state:
        st.session_state.address = "Buffalo, NY (Default)"
    if "state" not in st.session_state:
        st.session_state.state = "New York"

    # ── City Input ──
    city_input = st.text_input(
        "Search City",
        value=st.session_state.city,
        placeholder="e.g. Chicago, IL"
    )

    colA, colB = st.columns([2, 1])

    with colA:
        if st.button("🌍 Geocode City", use_container_width=True):

            if city_input.strip():

                try:
                    with st.spinner("Geocoding city..."):

                        geo = geocode_city(
                            city_input.strip()
                        )

                    if geo:

                        st.session_state.lat = round(
                            geo["lat"], 4
                        )

                        st.session_state.lon = round(
                            geo["lon"], 4
                        )

                        st.session_state.city = (
                                geo.get("city")
                                or city_input.strip()
                        )

                        st.session_state.address = geo["address"]

                        if geo.get("state"):
                            st.session_state.state = geo["state"]

                        st.success(
                            f"Location updated: "
                            f"{st.session_state.city}, "
                            f"{st.session_state.state}"
                        )

                        st.rerun()

                    else:
                        st.warning("City not found.")

                except Exception as e:

                    st.error(
                        "Unable to retrieve the location right now. "
                        "Please try again or enter coordinates manually."
                    )

            else:
                st.warning("Enter a city name.")

    with colB:
        if st.button("🔄 Reset"):
            st.session_state.lat = DEFAULT_LAT
            st.session_state.lon = DEFAULT_LON
            st.session_state.city = ""
            st.session_state.address = "Buffalo, NY (Default)"
            st.session_state.state = "New York"
            st.rerun()

    # ── Interactive Map ──
    st.markdown("### 🗺️ Location Map")

    map_df = pd.DataFrame({
        "lat": [st.session_state.lat],
        "lon": [st.session_state.lon]
    })
    # # ── Interactive Clickable Map ──
    st.markdown("### 🗺️ Click Map to Select Location")

    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=6
    )

    # Marker at current location
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        tooltip="Current Location"
    ).add_to(m)

    map_data = st_folium(
        m,
        height=350,
        width=None,
    )
    # ── Detect map click ──
    if map_data and map_data.get("last_clicked"):

        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        st.session_state.lat = round(
            clicked_lat,
            4
        )

        st.session_state.lon = round(
            clicked_lon,
            4
        )

        try:

            location_info = reverse_geocode(
                st.session_state.lat,
                st.session_state.lon
            )

            if location_info:

                st.session_state.address = (
                    location_info["address"]
                )

                if location_info.get("state"):
                    st.session_state.state = (
                        location_info["state"]
                    )

                if location_info.get("city"):
                    st.session_state.city = (
                        location_info["city"]
                    )

            else:

                st.session_state.address = (
                    f"Coordinates: "
                    f"{st.session_state.lat}, "
                    f"{st.session_state.lon}"
                )

        except Exception as e:

            st.session_state.address = (
                f"Coordinates: "
                f"{st.session_state.lat}, "
                f"{st.session_state.lon}"
            )

        st.success(
            "📍 Location selected from map"
        )

        st.rerun()


    st.markdown("### ✏️ Manual Coordinate Override")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            format="%.4f",
            key="lat"
        )
    with col2:
        st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            format="%.4f",
            key="lon"
        )
    if st.button(
            "📍 Apply Coordinates",
            use_container_width=True
    ):

        try:

            location_info = reverse_geocode(
                round(st.session_state.lat, 4),
                round(st.session_state.lon, 4)
            )

            if location_info:

                st.session_state.address = (
                    location_info["address"]
                )

                if location_info.get("state"):
                    st.session_state.state = (
                        location_info["state"]
                    )

                if location_info.get("city"):
                    st.session_state.city = (
                        location_info["city"]
                    )

                st.success(
                    "Coordinates updated."
                )

                st.rerun()

            else:

                st.warning(
                    "No U.S. location was found "
                    "for these coordinates."
                )

        except Exception as e:

            st.error(
                f"Coordinate lookup failed: {e}"
            )


    st.success(f"📌 Current Location: {st.session_state.address}")
    st.caption(f"Coordinates: {st.session_state.lat}, {st.session_state.lon}")
    st.caption(f"Detected State: {st.session_state.state}")

    # Final variables used by NASA fetch
    lat = st.session_state.lat
    lon = st.session_state.lon

    # ============================================================
    # DATE RANGE
    # Only for historical/current DTR and DCR
    # ============================================================

    if calc_mode != "Future U.S. DTR Projection":
        st.subheader("📅 Date Range")

        col3, col4 = st.columns(2)

        with col3:
            start_date = st.date_input(
                "Start date",
                value=datetime.date(2025, 1, 1)
            )

        with col4:
            end_date = st.date_input(
                "End date",
                value=datetime.date(2025, 12, 31)
            )

    if calc_mode == "Future U.S. DTR Projection":
        st.subheader("🌎 Future Climate Projection")

        future_scenario = st.selectbox(
            "Climate Scenario",
            [
                "SSP1-2.6",
                "SSP2-4.5",
                "SSP5-8.5"
            ],
            index=1,
            key="future_scenario"
        )

        future_period = st.selectbox(
            "Future Period",
            [
                "2026-2050",
                "2051-2075"
            ],
            key="future_period"
        )

        st.caption(
            "Climate model: ACCESS-CM2 | "
            "NASA NEX-GDDP-CMIP6 | "
            "Ensemble: r1i1p1f1"
        )

        # ========================================================
        # FUTURE STUDY TRANSFORMER INFORMATION
        # ========================================================

        st.subheader("🔌 Future Study Transformer")

        st.info(
            """
            National future projections use the validated
            **22 MVA research transformer**.

            • Nameplate: 22 MVA  
            • IEEE C57.91-2023 thermal model  
            • Hot-spot limit: 110°C  
            • Ambient safety margin: +5°C
            """
        )

    # --- Hot-spot limit ---
    hs_limit = 110
    if calc_mode == "Dynamic Transformer Rating":
        st.subheader("🌡️ Hot-Spot Limit")
        hs_limit = st.slider("Max hot-spot temperature (°C)", 90, 140, 110, step=1)
    # ============================================================
    # AMBIENT TEMPERATURE MARGIN
    # ============================================================
    if calc_mode != "Future U.S. DTR Projection":

        st.subheader("🌡️ Ambient Temperature Margin")

        temp_margin = st.slider(
            "Ambient Temperature Safety Margin (°C)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Adds a safety margin to ambient temperature before DTR calculation"
        )

    else:

        temp_margin = 5

    # --- Transformer CSV ---
    # ─────────────────────────────────────────────────────────────
    # 🔌 TRANSFORMER INPUT OPTIONS
    # ─────────────────────────────────────────────────────────────

    transformer_params = None
    conductor_params = None

    if calc_mode == "Dynamic Transformer Rating":
        # --- Transformer CSV ---
        # ─────────────────────────────────────────────────────────────
        # 🔌 TRANSFORMER INPUT OPTIONS
        # ─────────────────────────────────────────────────────────────
        st.subheader("🔌 Transformer Nameplate")

        input_mode = st.radio(
            "Select Transformer Input Method",
            [
                "Use Default File",
                "Upload CSV",
                "Manual Entry"
            ],
            key="transformer_input_mode"
        )

        BUNDLED_CSV = os.path.join(SCRIPT_DIR, "transformer_thermal_nameplate.csv")

        # ─────────────────────────────────
        # OPTION 1 — DEFAULT FILE
        # ─────────────────────────────────
        if input_mode == "Use Default File":

            if os.path.exists(BUNDLED_CSV):
                preview = pd.read_csv(BUNDLED_CSV).iloc[0]
                st.success("Using bundled transformer file")
                st.caption(
                    f"MVA: {preview['MVA_rated']} | "
                    f"Cooling: {preview['cooling_system']} | "
                    f"Winding: {preview['winding_material']}"
                )
                transformer_params = preview.to_dict()
            else:
                st.error("Default CSV not found.")

        # ─────────────────────────────────
        # OPTION 2 — UPLOAD CSV
        # ─────────────────────────────────
        elif input_mode == "Upload CSV":

            uploaded_file = st.file_uploader(
                "Upload Transformer CSV",
                type="csv"
            )

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                transformer_params = df.iloc[0].to_dict()
                st.success("Uploaded transformer loaded.")
            else:
                st.warning("Upload a CSV file.")

        # ─────────────────────────────────
        # OPTION 3 — MANUAL ENTRY
        # ─────────────────────────────────
        elif input_mode == "Manual Entry":

            st.markdown("### ⚙️ Core Ratings")

            MVA_rated = st.number_input("Rated MVA", value=100.0)
            MVA_loss = st.number_input("MVA at which losses are defined", value=100.0)

            cooling_system = st.selectbox(
                "Cooling System",
                ["ONAN", "ONAF", "OFAF", "ODAF"]
            )

            winding_material = st.selectbox(
                "Winding Material",
                ["Copper", "Aluminum"]
            )

            st.markdown("### 🌡 Thermal Parameters")

            T_tor = st.number_input("Top-Oil Rise at Rated Load (°C)", value=55.0)
            T_hsr = st.number_input("Hot-Spot Rise at Rated Load (°C)", value=80.0)

            T_k = st.number_input("Reference Temperature T_k (°C)", value=75.0)
            T_wr = st.number_input("Winding Reference Temp (°C)", value=75.0)
            T_loss = st.number_input("Loss Reference Temp (°C)", value=75.0)

            st.markdown("### 🔥 Loss Components (kW)")

            P_wr = st.number_input("Rated Winding Loss (kW)", value=200.0)
            P_er = st.number_input("Eddy Loss (kW)", value=50.0)
            P_sr = st.number_input("Stray Loss (kW)", value=30.0)
            P_cr = st.number_input("Core Loss (kW)", value=100.0)

            transformer_params = {
                "MVA_rated": MVA_rated,
                "MVA_loss": MVA_loss,
                "cooling_system": cooling_system,
                "winding_material": winding_material,
                "T_tor": T_tor,
                "T_hsr": T_hsr,
                "T_k": T_k,
                "T_wr": T_wr,
                "T_loss": T_loss,
                "P_wr": P_wr,
                "P_er": P_er,
                "P_sr": P_sr,
                "P_cr": P_cr,
            }

            st.success("Manual transformer parameters ready.")

    # --- Conductor Input Options ---
    if calc_mode == "Dynamic Conductor Rating":
        st.subheader("🔌 Conductor Parameters")
        st.subheader("🌬️ Wind Assumption")

        fixed_wind_speed = st.number_input(
            "Constant Wind Speed (m/s)",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.2,
            help="This fixed wind speed will be used for Dynamic Conductor Rating instead of NASA wind data."
        )

        # ─────────────────────────────────
        # CONDUCTOR PRESET LIBRARY
        # ─────────────────────────────────
        conductor_library = {
            "1 AWG": {
                "static_amp_summer": 210.0,
                "static_amp_winter": 245.0,

                # Geometry / thermal
                "D_inch": 0.398,                # conductor diameter in inches
                "Ts": 75.0,                   # conductor surface temperature (degC)
                "alpha": 0.5,                  # solar absorptivity
                "epsilon": 0.5,                # emissivity

                # Environment
                "Qse": 95.2,                   # total solar and sky radiated heat intensity (W/m^2)
                "theta_deg": 70.0,            # effective angle of incidence of sun rays
                "Hc_ft": 130.0,               # altitude above sea level (ft)

                # Electrical
                "Rac_75_ohm_per_ft": 0.54 / 1000,            #AC Resistance in ohm/ft
                "K_angle": 1.0,                               #wind angle
            },

            "2 AWG": {
                "static_amp_summer": 185.0,
                "static_amp_winter": 220.0,

                # Geometry / thermal
                "D_inch": 0.355,
                "Ts": 75.0,
                "alpha": 0.5,
                "epsilon": 0.5,

                # Environment
                "Qse": 95.2,
                "theta_deg": 70.0,
                "Hc_ft": 130.0,

                # Electrical
                "Rac_75_ohm_per_ft": 0.648 / 1000,
                "K_angle": 1.0,
            },
            "3 AWG": {
                "static_amp_summer": 310.0,
                "static_amp_winter": 360.0,

                # Geometry / thermal
                "D_inch": 0.502,
                "Ts": 75.0,
                "alpha": 0.5,
                "epsilon": 0.5,

                # Environment
                "Qse": 95.2,
                "theta_deg": 70.0,
                "Hc_ft": 130.0,

                # Electrical
                "Rac_75_ohm_per_ft": 0.272 / 1000,
                "K_angle": 1.0,
            },
            "4 AWG": {
                "static_amp_summer": 140.0,
                "static_amp_winter": 170.0,

                # Geometry / thermal
                "D_inch": 0.316,
                "Ts": 75.0,
                "alpha": 0.5,
                "epsilon": 0.5,

                # Environment
                "Qse": 95.2,
                "theta_deg": 70.0,
                "Hc_ft": 130.0,

                # Electrical
                "Rac_75_ohm_per_ft": 0.818 / 1000,
                "K_angle": 1.0,
            },
            "336.4": {
                "static_amp_summer": 529.0,
                "static_amp_winter": 615.0,

                # Geometry / thermal
                "D_inch": 0.720,
                "Ts": 75.0,
                "alpha": 0.5,
                "epsilon": 0.5,

                # Environment
                "Qse": 95.2,
                "theta_deg": 70.0,
                "Hc_ft": 130.0,

                # Electrical
                "Rac_75_ohm_per_ft": 0.062 / 1000,
                "K_angle": 1.0,
            },
            "477": {
                "static_amp_summer": 659.0,
                "static_amp_winter": 810.0,

                # Geometry / thermal
                "D_inch": 0.858,
                "Ts": 75.0,
                "alpha": 0.5,
                "epsilon": 0.5,

                # Environment
                "Qse": 95.2,
                "theta_deg": 70.0,
                "Hc_ft": 130.0,

                # Electrical
                "Rac_75_ohm_per_ft": 0.044 / 1000,
                "K_angle": 1.0,
            },
            "795": {
                "static_amp_summer": 907.0,
                "static_amp_winter": 1039.0,

                # Geometry / thermal
                "D_inch": 1.108,
                "Ts": 75.0,
                "alpha": 0.5,
                "epsilon": 0.5,

                # Environment
                "Qse": 95.2,
                "theta_deg": 70.0,
                "Hc_ft": 130.0,

                # Electrical
                "Rac_75_ohm_per_ft": 0.026 / 1000,
                "K_angle": 1.0,
            }
        }

        conductor_type = st.selectbox(
            "Select Conductor Type",
            ["1 AWG", "2 AWG", "3 AWG", "4 AWG", "336.4", "477", "795"],
        )

        selected_conductor_defaults = conductor_library[conductor_type]

        conductor_input_mode = st.radio(
            "Select Conductor Input Method",
            [
                "Use Default Values",
                "Manual Entry"
            ]
        )
        # ─────────────────────────────────
        # OPTION 1 — DEFAULT VALUES
        # ─────────────────────────────────
        if conductor_input_mode == "Use Default Values":
            conductor_params = selected_conductor_defaults.copy()
            conductor_params["conductor_type"] = conductor_type

            st.success(f"Using default parameters for {conductor_type}")
            st.caption(
                f"D = {conductor_params['D_inch']} in | "
                f"Ts = {conductor_params['Ts']} °C | "
                f"Static Ampacity Summer= {conductor_params['static_amp_summer']} A | "
                f"Static Ampacity Winter= {conductor_params['static_amp_winter']} A"
            )

        # ─────────────────────────────────
        # OPTION 2 — MANUAL ENTRY
        # ─────────────────────────────────
        elif conductor_input_mode == "Manual Entry":
            st.markdown("### ⚙️ Conductor Thermal / Physical Parameters")

            static_amp_summer = st.number_input(
                "Summer Static Ampacity (A) [May–Oct]",
                value=selected_conductor_defaults["static_amp_summer"]
            )

            static_amp_winter = st.number_input(
                "Winter Static Ampacity (A) [Nov–Apr]",
                value=selected_conductor_defaults["static_amp_winter"]
            )

            D_inch = st.number_input(
                "Conductor Diameter, D (in)",
                value=selected_conductor_defaults["D_inch"]
            )
            Ts = st.number_input(
                "Conductor Surface Temperature Limit, Ts (°C)",
                value=selected_conductor_defaults["Ts"]
            )
            alpha = st.number_input(
                "Solar Absorptivity, α",
                value=selected_conductor_defaults["alpha"]
            )
            epsilon = st.number_input(
                "Emissivity, ε",
                value=selected_conductor_defaults["epsilon"]
            )

            st.markdown("### ☀️ Environmental Parameters")

            Qse = st.number_input(
                "Total Solar & Sky Heat Intensity, Qse (W/m²)",
                value=selected_conductor_defaults["Qse"]
            )
            theta_deg = st.number_input(
                "Solar Angle, θ (deg)",
                value=selected_conductor_defaults["theta_deg"]
            )
            Hc_ft = st.number_input(
                "Altitude, Hc (ft)",
                value=selected_conductor_defaults["Hc_ft"]
            )

            st.markdown("### ⚡ Electrical Parameters")

            Rac_75_ohm_per_ft = st.number_input(
                "AC Resistance at 75°C, Rac (Ω/ft)",
                value=selected_conductor_defaults["Rac_75_ohm_per_ft"],
                format="%.8f"
            )

            K_angle = st.number_input(
                "Wind Angle Factor, K_angle",
                value=selected_conductor_defaults["K_angle"],
                format="%.3f"
            )

            conductor_params = {
                "conductor_type": conductor_type,
                "static_amp_summer": static_amp_summer,
                "static_amp_winter": static_amp_winter,
                "D_inch": D_inch,
                "Ts": Ts,
                "alpha": alpha,
                "epsilon": epsilon,
                "Qse": Qse,
                "theta_deg": theta_deg,
                "Hc_ft": Hc_ft,
                "Rac_75_ohm_per_ft": Rac_75_ohm_per_ft,
                "K_angle": K_angle,
            }

            st.success("Manual conductor parameters ready.")
    # --- Run button ---
    st.markdown("---")
    if calc_mode == "Future U.S. DTR Projection":

        run_button_label = "🔮 SHOW FUTURE DTR"

    else:

        run_button_label = "🚀 SIMULATE"

    run_btn = st.button(
        run_button_label,
        use_container_width=True,
        type="primary"
    )


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_future_dtr_database():

    required_files = {
        "case": FUTURE_CASE_FILE,
        "seasonal": FUTURE_SEASONAL_FILE,
        "annual": FUTURE_ANNUAL_FILE,
        "histogram": FUTURE_HISTOGRAM_FILE,
        "mapping": FUTURE_MAPPING_FILE,
    }

    missing = [
        path
        for path in required_files.values()
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing future DTR database file(s):\n"
            + "\n".join(missing)
        )

    case_df = pd.read_csv(
        FUTURE_CASE_FILE
    )

    seasonal_df = pd.read_csv(
        FUTURE_SEASONAL_FILE
    )

    annual_df = pd.read_csv(
        FUTURE_ANNUAL_FILE
    )

    histogram_df = pd.read_csv(
        FUTURE_HISTOGRAM_FILE
    )

    mapping_df = pd.read_csv(
        FUTURE_MAPPING_FILE
    )

    return (
        case_df,
        seasonal_df,
        annual_df,
        histogram_df,
        mapping_df
    )

def air_density_kg_m3(Tfilm_c, altitude_m):
    """
    Approximate air density as a function of film temperature and altitude.
    """
    p = 101325.0 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    Tfilm_k = Tfilm_c + 273.15
    R_air = 287.05
    return p / (R_air * Tfilm_k)


def air_dynamic_viscosity_kg_ms(Tfilm_c):
    """
    Dynamic viscosity via Sutherland-type approximation.
    """
    Tfilm_k = Tfilm_c + 273.15
    return 1.458e-6 * (Tfilm_k ** 1.5) / (Tfilm_k + 383.4)


def air_thermal_conductivity_W_mK(Tfilm_c):
    """
    Approximate thermal conductivity of air.
    """
    return 0.02424 + 7.477e-5 * Tfilm_c - 4.407e-9 * (Tfilm_c ** 2)

@st.cache_data(ttl=86400)
def load_us_states_geojson():

    url = (
        "https://raw.githubusercontent.com/"
        "PublicaMundi/MappingAPI/master/data/"
        "geojson/us-states.json"
    )

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    return response.json()

def extract_state(address):
    if not address:
        return None

    parts = [p.strip() for p in address.split(",")]

    # Try full state name first
    for part in parts:
        if part in US_STATE_ABBR.values():
            return part  # already full name

    # Fallback: try abbreviation
    for part in parts:
        if part.upper() in US_STATE_ABBR:
            return US_STATE_ABBR[part.upper()]

    return None

US_STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming"
}

def plot_us_highlight_map(selected_state_full):

    geojson = load_us_states_geojson()

    states = [feature["properties"]["name"] for feature in geojson["features"]]

    df_map = pd.DataFrame({
        "state": states,
        "value": [1 if s == selected_state_full else 0 for s in states]
    })

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="state",
        featureidkey="properties.name",
        color="value",
        color_continuous_scale=["lightgrey", "red"],
        scope="usa"
    )

    fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(
        title="US Map – Selected Location Highlight",
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    # 🔵 Add marker for exact location
    fig.add_scattergeo(
        lon=[st.session_state.lon],
        lat=[st.session_state.lat],
        mode="markers",
        marker=dict(size=8, color="blue"),
        name="Selected Location"
    )

    return fig

@st.cache_data(show_spinner=False)
def fetch_nasa_power(lat, lon, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly T2M from NASA POWER API and return clean DataFrame."""
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=T2M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=CSV"
    )

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    text = resp.text

    # Skip NASA metadata header lines until the YEAR column header row
    lines = text.splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("YEAR"):
            data_start = i
            break

    csv_text = "\n".join(lines[data_start:])
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = df.columns.str.strip()
    return df

def load_transformer_params(uploaded_file) -> dict:
    """Load transformer params from uploaded file or the bundled CSV in the app folder."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        bundled = os.path.join(SCRIPT_DIR, "transformer_thermal_nameplate.csv")
        df = pd.read_csv(bundled)
    return df.iloc[0].to_dict()

def ieee_hotspot(L, T_a, p):
    """IEEE C57.91 hot-spot temperature (°C). Kept for single-point use."""
    x = (p['MVA_rated'] / p['MVA_loss']) ** 2
    t = (p['T_k'] + p['T_wr']) / (p['T_k'] + p['T_loss'])
    P_w = x * p['P_wr'] * t
    P_e = x * p['P_er'] / t
    P_s = x * p['P_sr'] / t
    P_c = p['P_cr']
    R = (P_w + P_e + P_s) / P_c
    n = m = 0.8                                 #oil and winding exponents
    dT_to = p['T_tor'] * ((1 + L**2 * R) / (1 + R)) ** n           #Top oil Rise
    dT_hs = (p['T_hsr'] - p['T_tor']) * L ** (2 * m)              #Hot spot rise
    return T_a + dT_to + dT_hs

def compute_dtr_vectorized(T_amb_array, params, T_hs_limit=110):
    L_candidates = np.linspace(0.1, 3.0, 400)

    # ── precompute transformer constants once ──
    x   = (params['MVA_rated'] / params['MVA_loss']) ** 2
    t   = (params['T_k'] + params['T_wr']) / (params['T_k'] + params['T_loss'])
    P_w = x * params['P_wr'] * t
    P_e = x * params['P_er'] / t
    P_s = x * params['P_sr'] / t
    R   = (P_w + P_e + P_s) / params['P_cr']
    n = m = 0.8

    # ── hot-spot rise curve for all L candidates (shape: 400,) ──
    hs_rise = (
        params['T_tor'] * ((1 + L_candidates**2 * R) / (1 + R)) ** n
        + (params['T_hsr'] - params['T_tor']) * L_candidates ** (2 * m)
    )   # monotonically increasing with L → searchsorted is valid

    # ── for each ambient temp find first L where hs_rise >= limit - T_a ──
    thresholds = T_hs_limit - np.asarray(T_amb_array)   # shape (N,)
    indices    = np.searchsorted(hs_rise, thresholds, side='left')
    indices    = np.clip(indices, 0, len(L_candidates) - 1)

    return L_candidates[indices]

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def compute_seasonal_stats(results: pd.DataFrame, data_col: str, label: str) -> pd.DataFrame:
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    percentiles = [10, 30, 50, 60, 90]
    rows = []

    for season in seasons:
        vals = results[results['Season'] == season][data_col].dropna()

        if len(vals) == 0:
            rows.append([season] + [np.nan] * (len(percentiles) + 4))
            continue

        perc_values = np.percentile(vals, percentiles)
        rows.append(
            [season] + list(perc_values) +
            [np.mean(vals), np.min(vals), np.max(vals), np.std(vals)]
        )

    columns = (
        ['Season'] +
        [f'{p}th % ({label})' for p in percentiles] +
        [f'Mean ({label})', f'Min ({label})', f'Max ({label})', f'Std Dev ({label})']
    )

    return pd.DataFrame(rows, columns=columns)

def compute_conductor_dcr(T_amb_array, wind_array, params):

    # -----------------------------
    # Extract user parameters
    # -----------------------------
    static_amp_summer = params["static_amp_summer"]
    static_amp_winter = params["static_amp_winter"]

    D_inch = params["D_inch"]
    D = D_inch * 0.0254                # meters

    Ts = params["Ts"]                  # conductor surface temp (°C)
    alpha = params["alpha"]
    epsilon = params["epsilon"]

    Qse = params["Qse"]                # W/m²
    theta_deg = params["theta_deg"]

    Hc_ft = params["Hc_ft"]
    Hc = Hc_ft * 0.3048                # meters

    Rac_75_ohm_per_ft = params["Rac_75_ohm_per_ft"]
    Rac = Rac_75_ohm_per_ft / 0.3048   # ohm/m

    K_angle = params["K_angle"]

    I_with_wind = []

    for Ta, V in zip(T_amb_array, wind_array):

        if np.isnan(Ta) or np.isnan(V):
            I_with_wind.append(np.nan)
            continue

        deltaT = Ts - Ta
        if deltaT <= 0:
            I_with_wind.append(np.nan)
            continue

        # Film temperature
        Tfilm = 0.5 * (Ts + Ta)

        # Air properties
        rho_f = air_density_kg_m3(Tfilm, Hc)              # kg/m³
        mu_f = air_dynamic_viscosity_kg_ms(Tfilm)         # kg/(m·s)
        k_f = air_thermal_conductivity_W_mK(Tfilm)        # W/(m·K)

        # Reynolds number
        NRe = D * rho_f * V / mu_f

        # Solar heating (W/m)
        qs = alpha * Qse * np.sin(np.radians(theta_deg)) * D

        # Radiative cooling (W/m)
        qr = 17.8 * D * epsilon * (
            ((Ts + 273.0) / 100.0) ** 4
            - ((Ta + 273.0) / 100.0) ** 4
        )

        # Natural convection (W/m)
        qcn = 3.645 * (rho_f ** 0.5) * (D ** 0.75) * (deltaT ** 1.25)

        # Forced convection with wind (W/m)
        qc1 = K_angle * (1.01 + 1.35 * (NRe ** 0.52)) * k_f * deltaT
        qc2 = K_angle * (0.754 * (NRe ** 0.60)) * k_f * deltaT

        # Maximum convection
        qc_with_wind = max(qcn, qc1, qc2)

        # Net Joule heating balance
        net_with_wind = qc_with_wind + qr - qs

        ampacity_with_wind = np.sqrt(net_with_wind / Rac) if net_with_wind > 0 else np.nan

        I_with_wind.append(ampacity_with_wind)

    return np.array(I_with_wind)

def monthly_boxplot_simple(results, static_amp_summer=None, static_amp_winter=None):
    fig = go.Figure()

    # Dynamic Rating
    fig.add_trace(go.Box(
        x=results['month'].astype(str),
        y=results['DCR'],
        name='Dynamic Conductor Rating',
        marker_color='blue',
        boxpoints=False,
        line=dict(width=2),
        width=0.45
    ))

    if static_amp_summer is not None:
        fig.add_hline(
            y=static_amp_summer,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="Static Ampacity (Summer)",
            annotation_position="top right"
        )
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name='Static Ampacity (Summer)',
            line=dict(color='red', dash='dash', width=2),
            showlegend=True
        ))

    if static_amp_winter is not None:
        fig.add_hline(
            y=static_amp_winter,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text="Static Ampacity (Winter)",
            annotation_position="bottom right"
        )
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name='Static Ampacity (Winter)',
            line=dict(color='green', dash='dot', width=2),
            showlegend=True
        ))

    fig.update_layout(
        title="Monthly Dynamic Conductor Rating",
        xaxis_title="Month",
        yaxis_title="Ampacity (Amps)",
        boxmode='group',
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=[str(i) for i in range(1, 13)]
    )

    return fig

def compute_monthly_stats(results: pd.DataFrame, data_col: str, label: str) -> pd.DataFrame:
    months = list(range(1, 13))
    percentiles = [10, 25, 50, 75, 90]
    rows = []

    for month in months:
        vals = results[results['month'] == month][data_col].dropna()

        if len(vals) == 0:
            rows.append([month] + [np.nan] * (len(percentiles) + 4))
            continue

        perc_values = np.percentile(vals, percentiles)
        rows.append(
            [month] + list(perc_values) +
            [np.mean(vals), np.min(vals), np.max(vals), np.std(vals)]
        )

    columns = (
        ['Month'] +
        [f'{p}th % ({label})' for p in percentiles] +
        [f'Mean ({label})', f'Min ({label})', f'Max ({label})', f'Std Dev ({label})']
    )

    return pd.DataFrame(rows, columns=columns)

def monthly_boxplot_with_stats(results, data_col, y_label, title, static_amp_summer=None, static_amp_winter=None):
    months = list(range(1, 13))
    percentiles = [10, 25, 50, 75, 90]
    percentile_colors = ['blue', 'green', 'purple', 'brown', 'orange']

    fig = go.Figure()

    for idx, month in enumerate(months):
        vals = results[results['month'] == month][data_col].dropna()
        if len(vals) == 0:
            continue

        perc_vals = np.percentile(vals, percentiles)
        mean_val = np.mean(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        std_val = np.std(vals)

        # Percentiles
        for p, v, c in zip(percentiles, perc_vals, percentile_colors):
            fig.add_trace(go.Scatter(
                x=[str(month)],
                y=[v],
                mode='markers',
                marker=dict(color=c, size=8),
                name=f'{p}th Percentile',
                legendgroup=f'{p}th Percentile',
                showlegend=(idx == 0)
            ))

        # Mean
        fig.add_trace(go.Scatter(
            x=[str(month)],
            y=[mean_val],
            mode='markers+text',
            marker=dict(color='black', size=10, symbol='diamond'),
            text=['Mean'],
            textposition='middle right',
            name='Mean',
            legendgroup='Mean',
            showlegend=(idx == 0)
        ))

        # Min
        fig.add_trace(go.Scatter(
            x=[str(month)],
            y=[min_val],
            mode='markers+text',
            marker=dict(color='grey', size=9, symbol='triangle-down'),
            text=['Min'],
            textposition='middle right',
            name='Min',
            legendgroup='Min',
            showlegend=(idx == 0)
        ))

        # Max
        fig.add_trace(go.Scatter(
            x=[str(month)],
            y=[max_val],
            mode='markers+text',
            marker=dict(color='magenta', size=9, symbol='triangle-up'),
            text=['Max'],
            textposition='middle right',
            name='Max',
            legendgroup='Max',
            showlegend=(idx == 0)
        ))

        # Std Dev (error bar)
        fig.add_trace(go.Scatter(
            x=[str(month)],
            y=[mean_val],
            error_y=dict(
                type='data',
                array=[std_val],
                visible=True,
                color='red',
                thickness=2,
                width=6
            ),
            mode='markers+text',
            marker=dict(color='red', size=1),
            text=['±σ'],
            textposition='bottom left',
            name='Std Dev',
            legendgroup='Std Dev',
            showlegend=(idx == 0)
        ))

    # Static ampacity line
    if static_amp_summer is not None:
        fig.add_hline(
            y=static_amp_summer,
            line_dash="dash",
            line_color="red",
            annotation_text="Summer Static"
        )

    if static_amp_winter is not None:
        fig.add_hline(
            y=static_amp_winter,
            line_dash="dot",
            line_color="green",
            annotation_text="Winter Static"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=y_label,
        hovermode="closest",
        legend=dict(
            x=1.02,
            y=1,
            bordercolor="black",
            borderwidth=1
        ),
        boxmode='group'
    )
    return fig

# ─────────────────────────────────────────────────────────────
# SEASON COLOURS
# ─────────────────────────────────────────────────────────────
SEASON_COLOR = {
    'Winter': '#3399ff',
    'Spring': '#33cc66',
    'Summer': '#ff4444',
    'Fall':   '#ff9900',
}
PERCENTILE_COLORS = ['blue', 'green', 'purple', 'orange']


# ============================================================
# FUTURE DTR PLOT STYLING
# ============================================================
# ============================================================
# DARK BLACK TEXT FOR PUBLICATION FIGURES
# ============================================================

def make_plot_text_black(fig):

    fig.update_layout(
        font=dict(color="#000000", size=14),

        title_font=dict(
            color="#000000",
            size=18
        ),

        legend_font=dict(
            color="#000000",
            size=13
        )
    )

    fig.update_xaxes(
        title_font=dict(color="#000000", size=16),
        tickfont=dict(color="#000000", size=14),
        linecolor="#000000",
        tickcolor="#000000"
    )

    fig.update_yaxes(
        title_font=dict(color="#000000", size=16),
        tickfont=dict(color="#000000", size=14),
        linecolor="#000000",
        tickcolor="#000000"
    )

    fig.update_annotations(
        font_color="#000000"
    )

    return fig

def style_future_plot(
    fig,
    height=470,
    hovermode=None
):

    layout_updates = dict(
        template="plotly_white",

        height=height,

        font=dict(
            size=13
        ),

        title=dict(
            x=0.01,
            xanchor="left",
            font=dict(
                size=17
            )
        ),

        margin=dict(
            l=60,
            r=35,
            t=90,
            b=60
        ),

        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    if hovermode is not None:
        layout_updates["hovermode"] = hovermode

    fig.update_layout(
        **layout_updates
    )

    fig.update_xaxes(
        title_font=dict(
            size=14
        ),
        tickfont=dict(
            size=12
        ),
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        gridcolor="rgba(0,0,0,0.07)",
        zeroline=False
    )

    fig.update_yaxes(
        title_font=dict(
            size=14
        ),
        tickfont=dict(
            size=12
        ),
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        gridcolor="rgba(0,0,0,0.07)",
        zeroline=False
    )

    return make_plot_text_black(fig)

# ============================================================
# FUTURE / PHASE 7 DASHBOARD SETTINGS
# ============================================================

FUTURE_SCENARIO_ORDER = [
    "ssp126",
    "ssp245",
    "ssp585"
]

FUTURE_SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5"
}

FUTURE_SCENARIO_COLORS = {
    "SSP1-2.6": "#1f77b4",
    "SSP2-4.5": "#ff7f0e",
    "SSP5-8.5": "#2ca02c"
}

FUTURE_PERIOD_ORDER = [
    "2026-2050",
    "2051-2075"
]

FUTURE_PERIOD_COLORS = {
    "2026-2050": "#1f77b4",
    "2051-2075": "#ff7f0e"
}

FUTURE_SEASON_ORDER = [
    "Winter",
    "Spring",
    "Summer",
    "Fall"
]

STATE_TO_ABBR = {
    full_name: abbreviation
    for abbreviation, full_name in US_STATE_ABBR.items()
}

# ============================================================
# TAB 8 — ADDITIONAL RESEARCH FIGURES (NO DTR RECALCULATION)
# ============================================================

FUTURE_RESEARCH_FIGURE_OPTIONS = {
    "Fig. 2 — Locations of 50 state capitals": "capital_locations",
    "Fig. 3 — Temperature spread across 50 capitals": "temperature_spread",
    "Fig. 4 — DTR spread across 50 capitals": "dtr_spread",
    "Fig. 9 — Summer DTR: short term vs. long term": "summer_comparison",
    "Fig. 10 — Temperature versus DTR relationship": "temperature_dtr_scatter",
    "Fig. 11 — State-capital DTR variation": "state_dtr_variation",
    "Fig. 12 — Change in DTR between future periods": "state_dtr_change",
}


def build_future_research_figure(
    figure_key,
    case_df,
    seasonal_df,
    mapping_df,
    selected_scenario,
    selected_period,
    selected_state,
):
    """Return (Plotly figure, caption, plotted-data DataFrame, export slug).

    Source tables are the already-computed national CMIP6 DTR CSVs. Each
    State value represents ONE state-capital location, not a statewide mean.
    National group means are simple, equally weighted means of 50 capitals.
    """
    scenario_order = ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    period_order = ["2026-2050", "2051-2075"]
    scenario_colors = FUTURE_SCENARIO_COLORS
    period_colors = FUTURE_PERIOD_COLORS

    def finish_xy(fig, title, height=560, hovermode="closest", left_margin=80):

        fig = style_future_plot(
            fig,
            height=height,
            hovermode=hovermode
        )

        fig.update_layout(

            # Figure title
            title=dict(
                text=title,
                x=0.01,
                xanchor="left",
                y=0.98,
                yanchor="top",
                font=dict(size=18)
            ),

            # Extra upper margin for title and legend
            margin=dict(
                l=left_margin,
                r=65,
                t=140,
                b=75
            ),

            # Legend positioned below the title
            legend=dict(
                orientation="h",
                x=0,
                xanchor="left",
                y=1.06,
                yanchor="top",
                title_text="",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0)"
            )
        )

        return fig

    if figure_key == "capital_locations":
        data = mapping_df.sort_values("State").drop_duplicates("State").copy()
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=data["Target_Lon"], lat=data["Target_Lat"],
            text=data["Capital"] + ", " + data["State"],
            customdata=data[["CMIP6_Lat", "CMIP6_Lon"]].to_numpy(),
            mode="markers", name="State-capital location",
            marker=dict(size=8, color="#2768A0", line=dict(width=0.8, color="white")),
            hovertemplate=(
                "%{text}<br>Capital: %{lat:.3f}°, %{lon:.3f}°"
                "<br>Selected CMIP6 cell: %{customdata[0]:.3f}°, "
                "%{customdata[1]:.3f}°<extra></extra>"
            ),
        ))
        highlighted = data[data["State"] == selected_state]
        if not highlighted.empty:
            fig.add_trace(go.Scattergeo(
                lon=highlighted["Target_Lon"], lat=highlighted["Target_Lat"],
                text=highlighted["Capital"] + ", " + highlighted["State"],
                mode="markers", name="Current selection",
                marker=dict(size=14, symbol="diamond", color="#D84C22",
                            line=dict(width=1.4, color="white")),
                hovertemplate="%{text}<extra>Current selection</extra>",
            ))
        fig.update_geos(
            scope="usa", projection_type="albers usa", showland=True,
            landcolor="#F7F9FB", showsubunits=True, subunitcolor="#C5CED8",
            showlakes=True, lakecolor="#ECF3F9", showcoastlines=False,
            bgcolor="white",
        )
        fig.update_layout(
            title=dict(text="50 representative U.S. state-capital locations",
                       x=0.01, font=dict(size=18)),
            template="plotly_white", height=630,
            margin=dict(l=25, r=25, t=80, b=10),
            legend=dict(orientation="h", y=0.03, x=0.02),
        )
        caption = (
            "Fig. 2. Geographic location of the 50 representative U.S. state capitals. "
            "The marker is the capital coordinate; modeled ambient temperatures "
            "come from the corresponding nearest ACCESS-CM2 downscaled grid cell."
        )
        export = data[["State", "Capital", "Target_Lat", "Target_Lon",
                       "CMIP6_Lat", "CMIP6_Lon"]].copy()
        return fig, caption, export, "fig02_capital_locations"

    if figure_key in ("temperature_spread", "dtr_spread"):
        is_temperature = figure_key == "temperature_spread"
        column = "Mean_Temperature_C" if is_temperature else "Mean_DTR_MVA"
        label = "Mean ambient temperature (°C)" if is_temperature else "Mean DTR (MVA)"
        data = case_df[["State", "Capital", "Scenario", "Period", column]].copy()
        fig = px.box(
            data, x="Scenario", y=column, color="Period", points="all",
            hover_data=["State", "Capital"],
            category_orders={"Scenario": scenario_order, "Period": period_order},
            color_discrete_map=period_colors,
            labels={"Scenario": "Climate scenario", column: label,
                    "Period": "Future period"},
        )
        fig.update_layout(boxmode="group")
        fig.update_traces(
            boxmean=True, jitter=0.34, pointpos=0,
            marker=dict(size=4, opacity=0.58), selector=dict(type="box"),
        )
        if not is_temperature:
            fig.add_hline(
                y=22.0, line_dash="dash", line_color="#252525",
                annotation_text="22 MVA nameplate",
                annotation_position="top left",
            )
        number = 3 if is_temperature else 4
        fig = finish_xy(
            fig,
            f"Across-capital spread of {label.lower()} | 50 capitals per scenario–period",
            height=590,
        )
        fig.update_xaxes(title_text="Climate scenario")
        fig.update_yaxes(title_text=label)
        metric_text = "temperature" if is_temperature else "DTR"
        caption = (
            f"Fig. {number}. Distribution of capital-location mean {metric_text} "
            "under three SSP scenarios for 2026–2050 and 2051–2075. "
            "Each box summarizes 50 separate representative locations; points "
            "are capital-location means, not individual minutes or statewide averages."
        )
        slug = "fig03_capital_temperature_spread" if is_temperature else "fig04_capital_dtr_spread"
        return fig, caption, data, slug

    if figure_key == "summer_comparison":
        summer = seasonal_df[seasonal_df["Season"] == "Summer"].copy()
        data = (
            summer.groupby(["Scenario", "Period"], as_index=False)
            .agg(
                Mean_DTR_MVA=("Mean_DTR_MVA", "mean"),
                Below_Nameplate_Percent=("DTR_Below_22MVA_Percent", "mean"),
                Number_of_Capitals=("State", "nunique"),
            )
        )
        fig = px.bar(
            data, x="Scenario", y="Mean_DTR_MVA", color="Period",
            barmode="group", text="Mean_DTR_MVA",
            custom_data=["Below_Nameplate_Percent", "Number_of_Capitals"],
            category_orders={"Scenario": scenario_order, "Period": period_order},
            color_discrete_map=period_colors,
            labels={"Scenario": "Climate scenario", "Mean_DTR_MVA": "Summer mean DTR (MVA)",
                    "Period": "Future period"},
        )
        fig.update_traces(
            texttemplate="%{y:.2f}", textposition="outside",
            hovertemplate=("%{x}<br>Mean summer DTR: %{y:.3f} MVA"
                           "<br>Mean modeled time below nameplate: %{customdata[0]:.2f}%"
                           "<br>Representative capitals: %{customdata[1]}"
                           "<extra>%{fullData.name}</extra>"),
        )
        fig.add_hline(
            y=22.0, line_color="#333333", line_dash="dash",
            annotation_text="22 MVA nameplate", annotation_position="top left",
        )
        fig = finish_xy(
            fig, "National capital-location summer DTR by climate scenario and period",
            height=575,
        )
        fig.update_yaxes(range=[0, float(data["Mean_DTR_MVA"].max()) + 3.5])
        caption = (
            "Fig. 9. Comparison of summer mean DTR between 2026–2050 and "
            "2051–2075 for the three SSPs. Each bar is the arithmetic mean of "
            "the 50 representative capital-location summer DTR values; "
            "the dashed line indicates the 22 MVA nameplate rating."
        )
        return fig, caption, data, "fig09_summer_dtr_periods"

    if figure_key == "temperature_dtr_scatter":
        data = case_df[case_df["Period"] == selected_period][
            ["State", "Capital", "Scenario", "Period", "Mean_Temperature_C",
             "Mean_DTR_MVA", "DTR_Below_22MVA_Percent"]
        ].copy()
        fig = px.scatter(
            data, x="Mean_Temperature_C", y="Mean_DTR_MVA", color="Scenario",
            hover_name="Capital", hover_data=["State", "DTR_Below_22MVA_Percent"],
            color_discrete_map=scenario_colors,
            category_orders={"Scenario": scenario_order}, opacity=0.78,
            labels={"Mean_Temperature_C": "Mean ambient temperature (°C)",
                    "Mean_DTR_MVA": "Mean dynamic rating (MVA)",
                    "Scenario": "Climate scenario"},
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        for scenario in scenario_order:
            subset = data[data["Scenario"] == scenario]
            if len(subset) < 3:
                continue
            xx = subset["Mean_Temperature_C"].to_numpy(dtype=float)
            yy = subset["Mean_DTR_MVA"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xx, yy, 1)
            r = float(np.corrcoef(xx, yy)[0, 1])
            line_x = np.array([xx.min(), xx.max()])
            fig.add_trace(go.Scatter(
                x=line_x, y=slope * line_x + intercept,
                mode="lines", line=dict(color=scenario_colors[scenario], dash="dash", width=2),
                name=f"{scenario} fit (r={r:+.2f})", legendgroup=scenario,
                hoverinfo="skip",
            ))
        fig.add_hline(y=22, line_dash="dot", line_color="#555555")
        fig = finish_xy(
            fig, f"Capital-location mean temperature and DTR | {selected_period}",
            height=620,
        )
        caption = (
            "Fig. 10. Relationship between period-mean ambient temperature and "
            f"period-mean DTR across 50 capital locations for each SSP ({selected_period}). "
            "Dashed curves show descriptive linear fits across locations; their "
            "correlations are not a site-level causal estimate of climate change."
        )
        return fig, caption, data, "fig10_temperature_vs_dtr_" + selected_period.replace("-", "_")

    if figure_key == "state_dtr_variation":
        data = case_df[
            (case_df["Scenario"] == selected_scenario)
            & (case_df["Period"] == selected_period)
        ][["State", "Capital", "Mean_DTR_MVA", "Mean_Temperature_C",
           "Capacity_Gain_Percent"]].copy().sort_values("Mean_DTR_MVA")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["Mean_DTR_MVA"], y=data["State"],
            mode="markers", name="Capital-location mean DTR",
            customdata=data[["Capital", "Mean_Temperature_C", "Capacity_Gain_Percent"]].to_numpy(),
            marker=dict(
                size=10, color=data["Mean_DTR_MVA"], colorscale="Viridis",
                showscale=True, colorbar=dict(title="Mean DTR<br>(MVA)"),
                line=dict(color="white", width=0.7),
            ),
            hovertemplate=("%{y} (%{customdata[0]})<br>Mean DTR: %{x:.3f} MVA"
                           "<br>Mean temperature: %{customdata[1]:.2f} °C"
                           "<br>Capacity gain: %{customdata[2]:+.2f}%<extra></extra>"),
        ))
        selected = data[data["State"] == selected_state]
        if not selected.empty:
            fig.add_trace(go.Scatter(
                x=selected["Mean_DTR_MVA"], y=selected["State"],
                mode="markers", name="Current selection",
                marker=dict(symbol="diamond", size=15, color="#D84C22",
                            line=dict(color="white", width=1)),
                hovertemplate="%{y}<br>%{x:.3f} MVA<extra>Current selection</extra>",
            ))
        fig.add_vline(x=22, line_dash="dash", line_color="#252525")
        fig = finish_xy(
            fig, f"Mean DTR by representative capital | {selected_scenario}, {selected_period}",
            height=1490, left_margin=145,
        )
        fig.update_yaxes(
            title_text="Representative state (capital location)",
            categoryorder="array", categoryarray=data["State"].tolist()[::-1],
        )
        fig.update_xaxes(
            title_text="Mean DTR (MVA)",
            range=[float(data["Mean_DTR_MVA"].min()) - 0.65,
                   float(data["Mean_DTR_MVA"].max()) + 0.65],
        )
        caption = (
            "Fig. 11. Geographic variation in the 50 individual capital-location mean "
            f"DTR results for {selected_scenario}, {selected_period}, ordered by DTR. "
            "Each dot represents one modeled capital; this is not a statewide spatial average."
        )
        return fig, caption, data, (
            "fig11_ranked_capitals_" + selected_scenario.replace(".", "").replace("-", "")
            + "_" + selected_period.replace("-", "_")
        )

    if figure_key == "state_dtr_change":
        selected = case_df[case_df["Scenario"] == selected_scenario]
        wide = selected.pivot_table(
            index=["State", "Capital"], columns="Period", values="Mean_DTR_MVA", aggfunc="first",
        ).reset_index()
        # Differences are LONG-TERM minus SHORT-TERM, for the same location and SSP.
        data = wide.dropna(subset=period_order).copy()
        data["Change_MVA"] = data["2051-2075"] - data["2026-2050"]
        data = data.sort_values("Change_MVA")
        fig = go.Figure(go.Bar(
            x=data["Change_MVA"], y=data["State"], orientation="h",
            marker_color=["#B54545" if change < 0 else "#1975AD"
                          for change in data["Change_MVA"]],
            customdata=data[["Capital", "2026-2050", "2051-2075"]].to_numpy(),
            hovertemplate=("%{y} (%{customdata[0]})"
                           "<br>2026–2050: %{customdata[1]:.3f} MVA"
                           "<br>2051–2075: %{customdata[2]:.3f} MVA"
                           "<br>Long minus short: %{x:+.3f} MVA<extra></extra>"),
            name="DTR change",
        ))
        fig.add_vline(x=0, line_color="#222222", line_width=1.5)
        fig = finish_xy(
            fig, f"Period-to-period change in capital-location mean DTR | {selected_scenario}",
            height=1490, left_margin=145,
        )
        fig.update_yaxes(
            title_text="State (capital location)",
            categoryorder="array", categoryarray=data["State"].tolist()[::-1],
        )
        fig.update_xaxes(title_text="Δ mean DTR: 2051–2075 minus 2026–2050 (MVA)")
        caption = (
            "Fig. 12. Change in mean DTR from 2026–2050 to 2051–2075 for "
            f"each of the 50 representative capital locations under {selected_scenario}. "
            "Negative changes indicate reduced projected dynamic capacity in the later period."
        )
        slug = "fig12_dtr_period_change_" + selected_scenario.replace(".", "").replace("-", "")
        return fig, caption, data, slug

    raise ValueError(f"Unknown research figure: {figure_key}")



# ─────────────────────────────────────────────────────────────
# MAIN COMPUTATION & PLOTTING
# ─────────────────────────────────────────────────────────────
# ============================================================
# KEEP FUTURE DASHBOARD ACTIVE AFTER INITIAL RUN
# ============================================================

if "future_dashboard_active" not in st.session_state:
    st.session_state.future_dashboard_active = False


# Activate Future DTR dashboard when user presses the button
if calc_mode == "Future U.S. DTR Projection":

    if run_btn:
        st.session_state.future_dashboard_active = True

else:

    # Reset when user switches to DTR or DCR mode
    st.session_state.future_dashboard_active = False


run_future_dashboard = (
    calc_mode == "Future U.S. DTR Projection"
    and
    st.session_state.future_dashboard_active
)


# ============================================================
# MAIN COMPUTATION & PLOTTING
# ============================================================

if run_btn or run_future_dashboard:

    # ============================================================
    # MODE 3: FUTURE U.S. DTR PROJECTION
    # ============================================================

    if calc_mode == "Future U.S. DTR Projection":

        # ========================================================
        # 1. IDENTIFY SELECTED STATE
        # ========================================================

        state_full = st.session_state.get(
            "state"
        )

        if not state_full:
            # Backup method only
            state_full = extract_state(
                st.session_state.address
            )

        if not state_full:
            st.error(
                "The selected U.S. state could not be identified. "
                "Please search for a U.S. city, click a U.S. "
                "location on the map, or enter coordinates."
            )

            st.stop()

        # ========================================================
        # 2. SCENARIO CODE
        # ========================================================

        scenario_code_map = {
            "SSP1-2.6": "ssp126",
            "SSP2-4.5": "ssp245",
            "SSP5-8.5": "ssp585"
        }

        selected_scenario_code = (
            scenario_code_map[future_scenario]
        )


        # ========================================================
        # 3. LOAD FUTURE DATABASE
        # ========================================================

        try:

            (
                future_case_df,
                future_seasonal_df,
                future_annual_df,
                future_histogram_df,
                future_mapping_df

            ) = load_future_dtr_database()

        except Exception as e:

            st.error(
                f"Future DTR database could not be loaded: {e}"
            )

            st.stop()


        # ========================================================
        # 4. SELECTED CASE
        # ========================================================

        selected_case_df = future_case_df[
            (future_case_df["State"] == state_full)
            &
            (
                future_case_df["Scenario_Code"]
                == selected_scenario_code
            )
            &
            (
                future_case_df["Period"]
                == future_period
            )
        ].copy()


        if selected_case_df.empty:

            st.error(
                f"No future DTR results were found for "
                f"{state_full}, {future_scenario}, {future_period}."
            )

            st.stop()


        selected_case = selected_case_df.iloc[0]

        representative_capital = (
            selected_case["Capital"]
        )


        # ========================================================
        # 5. HEADER
        # ========================================================

        st.success(
            "✅ Future DTR results loaded successfully!"
        )

        # ============================================================
        # FUTURE DTR DASHBOARD HEADER
        # ============================================================

        st.markdown(
            "## 🌎 Future U.S. Dynamic Transformer Rating"
        )

        st.markdown(
            """
            Explore how projected climate conditions may affect
            transformer dynamic rating across the United States.

            This dashboard combines **NASA NEX-GDDP-CMIP6 climate
            projections** with an **IEEE C57.91-2023 transformer
            thermal model** to evaluate future temperature-driven
            changes in transformer capacity.
            """
        )

        st.info(
            """
            **Study Configuration**

            • Climate model: **ACCESS-CM2**  
            • Ensemble: **r1i1p1f1**  
            • Climate scenarios: **SSP1-2.6, SSP2-4.5, SSP5-8.5**  
            • Future periods: **2026–2050 and 2051–2075**  
            • Transformer nameplate: **22 MVA**  
            • Hot-spot temperature limit: **110°C**  
            • Ambient safety margin: **+5°C**  
            • Geographic representation: **one state-capital location per state**
            """
        )

        st.caption(
            "National and state-polygon results represent the modeled "
            "capital-location value for each state, not a statewide "
            "spatial average."
        )

        st.info(
            f"""
            **Selected State:** {state_full}

            **Representative Research Location:**  
            {representative_capital}, {state_full}

            Future projections use one representative
            **state-capital location per state**.

            Results therefore represent the validated
            state-capital planning case rather than a site-specific
            projection for the exact selected coordinates.
            """
        )


        # ========================================================
        # 6. PREPARE STATE DATA
        # ========================================================

        state_case_df = future_case_df[
            future_case_df["State"] == state_full
        ].copy()


        state_selected_scenario_df = (
            state_case_df[
                state_case_df["Scenario_Code"]
                == selected_scenario_code
            ].copy()
        )


        state_seasonal_df = (
            future_seasonal_df[
                future_seasonal_df["State"]
                == state_full
            ].copy()
        )


        state_annual_df = (
            future_annual_df[
                future_annual_df["State"]
                == state_full
            ].copy()
        )


        state_histogram_df = (
            future_histogram_df[
                future_histogram_df["State"]
                == state_full
            ].copy()
        )


        # ========================================================
        # 7. NATIONAL CAPITAL-LOCATION AVERAGES
        # ========================================================

        national_case_df = (
            future_case_df
            .groupby(
                [
                    "Scenario_Code",
                    "Period"
                ],
                as_index=False
            )
            .agg(
                Mean_Temperature_C=(
                    "Mean_Temperature_C",
                    "mean"
                ),

                Mean_DTR_MVA=(
                    "Mean_DTR_MVA",
                    "mean"
                ),

                Capacity_Gain_Percent=(
                    "Capacity_Gain_Percent",
                    "mean"
                ),

                DTR_Below_22MVA_Percent=(
                    "DTR_Below_22MVA_Percent",
                    "mean"
                )
            )
        )


        national_case_df["Scenario_Label"] = (
            national_case_df["Scenario_Code"]
            .map(FUTURE_SCENARIO_LABELS)
        )


        # ========================================================
        # 8. NATIONAL ANNUAL SERIES
        # ========================================================

        national_annual_df = (
            future_annual_df
            .groupby(
                [
                    "Scenario_Code",
                    "Year"
                ],
                as_index=False
            )
            .agg(
                Mean_Temperature_C=(
                    "Mean_Temperature_C",
                    "mean"
                ),

                Mean_DTR_MVA=(
                    "Mean_DTR_MVA",
                    "mean"
                ),

                Capacity_Gain_Percent=(
                    "Capacity_Gain_Percent",
                    "mean"
                ),

                DTR_Below_22MVA_Percent=(
                    "DTR_Below_22MVA_Percent",
                    "mean"
                )
            )
        )


        national_annual_df["Scenario_Label"] = (
            national_annual_df[
                "Scenario_Code"
            ].map(
                FUTURE_SCENARIO_LABELS
            )
        )


        # ========================================================
        # 9. NATIONAL SEASONAL SERIES
        # ========================================================

        national_seasonal_df = (
            future_seasonal_df
            .groupby(
                [
                    "Scenario_Code",
                    "Period",
                    "Season"
                ],
                as_index=False
            )
            .agg(
                Mean_Temperature_C=(
                    "Mean_Temperature_C",
                    "mean"
                ),

                Mean_DTR_MVA=(
                    "Mean_DTR_MVA",
                    "mean"
                ),

                Capacity_Gain_Percent=(
                    "Capacity_Gain_Percent",
                    "mean"
                ),

                DTR_Below_22MVA_Percent=(
                    "DTR_Below_22MVA_Percent",
                    "mean"
                )
            )
        )


        national_seasonal_df[
            "Scenario_Label"
        ] = (
            national_seasonal_df[
                "Scenario_Code"
            ].map(
                FUTURE_SCENARIO_LABELS
            )
        )


        # ========================================================
        # 10. CREATE FUTURE DASHBOARD TABS
        # ========================================================

        (
            future_tab1,
            future_tab2,
            future_tab3,
            future_tab4,
            future_tab5,
            future_tab6,
            future_tab7,
            future_tab8

        ) = st.tabs(
            [
                "🌎 Overview",
                "🔄 Future Change",
                "🇺🇸 National Comparison",
                "📈 Annual Trends",
                "🍂 Seasonal Analysis",
                "🗺️ National Maps",
                "📊 Distribution",
                "📑 Research Figures"
            ]
        )


        # ========================================================
        # TAB 1 — OVERVIEW
        # ========================================================

        with future_tab1:

            st.subheader(
                f"{state_full} | "
                f"{future_scenario} | "
                f"{future_period}"
            )

            st.caption(
                f"Representative location: "
                f"{representative_capital}, {state_full}"
            )


            # ----------------------------------------------------
            # Main KPIs
            # ----------------------------------------------------

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Mean Temperature",
                f"{selected_case['Mean_Temperature_C']:.2f} °C"
            )

            col2.metric(
                "Mean DTR",
                f"{selected_case['Mean_DTR_MVA']:.2f} MVA"
            )

            col3.metric(
                "Capacity Gain",
                f"{selected_case['Capacity_Gain_Percent']:.2f}%"
            )

            col4.metric(
                "DTR < 22 MVA",
                f"{selected_case['DTR_Below_22MVA_Percent']:.2f}%"
            )


            st.markdown(
                "### ⚡ DTR Range"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Minimum DTR",
                f"{selected_case['Min_DTR_MVA']:.2f} MVA"
            )

            col2.metric(
                "Mean DTR",
                f"{selected_case['Mean_DTR_MVA']:.2f} MVA"
            )

            col3.metric(
                "Maximum DTR",
                f"{selected_case['Max_DTR_MVA']:.2f} MVA"
            )


            st.markdown(
                "### 📊 DTR Percentiles"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "P10 DTR",
                f"{selected_case['P10_DTR_MVA']:.2f} MVA"
            )

            col2.metric(
                "Median DTR",
                f"{selected_case['Median_DTR_MVA']:.2f} MVA"
            )

            col3.metric(
                "P90 DTR",
                f"{selected_case['P90_DTR_MVA']:.2f} MVA"
            )


            st.markdown(
                "### 🚦 Dynamic Capacity Exposure"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "DTR < 22 MVA",
                f"{selected_case['DTR_Below_22MVA_Percent']:.2f}%"
            )

            col2.metric(
                "DTR < 21 MVA",
                f"{selected_case['DTR_Below_21MVA_Percent']:.2f}%"
            )

            col3.metric(
                "DTR ≥ 25 MVA",
                f"{selected_case['DTR_GE_25MVA_Percent']:.2f}%"
            )


            st.markdown(
                "### 🌡️ Temperature Range"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Minimum Temperature",
                f"{selected_case['Min_Temperature_C']:.2f} °C"
            )

            col2.metric(
                "Mean Temperature",
                f"{selected_case['Mean_Temperature_C']:.2f} °C"
            )

            col3.metric(
                "Maximum Temperature",
                f"{selected_case['Max_Temperature_C']:.2f} °C"
            )

            # ============================================================
            # SELECTED CASE SUMMARY
            # ============================================================

            st.markdown("---")
            st.markdown("### 📝 Selected Case Summary")

            # ------------------------------------------------------------
            # 1. Find the most constrained season
            # ------------------------------------------------------------

            summary_seasonal_df = state_seasonal_df[
                (
                        state_seasonal_df["Scenario_Code"]
                        == selected_scenario_code
                )
                &
                (
                        state_seasonal_df["Period"]
                        == future_period
                )
                ].copy()

            if not summary_seasonal_df.empty:

                most_constrained_row = (
                    summary_seasonal_df
                    .sort_values("Mean_DTR_MVA")
                    .iloc[0]
                )

                most_constrained_season = (
                    most_constrained_row["Season"]
                )

                most_constrained_dtr = (
                    most_constrained_row["Mean_DTR_MVA"]
                )

                if (
                        "DTR_Below_22MVA_Percent"
                        in most_constrained_row.index
                ):
                    most_constrained_below22 = (
                        most_constrained_row[
                            "DTR_Below_22MVA_Percent"
                        ]
                    )
                else:
                    most_constrained_below22 = np.nan

            else:

                most_constrained_season = "N/A"
                most_constrained_dtr = np.nan
                most_constrained_below22 = np.nan

            # ------------------------------------------------------------
            # 2. Calculate short-term to long-term change
            # ------------------------------------------------------------

            summary_comparison_df = (
                state_selected_scenario_df
                .set_index("Period")
            )

            if (
                    "2026-2050" in summary_comparison_df.index
                    and
                    "2051-2075" in summary_comparison_df.index
            ):

                summary_short = (
                    summary_comparison_df.loc[
                        "2026-2050"
                    ]
                )

                summary_long = (
                    summary_comparison_df.loc[
                        "2051-2075"
                    ]
                )

                summary_delta_temp = (
                        summary_long["Mean_Temperature_C"]
                        -
                        summary_short["Mean_Temperature_C"]
                )

                summary_delta_dtr = (
                        summary_long["Mean_DTR_MVA"]
                        -
                        summary_short["Mean_DTR_MVA"]
                )

                summary_delta_gain = (
                        summary_long["Capacity_Gain_Percent"]
                        -
                        summary_short["Capacity_Gain_Percent"]
                )

                summary_delta_below22 = (
                        summary_long["DTR_Below_22MVA_Percent"]
                        -
                        summary_short["DTR_Below_22MVA_Percent"]
                )

            else:

                summary_delta_temp = np.nan
                summary_delta_dtr = np.nan
                summary_delta_gain = np.nan
                summary_delta_below22 = np.nan

            # ------------------------------------------------------------
            # 3. Main selected-case information
            # ------------------------------------------------------------

            st.markdown(
                f"""
                **State:** {state_full}  
                **Representative location:** {representative_capital}, {state_full}  
                **Climate scenario:** {future_scenario}  
                **Study period:** {future_period}
                """
            )

            # ------------------------------------------------------------
            # 4. Main case KPI cards
            # ------------------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Mean Temperature",
                f"{selected_case['Mean_Temperature_C']:.2f} °C"
            )

            col2.metric(
                "Mean DTR",
                f"{selected_case['Mean_DTR_MVA']:.2f} MVA"
            )

            col3.metric(
                "Capacity Gain",
                f"{selected_case['Capacity_Gain_Percent']:.2f}%"
            )

            col4.metric(
                "DTR < 22 MVA",
                f"{selected_case['DTR_Below_22MVA_Percent']:.2f}%"
            )

            # ------------------------------------------------------------
            # 5. Distribution summary
            # ------------------------------------------------------------

            st.markdown("#### DTR Distribution")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "P10",
                f"{selected_case['P10_DTR_MVA']:.2f} MVA"
            )

            col2.metric(
                "Median",
                f"{selected_case['Median_DTR_MVA']:.2f} MVA"
            )

            col3.metric(
                "P90",
                f"{selected_case['P90_DTR_MVA']:.2f} MVA"
            )

            # ------------------------------------------------------------
            # 6. Most constrained season
            # ------------------------------------------------------------

            st.markdown("#### Most Constrained Season")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Season",
                most_constrained_season
            )

            if not np.isnan(most_constrained_dtr):

                col2.metric(
                    "Seasonal Mean DTR",
                    f"{most_constrained_dtr:.2f} MVA"
                )

            else:

                col2.metric(
                    "Seasonal Mean DTR",
                    "N/A"
                )

            if not np.isnan(most_constrained_below22):

                col3.metric(
                    "Seasonal DTR < 22",
                    f"{most_constrained_below22:.2f}%"
                )

            else:

                col3.metric(
                    "Seasonal DTR < 22",
                    "N/A"
                )

            # ------------------------------------------------------------
            # 7. Long-term change summary
            # ------------------------------------------------------------

            st.markdown(
                "#### 2051–2075 Change Relative to 2026–2050"
            )

            col1, col2, col3, col4 = st.columns(4)

            if not np.isnan(summary_delta_temp):
                col1.metric(
                    "Δ Temperature",
                    f"{summary_delta_temp:+.2f} °C"
                )

                col2.metric(
                    "Δ Mean DTR",
                    f"{summary_delta_dtr:+.2f} MVA"
                )

                col3.metric(
                    "Δ Capacity Gain",
                    f"{summary_delta_gain:+.2f} pp"
                )

                col4.metric(
                    "Δ DTR < 22",
                    f"{summary_delta_below22:+.2f} pp"
                )

            # ------------------------------------------------------------
            # 8. Short interpretation
            # ------------------------------------------------------------

            st.info(
                f"""
                For **{state_full}** under **{future_scenario}**
                during **{future_period}**, the modeled mean dynamic
                transformer rating is **{selected_case['Mean_DTR_MVA']:.2f} MVA**,
                corresponding to an average capacity gain of
                **{selected_case['Capacity_Gain_Percent']:.2f}%**
                relative to the 22 MVA nameplate.

                The calculated DTR is below the 22 MVA nameplate for
                **{selected_case['DTR_Below_22MVA_Percent']:.2f}%**
                of modeled time.

                The most constrained season is **{most_constrained_season}**,
                with a seasonal mean DTR of
                **{most_constrained_dtr:.2f} MVA**.
                """
            )

            # ============================================================
            # DOWNLOAD SELECTED FUTURE RESULTS
            # ============================================================

            st.markdown("---")
            st.markdown("### ⬇️ Download Future Results")

            download_case = state_case_df.copy()

            download_seasonal = future_seasonal_df[
                future_seasonal_df["State"]
                == state_full
                ].copy()

            download_annual = future_annual_df[
                future_annual_df["State"]
                == state_full
                ].copy()

            download_histogram = future_histogram_df[
                future_histogram_df["State"]
                == state_full
                ].copy()

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "⬇️ State Summary CSV",
                    data=download_case.to_csv(
                        index=False
                    ),
                    file_name=(
                        f"{state_full}_Future_DTR_Summary.csv"
                    ),
                    mime="text/csv"
                )

            with col2:
                st.download_button(
                    "⬇️ Seasonal Results CSV",
                    data=download_seasonal.to_csv(
                        index=False
                    ),
                    file_name=(
                        f"{state_full}_Future_DTR_Seasonal.csv"
                    ),
                    mime="text/csv"
                )

            col3, col4 = st.columns(2)

            with col3:
                st.download_button(
                    "⬇️ Annual Results CSV",
                    data=download_annual.to_csv(
                        index=False
                    ),
                    file_name=(
                        f"{state_full}_Future_DTR_Annual.csv"
                    ),
                    mime="text/csv"
                )

            with col4:
                st.download_button(
                    "⬇️ DTR Distribution CSV",
                    data=download_histogram.to_csv(
                        index=False
                    ),
                    file_name=(
                        f"{state_full}_Future_DTR_Distribution.csv"
                    ),
                    mime="text/csv"
                )

            # ============================================================
            # FUTURE DTR METHODOLOGY / LIMITATIONS
            # ============================================================

            st.markdown("---")

            with st.expander(
                    "ℹ️ Future DTR Methodology and Interpretation"
            ):

                st.markdown(
                    """
                    ### Climate dataset
    
                    Future temperature projections are obtained from
                    **NASA NEX-GDDP-CMIP6** using:
    
                    - Climate model: **ACCESS-CM2**
                    - Ensemble: **r1i1p1f1**
                    - Variable: **near-surface air temperature (tas)**
                    - Scenarios:
                      **SSP1-2.6, SSP2-4.5, SSP5-8.5**
                    - Study periods:
                      **2026–2050 and 2051–2075**
    
                    ### Geographic representation
    
                    One representative location is used for each U.S. state:
                    the **state capital**.
    
                    Values displayed by state polygons therefore represent
                    the corresponding capital-location result and should not
                    be interpreted as a spatial average across the entire state.
    
                    ### Temporal processing
    
                    Daily CMIP6 temperature values were converted to hourly
                    temperature using linear interpolation and subsequently
                    evaluated at minute resolution for the transformer DTR model.
    
                    The interpolation is a temporal-disaggregation assumption;
                    it does not create genuine hourly climate-model information.
    
                    ### Transformer model
    
                    Future DTR calculations use the same validated research
                    transformer for all states and scenarios:
    
                    - Nameplate rating: **22 MVA**
                    - Thermal model: **IEEE C57.91-2023**
                    - Maximum hot-spot temperature: **110°C**
                    - Ambient safety margin: **+5°C**
    
                    ### Interpretation of DTR < 22 MVA
    
                    The reported percentage represents the fraction of modeled
                    time for which the calculated dynamic transformer rating is
                    below the 22 MVA nameplate rating.
    
                    It does **not** represent transformer overload frequency,
                    because actual electrical loading or customer demand is not
                    modeled in this analysis.
    
                    ### National summaries
    
                    National results are simple arithmetic averages of the
                    50 representative state-capital locations.
    
                    They are not area-weighted, population-weighted,
                    or statewide spatial averages.
                    """
                )


        # ========================================================
        # TAB 2 — SHORT TERM VS LONG TERM
        # ========================================================

        with future_tab2:

            st.subheader(
                f"Projected Change — "
                f"{state_full} | {future_scenario}"
            )

            comparison_df = (
                state_selected_scenario_df
                .set_index("Period")
            )


            if (
                "2026-2050"
                in comparison_df.index
                and
                "2051-2075"
                in comparison_df.index
            ):

                short_row = (
                    comparison_df.loc[
                        "2026-2050"
                    ]
                )

                long_row = (
                    comparison_df.loc[
                        "2051-2075"
                    ]
                )


                delta_temp = (
                    long_row[
                        "Mean_Temperature_C"
                    ]
                    -
                    short_row[
                        "Mean_Temperature_C"
                    ]
                )

                delta_dtr = (
                    long_row[
                        "Mean_DTR_MVA"
                    ]
                    -
                    short_row[
                        "Mean_DTR_MVA"
                    ]
                )

                delta_gain = (
                    long_row[
                        "Capacity_Gain_Percent"
                    ]
                    -
                    short_row[
                        "Capacity_Gain_Percent"
                    ]
                )

                delta_below22 = (
                    long_row[
                        "DTR_Below_22MVA_Percent"
                    ]
                    -
                    short_row[
                        "DTR_Below_22MVA_Percent"
                    ]
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Δ Mean Temperature",
                    f"{delta_temp:+.2f} °C"
                )

                col2.metric(
                    "Δ Mean DTR",
                    f"{delta_dtr:+.2f} MVA"
                )

                col3.metric(
                    "Δ Capacity Gain",
                    f"{delta_gain:+.2f} pp"
                )

                col4.metric(
                    "Δ DTR < 22 MVA",
                    f"{delta_below22:+.2f} pp"
                )


                st.caption(
                    "Change = 2051–2075 minus 2026–2050. "
                    "pp = percentage points."
                )


                comparison_table = pd.DataFrame(
                    {
                        "Metric": [
                            "Mean Temperature (°C)",
                            "Mean DTR (MVA)",
                            "Capacity Gain (%)",
                            "DTR < 22 MVA (%)"
                        ],

                        "2026-2050": [
                            short_row[
                                "Mean_Temperature_C"
                            ],

                            short_row[
                                "Mean_DTR_MVA"
                            ],

                            short_row[
                                "Capacity_Gain_Percent"
                            ],

                            short_row[
                                "DTR_Below_22MVA_Percent"
                            ]
                        ],

                        "2051-2075": [
                            long_row[
                                "Mean_Temperature_C"
                            ],

                            long_row[
                                "Mean_DTR_MVA"
                            ],

                            long_row[
                                "Capacity_Gain_Percent"
                            ],

                            long_row[
                                "DTR_Below_22MVA_Percent"
                            ]
                        ],

                        "Change": [
                            delta_temp,
                            delta_dtr,
                            delta_gain,
                            delta_below22
                        ]
                    }
                )


                st.dataframe(
                    comparison_table.style.format(
                        {
                            "2026-2050": "{:.2f}",
                            "2051-2075": "{:.2f}",
                            "Change": "{:+.2f}"
                        }
                    ),

                    use_container_width=True,
                    hide_index=True
                )


        # ========================================================
        # TAB 3 — NATIONAL COMPARISON
        # ========================================================

        with future_tab3:

            st.subheader(
                "🇺🇸 National Capital-Location Comparison"
            )

            st.info(
                """
                National values are arithmetic averages across
                the 50 representative state-capital locations.

                They are not area-weighted, population-weighted,
                or statewide spatial averages.
                """
            )


            # ----------------------------------------------------
            # Mean Temperature
            # ----------------------------------------------------

            fig_temp_compare = px.bar(
                national_case_df,
                x="Scenario_Label",
                y="Mean_Temperature_C",
                color="Period",
                barmode="group",
                text_auto=".2f",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ],

                    "Period": FUTURE_PERIOD_ORDER
                },

                color_discrete_map=
                    FUTURE_PERIOD_COLORS
            )


            fig_temp_compare.update_layout(
                title=(
                    "National Capital-Location Average "
                    "Mean Temperature"
                    "<br>"
                    "<sup>Comparison by Scenario "
                    "and Period</sup>"
                ),

                xaxis_title="Climate Scenario",

                yaxis_title=(
                    "Mean Temperature (°C)"
                ),

                legend_title="Period"
            )


            fig_temp_compare.update_traces(
                textposition="outside"
            )


            # ----------------------------------------------------
            # Mean DTR
            # ----------------------------------------------------

            fig_dtr_compare = px.bar(
                national_case_df,
                x="Scenario_Label",
                y="Mean_DTR_MVA",
                color="Period",
                barmode="group",
                text_auto=".2f",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ],

                    "Period":
                        FUTURE_PERIOD_ORDER
                },

                color_discrete_map=
                    FUTURE_PERIOD_COLORS
            )


            fig_dtr_compare.add_hline(
                y=22,
                line_dash="dash",
                line_color="black",
                annotation_text=(
                    "22 MVA Nameplate"
                )
            )


            fig_dtr_compare.update_layout(
                title=(
                    "National Capital-Location Average "
                    "Mean DTR"
                    "<br>"
                    "<sup>Comparison by Scenario "
                    "and Period</sup>"
                ),

                xaxis_title="Climate Scenario",

                yaxis_title="Mean DTR (MVA)",

                legend_title="Period"
            )


            fig_dtr_compare.update_traces(
                textposition="outside"
            )

            fig_temp_compare = style_future_plot(
                fig_temp_compare,
                height=440,
                hovermode="closest"
            )

            fig_dtr_compare = style_future_plot(
                fig_dtr_compare,
                height=440,
                hovermode="closest"
            )


            # ============================================================
            # FIX NATIONAL BAR CHART LABEL CLIPPING
            # ============================================================

            def fix_national_bar_chart(fig, max_value, short_title):
                fig.update_layout(

                    # Shorter title prevents horizontal clipping
                    title=dict(
                        text=short_title,
                        x=0.02,
                        xanchor="left",
                        font=dict(
                            color="#000000",
                            size=16
                        )
                    ),

                    margin=dict(
                        l=60,
                        r=20,
                        t=95,
                        b=65
                    )
                )

                # Add 30% extra space above tallest bar
                fig.update_yaxes(
                    range=[0, float(max_value) * 1.30],
                    autorange=False
                )

                # Make numerical values completely visible
                fig.update_traces(
                    texttemplate="%{y:.2f}",
                    textposition="outside",
                    textfont=dict(
                        color="#000000",
                        size=12
                    ),
                    cliponaxis=False,
                    selector=dict(type="bar")
                )

                return fig


            # Fix Temperature chart
            fig_temp_compare = fix_national_bar_chart(
                fig_temp_compare,
                national_case_df["Mean_Temperature_C"].max(),
                "National Mean Temperature"
            )

            # Fix DTR chart
            fig_dtr_compare = fix_national_bar_chart(
                fig_dtr_compare,
                national_case_df["Mean_DTR_MVA"].max(),
                "National Mean DTR"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.plotly_chart(
                    fig_temp_compare,
                    use_container_width=True
                )

            with col2:

                st.plotly_chart(
                    fig_dtr_compare,
                    use_container_width=True
                )


            # ----------------------------------------------------
            # Capacity Gain
            # ----------------------------------------------------

            fig_gain_compare = px.bar(
                national_case_df,
                x="Scenario_Label",
                y="Capacity_Gain_Percent",
                color="Period",
                barmode="group",
                text_auto=".2f",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ],

                    "Period":
                        FUTURE_PERIOD_ORDER
                },

                color_discrete_map=
                    FUTURE_PERIOD_COLORS
            )


            fig_gain_compare.update_layout(
                title=(
                    "National Capital-Location Average "
                    "Capacity Gain"
                    "<br>"
                    "<sup>Comparison by Scenario "
                    "and Period</sup>"
                ),

                xaxis_title="Climate Scenario",

                yaxis_title="Capacity Gain (%)",

                legend_title="Period"
            )


            fig_gain_compare.update_traces(
                textposition="outside"
            )


            # ----------------------------------------------------
            # DTR < 22 MVA
            # ----------------------------------------------------

            fig_below22_compare = px.bar(
                national_case_df,
                x="Scenario_Label",
                y="DTR_Below_22MVA_Percent",
                color="Period",
                barmode="group",
                text_auto=".2f",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ],

                    "Period":
                        FUTURE_PERIOD_ORDER
                },

                color_discrete_map=
                    FUTURE_PERIOD_COLORS
            )


            fig_below22_compare.update_layout(
                title=(
                    "National Capital-Location Average "
                    "DTR < 22 MVA Exposure"
                    "<br>"
                    "<sup>Comparison by Scenario "
                    "and Period</sup>"
                ),

                xaxis_title="Climate Scenario",

                yaxis_title=(
                    "Percent of Modeled Time (%)"
                ),

                legend_title="Period"
            )


            fig_below22_compare.update_traces(
                textposition="outside"
            )

            fig_gain_compare = style_future_plot(
                fig_gain_compare,
                height=440,
                hovermode="closest"
            )

            fig_below22_compare = style_future_plot(
                fig_below22_compare,
                height=440,
                hovermode="closest"
            )
            # Fix Capacity Gain chart
            fig_gain_compare = fix_national_bar_chart(
                fig_gain_compare,
                national_case_df["Capacity_Gain_Percent"].max(),
                "National Capacity Gain"
            )

            # Fix DTR Below 22 MVA chart
            fig_below22_compare = fix_national_bar_chart(
                fig_below22_compare,
                national_case_df["DTR_Below_22MVA_Percent"].max(),
                "National DTR Below 22 MVA"
            )


            col1, col2 = st.columns(2)

            with col1:

                st.plotly_chart(
                    fig_gain_compare,
                    use_container_width=True
                )

            with col2:

                st.plotly_chart(
                    fig_below22_compare,
                    use_container_width=True
                )


        # ========================================================
        # TAB 4 — ANNUAL TRENDS
        # ========================================================

        with future_tab4:

            st.subheader(
                "📈 National Annual Trends — 2026 to 2075"
            )


            # ============================================================
            # NATIONAL TREND KPI CARDS
            # ============================================================

            def calculate_trend_per_decade(df, value_column):
                df = df.sort_values("Year")

                x = df["Year"].to_numpy(dtype=float)
                y = df[value_column].to_numpy(dtype=float)

                slope_per_year = np.polyfit(x, y, 1)[0]

                return slope_per_year * 10.0


            trend_rows = []

            for scenario_code in FUTURE_SCENARIO_ORDER:
                scenario_df = national_annual_df[
                    national_annual_df["Scenario_Code"]
                    == scenario_code
                    ].copy()

                trend_rows.append(
                    {
                        "Scenario":
                            FUTURE_SCENARIO_LABELS[scenario_code],

                        "Temperature Trend":
                            calculate_trend_per_decade(
                                scenario_df,
                                "Mean_Temperature_C"
                            ),

                        "DTR Trend":
                            calculate_trend_per_decade(
                                scenario_df,
                                "Mean_DTR_MVA"
                            ),

                        "DTR < 22 Trend":
                            calculate_trend_per_decade(
                                scenario_df,
                                "DTR_Below_22MVA_Percent"
                            )
                    }
                )

            trend_df = pd.DataFrame(trend_rows)

            st.markdown(
                "### 📌 2026–2075 Trend Summary"
            )

            for _, row in trend_df.iterrows():
                st.markdown(
                    f"#### {row['Scenario']}"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Temperature Trend",
                    f"{row['Temperature Trend']:+.3f} °C/decade"
                )

                col2.metric(
                    "DTR Trend",
                    f"{row['DTR Trend']:+.3f} MVA/decade"
                )

                col3.metric(
                    "DTR < 22 Trend",
                    f"{row['DTR < 22 Trend']:+.3f} pp/decade"
                )


            # ----------------------------------------------------
            # Annual Temperature
            # ----------------------------------------------------

            fig_annual_temp = px.line(
                national_annual_df,
                x="Year",
                y="Mean_Temperature_C",
                color="Scenario_Label",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ]
                },

                color_discrete_map=
                    FUTURE_SCENARIO_COLORS
            )


            fig_annual_temp.update_layout(
                title=(
                    "National Capital-Location Average "
                    "Annual Mean Temperature "
                    "(2026–2075)"
                ),

                xaxis_title="Year",

                yaxis_title=(
                    "Mean Temperature (°C)"
                ),

                legend_title=""
            )


            fig_annual_temp.update_traces(
                line=dict(width=3)
            )

            fig_annual_temp = style_future_plot(
                fig_annual_temp,
                height=500,
                hovermode="x unified"
            )


            st.plotly_chart(
                fig_annual_temp,
                use_container_width=True
            )


            # ----------------------------------------------------
            # Annual DTR
            # ----------------------------------------------------

            fig_annual_dtr = px.line(
                national_annual_df,
                x="Year",
                y="Mean_DTR_MVA",
                color="Scenario_Label",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ]
                },

                color_discrete_map=
                    FUTURE_SCENARIO_COLORS
            )


            fig_annual_dtr.add_hline(
                y=22,
                line_dash="dash",
                line_color="black",
                annotation_text="22 MVA"
            )


            fig_annual_dtr.update_layout(
                title=(
                    "National Capital-Location Average "
                    "Annual Mean DTR (2026–2075)"
                ),

                xaxis_title="Year",

                yaxis_title="Mean DTR (MVA)",

                legend_title=""
            )


            fig_annual_dtr.update_traces(
                line=dict(width=3)
            )

            fig_annual_dtr = style_future_plot(
                fig_annual_dtr,
                height=500,
                hovermode="x unified"
            )


            st.plotly_chart(
                fig_annual_dtr,
                use_container_width=True
            )


            # ----------------------------------------------------
            # Annual DTR < 22
            # ----------------------------------------------------

            fig_annual_below22 = px.line(
                national_annual_df,
                x="Year",
                y="DTR_Below_22MVA_Percent",
                color="Scenario_Label",

                category_orders={
                    "Scenario_Label": [
                        "SSP1-2.6",
                        "SSP2-4.5",
                        "SSP5-8.5"
                    ]
                },

                color_discrete_map=
                    FUTURE_SCENARIO_COLORS
            )


            fig_annual_below22.update_layout(
                title=(
                    "National Capital-Location Average "
                    "Annual DTR < 22 MVA Exposure "
                    "(2026–2075)"
                ),

                xaxis_title="Year",

                yaxis_title=(
                    "Percent of Modeled Time (%)"
                ),

                legend_title=""
            )

            fig_annual_below22.update_traces(
                line=dict(width=3)
            )

            fig_annual_below22 = style_future_plot(
                fig_annual_below22,
                height=500,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_annual_below22,
                use_container_width=True
            )


        # ========================================================
        # TAB 5 — SEASONAL ANALYSIS
        # ========================================================

        with future_tab5:

            st.subheader(
                "🍂 Seasonal DTR Analysis"
            )


            # ----------------------------------------------------
            # Selected state
            # ----------------------------------------------------

            selected_state_seasonal = (
                state_seasonal_df[
                    (
                        state_seasonal_df[
                            "Scenario_Code"
                        ]
                        ==
                        selected_scenario_code
                    )
                    &
                    (
                        state_seasonal_df[
                            "Period"
                        ]
                        ==
                        future_period
                    )
                ].copy()
            )


            selected_state_seasonal[
                "Season"
            ] = pd.Categorical(
                selected_state_seasonal[
                    "Season"
                ],

                categories=
                    FUTURE_SEASON_ORDER,

                ordered=True
            )


            selected_state_seasonal = (
                selected_state_seasonal
                .sort_values("Season")
            )


            st.markdown(
                f"### 📍 {state_full} Seasonal Mean DTR"
            )


            fig_state_season = px.bar(
                selected_state_seasonal,
                x="Season",
                y="Mean_DTR_MVA",
                color="Season",
                text_auto=".2f",

                category_orders={
                    "Season":
                        FUTURE_SEASON_ORDER
                },

                color_discrete_map=
                    SEASON_COLOR
            )


            fig_state_season.add_hline(
                y=22,
                line_dash="dash",
                line_color="black",
                annotation_text="22 MVA"
            )

            fig_state_season.update_layout(
                title=(
                    f"{state_full} Seasonal Mean DTR | "
                    f"{future_scenario} | "
                    f"{future_period}"
                ),

                yaxis_title="Mean DTR (MVA)",

                showlegend=False
            )

            fig_state_season = style_future_plot(
                fig_state_season,
                height=460,
                hovermode="closest"
            )

            st.plotly_chart(
                fig_state_season,
                use_container_width=True
            )


            # ====================================================
            # NATIONAL SEASONAL FUNCTION
            # ====================================================

            def make_seasonal_dtr_figure(
                period
            ):

                df = national_seasonal_df[
                    national_seasonal_df[
                        "Period"
                    ] == period
                ].copy()


                fig = px.bar(
                    df,
                    x="Season",
                    y="Mean_DTR_MVA",
                    color="Scenario_Label",
                    barmode="group",
                    text_auto=".2f",

                    category_orders={
                        "Season":
                            FUTURE_SEASON_ORDER,

                        "Scenario_Label": [
                            "SSP1-2.6",
                            "SSP2-4.5",
                            "SSP5-8.5"
                        ]
                    },

                    color_discrete_map=
                        FUTURE_SCENARIO_COLORS
                )


                fig.add_hline(
                    y=22,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="22 MVA"
                )


                fig.update_layout(
                    title=(
                        "Seasonal National "
                        "Capital-Location Average "
                        f"Mean DTR ({period})"
                    ),

                    xaxis_title="Season",

                    yaxis_title="Mean DTR (MVA)",

                    legend_title="Scenario"
                )

                fig = style_future_plot(
                    fig,
                    height=460,
                    hovermode="closest"
                )

                return fig


            def make_seasonal_below22_figure(
                period
            ):

                df = national_seasonal_df[
                    national_seasonal_df[
                        "Period"
                    ] == period
                ].copy()


                fig = px.bar(
                    df,
                    x="Season",
                    y="DTR_Below_22MVA_Percent",
                    color="Scenario_Label",
                    barmode="group",
                    text_auto=".1f",

                    category_orders={
                        "Season":
                            FUTURE_SEASON_ORDER,

                        "Scenario_Label": [
                            "SSP1-2.6",
                            "SSP2-4.5",
                            "SSP5-8.5"
                        ]
                    },

                    color_discrete_map=
                        FUTURE_SCENARIO_COLORS
                )


                fig.update_layout(
                    title=(
                        "Seasonal National "
                        "Capital-Location Average "
                        "DTR < 22 MVA Exposure "
                        f"({period})"
                    ),

                    xaxis_title="Season",

                    yaxis_title=(
                        "Percent of Modeled Time (%)"
                    ),

                    legend_title="Scenario"
                )

                fig = style_future_plot(
                    fig,
                    height=460,
                    hovermode="closest"
                )

                return fig


            st.markdown(
                "### 🇺🇸 National Seasonal Comparison"
            )


            st.plotly_chart(
                make_seasonal_dtr_figure(
                    "2026-2050"
                ),

                use_container_width=True
            )


            st.plotly_chart(
                make_seasonal_dtr_figure(
                    "2051-2075"
                ),

                use_container_width=True
            )


            st.plotly_chart(
                make_seasonal_below22_figure(
                    "2026-2050"
                ),

                use_container_width=True
            )


            st.plotly_chart(
                make_seasonal_below22_figure(
                    "2051-2075"
                ),

                use_container_width=True
            )


        # ========================================================
        # TAB 6 — NATIONAL MAPS
        # ========================================================

        with future_tab6:

            st.subheader(
                "🗺️ Interactive U.S. Future DTR Maps"
            )

            st.info(
                """
                Each state polygon displays the metric calculated
                at that state's representative capital-location point.

                The coloring should not be interpreted as a spatial
                average across the entire state.
                """
            )


            map_metric = st.selectbox(
                "Select map variable",
                [
                    "Mean DTR (MVA)",
                    "Mean Temperature (°C)",
                    "Capacity Gain (%)",
                    "DTR < 22 MVA (%)",
                    "P10 DTR (MVA)",
                    "Summer Mean DTR (MVA)",
                    "Change in Mean DTR (MVA)",
                    "Change in Mean Temperature (°C)"
                ],

                key="future_map_metric"
            )


            # ----------------------------------------------------
            # NORMAL PERIOD METRICS
            # ----------------------------------------------------

            if map_metric in [
                "Mean DTR (MVA)",
                "Mean Temperature (°C)",
                "Capacity Gain (%)",
                "DTR < 22 MVA (%)",
                "P10 DTR (MVA)"
            ]:

                map_df = future_case_df[
                    (
                        future_case_df[
                            "Scenario_Code"
                        ]
                        ==
                        selected_scenario_code
                    )
                    &
                    (
                        future_case_df[
                            "Period"
                        ]
                        ==
                        future_period
                    )
                ].copy()


                metric_column_map = {
                    "Mean DTR (MVA)":
                        "Mean_DTR_MVA",

                    "Mean Temperature (°C)":
                        "Mean_Temperature_C",

                    "Capacity Gain (%)":
                        "Capacity_Gain_Percent",

                    "DTR < 22 MVA (%)":
                        "DTR_Below_22MVA_Percent",

                    "P10 DTR (MVA)":
                        "P10_DTR_MVA"
                }


                map_df["Map_Value"] = (
                    map_df[
                        metric_column_map[
                            map_metric
                        ]
                    ]
                )


            # ----------------------------------------------------
            # SUMMER MEAN DTR
            # ----------------------------------------------------

            elif (
                map_metric
                ==
                "Summer Mean DTR (MVA)"
            ):

                map_df = (
                    future_seasonal_df[
                        (
                            future_seasonal_df[
                                "Scenario_Code"
                            ]
                            ==
                            selected_scenario_code
                        )
                        &
                        (
                            future_seasonal_df[
                                "Period"
                            ]
                            ==
                            future_period
                        )
                        &
                        (
                            future_seasonal_df[
                                "Season"
                            ]
                            ==
                            "Summer"
                        )
                    ].copy()
                )


                map_df["Map_Value"] = (
                    map_df["Mean_DTR_MVA"]
                )


            # ----------------------------------------------------
            # PERIOD CHANGE MAPS
            # ----------------------------------------------------

            else:

                temp = future_case_df[
                    future_case_df[
                        "Scenario_Code"
                    ]
                    ==
                    selected_scenario_code
                ].copy()


                if (
                    map_metric
                    ==
                    "Change in Mean DTR (MVA)"
                ):

                    value_col = (
                        "Mean_DTR_MVA"
                    )

                else:

                    value_col = (
                        "Mean_Temperature_C"
                    )


                pivot = temp.pivot(
                    index=[
                        "State",
                        "Capital"
                    ],

                    columns="Period",

                    values=value_col
                ).reset_index()


                pivot["Map_Value"] = (
                    pivot["2051-2075"]
                    -
                    pivot["2026-2050"]
                )
                map_df = pivot


            # ----------------------------------------------------
            # STATE ABBREVIATIONS
            # ----------------------------------------------------

            map_df["State_Code"] = (
                map_df["State"]
                .map(STATE_TO_ABBR)
            )

            # ============================================================
            # MAP COLOR SETTINGS
            # ============================================================

            change_metrics = [
                "Change in Mean DTR (MVA)",
                "Change in Mean Temperature (°C)"
            ]

            if map_metric in change_metrics:

                map_fig = px.choropleth(
                    map_df,
                    locations="State_Code",
                    locationmode="USA-states",
                    color="Map_Value",
                    hover_name="State",
                    hover_data={
                        "State_Code": False,
                        "Map_Value": ":.2f"
                    },
                    scope="usa",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0
                )

            else:

                map_fig = px.choropleth(
                    map_df,
                    locations="State_Code",
                    locationmode="USA-states",
                    color="Map_Value",
                    hover_name="State",
                    hover_data={
                        "State_Code": False,
                        "Map_Value": ":.2f"
                    },
                    scope="usa",
                    color_continuous_scale="Viridis"
                )

            map_fig.update_layout(

                title=(
                    f"{map_metric} | "
                    f"{future_scenario} | "
                    f"{future_period}"
                ),

                coloraxis_colorbar=dict(
                    title=map_metric
                ),

                margin=dict(
                    l=0,
                    r=0,
                    t=70,
                    b=0
                )
            )

            # ============================================================
            # SELECTED STATE MAP KPI
            # ============================================================

            selected_map_row = map_df[
                map_df["State"] == state_full
                ]

            if not selected_map_row.empty:
                selected_map_value = (
                    selected_map_row["Map_Value"].iloc[0]
                )

                sorted_values = (
                    map_df["Map_Value"]
                    .sort_values()
                    .reset_index(drop=True)
                )

                position_from_lowest = (
                        sorted_values
                        .searchsorted(
                            selected_map_value,
                            side="left"
                        )
                        + 1
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Selected State",
                    state_full
                )

                col2.metric(
                    map_metric,
                    f"{selected_map_value:.2f}"
                )

                col3.metric(
                    "Position from Lowest Value",
                    f"{position_from_lowest} of {len(map_df)}"
                )

                st.caption(
                    "Position is based only on the selected metric "
                    "across the 50 representative capital-location cases."
                )

            st.plotly_chart(
                map_fig,
                use_container_width=True
            )


            # ----------------------------------------------------
            # STATE TABLE / RANKING
            # ----------------------------------------------------

            st.markdown(
                "### State Capital-Location Results"
            )


            ranking_table = (
                map_df[
                    [
                        "State",
                        "Capital",
                        "Map_Value"
                    ]
                ]
                .sort_values(
                    "Map_Value",
                    ascending=True
                )
                .copy()
            )


            ranking_table.columns = [
                "State",
                "Representative Capital",
                map_metric
            ]


            st.dataframe(
                ranking_table,
                use_container_width=True,
                hide_index=True
            )


        # ========================================================
        # TAB 7 — DISTRIBUTION / ECDF
        # ========================================================

        with future_tab7:

            st.subheader(
                f"📊 DTR Distribution — {state_full}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "P10 DTR",
                f"{selected_case['P10_DTR_MVA']:.2f} MVA"
            )

            col2.metric(
                "Median DTR",
                f"{selected_case['Median_DTR_MVA']:.2f} MVA"
            )

            col3.metric(
                "P90 DTR",
                f"{selected_case['P90_DTR_MVA']:.2f} MVA"
            )

            col4.metric(
                "DTR < 22 MVA",
                f"{selected_case['DTR_Below_22MVA_Percent']:.2f}%"
            )


            selected_hist = (
                state_histogram_df[
                    (
                        state_histogram_df[
                            "Scenario_Code"
                        ]
                        ==
                        selected_scenario_code
                    )
                    &
                    (
                        state_histogram_df[
                            "Period"
                        ]
                        ==
                        future_period
                    )
                ].copy()
            )

            # Keep only DTR bins that actually occurred
            selected_hist_active = selected_hist[
                selected_hist["Minute_Count"] > 0
                ].copy()

            selected_hist_active = (
                selected_hist_active
                .sort_values("Dynamic_MVA")
            )

            # Practical x-axis limits
            x_min = selected_hist_active["Dynamic_MVA"].min() - 0.5
            x_max = selected_hist_active["Dynamic_MVA"].max() + 0.5


            if not selected_hist.empty:


                # ------------------------------------------------
                # Probability Distribution
                # ------------------------------------------------

                fig_probability = go.Figure()


                fig_probability.add_trace(
                    go.Scatter(
                        x=selected_hist["Dynamic_MVA"],
                        y=selected_hist["Probability"] *100,
                        mode="lines",
                        fill="tozeroy",
                        name="DTR Probability"
                    )
                )


                fig_probability.add_vline(
                    x=22,

                    line_dash="dash",

                    line_color="black",

                    annotation_text=(
                        "22 MVA Nameplate"
                    )
                )

                fig_probability.add_vline(
                    x=selected_case["P10_DTR_MVA"],
                    line_dash="dot",
                    line_color="blue",
                    annotation_text=(
                        f"P10 = {selected_case['P10_DTR_MVA']:.2f}"
                    )
                )

                fig_probability.add_vline(
                    x=selected_case["Median_DTR_MVA"],
                    line_dash="dot",
                    line_color="green",
                    annotation_text=(
                        f"Median = {selected_case['Median_DTR_MVA']:.2f}"
                    )
                )

                fig_probability.add_vline(
                    x=selected_case["P90_DTR_MVA"],
                    line_dash="dot",
                    line_color="purple",
                    annotation_text=(
                        f"P90 = {selected_case['P90_DTR_MVA']:.2f}"
                    )
                )


                fig_probability.update_layout(
                    title=(
                        "DTR Probability Distribution"
                        f"<br><sup>{state_full} | "
                        f"{future_scenario} | "
                        f"{future_period}</sup>"
                    ),

                    xaxis_title=(
                        "Dynamic Transformer Rating "
                        "(MVA)"
                    ),

                    yaxis_title="Probability (%)"
                )

                fig_probability.update_xaxes(
                    range=[x_min, x_max]
                )

                fig_probability = style_future_plot(
                    fig_probability,
                    height=480,
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_probability,
                    use_container_width=True
                )

                # ------------------------------------------------
                # ECDF
                # ------------------------------------------------

                fig_ecdf = go.Figure()

                fig_ecdf.add_trace(
                    go.Scatter(
                        x=selected_hist["Dynamic_MVA"],
                        y=selected_hist["CDF"] *100,
                        mode="lines",
                        name="ECDF"
                    )
                )

                fig_ecdf.add_vline(
                    x=22,

                    line_dash="dash",

                    line_color="black",

                    annotation_text="22 MVA"
                )


                fig_ecdf.update_layout(
                    title=(
                        "Empirical Cumulative "
                        "Distribution of DTR"
                    ),

                    xaxis_title=(
                        "Dynamic Transformer Rating "
                        "(MVA)"
                    ),

                    yaxis_title=(
                        "Cumulative Probability (%)"
                    )
                )

                fig_ecdf.update_xaxes(
                    range=[x_min, x_max]
                )

                fig_ecdf.update_yaxes(
                    range=[0, 100]
                )

                fig_ecdf = style_future_plot(
                    fig_ecdf,
                    height=480,
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_ecdf,
                    use_container_width=True
                )
            else:

                st.warning(
                    "No histogram data were found "
                    "for this selection."
                )

        # ============================================================
        # TAB 8 — ADDITIONAL PUBLICATION / RESEARCH FIGURES
        # ============================================================
        with future_tab8:
            st.subheader("📑 Additional Research Figures")
            st.write(
                "Generate the supplementary manuscript visualizations directly "
                "from the validated **50-state CMIP6 results**. The existing "
                "Overview, National Comparison, Annual Trends, Seasonal "
                "Analysis, National Maps, and Distribution figures are not duplicated here."
            )
            st.caption(
                "All geographic observations represent individual state-capital "
                "locations, not statewide spatial averages. Figure labels follow "
                "the working manuscript and may be renumbered during editing."
            )

            research_figure_label = st.selectbox(
                "Select an additional manuscript figure",
                options=list(FUTURE_RESEARCH_FIGURE_OPTIONS.keys()),
                key="future_research_figure_choice",
            )
            research_figure_key = FUTURE_RESEARCH_FIGURE_OPTIONS[research_figure_label]

            if research_figure_key in (
                "temperature_dtr_scatter", "state_dtr_variation", "state_dtr_change"
            ):
                st.info(
                    f"Research figure filters: **{future_scenario}** / "
                    f"**{future_period}**. Adjust these in the Future Climate "
                    "Projection controls on the sidebar. For Fig. 10, all three "
                    "SSPs are shown within the chosen period; for Fig. 12, "
                    "both periods are compared within the selected SSP."
                )
            elif research_figure_key in (
                "temperature_spread", "dtr_spread", "summer_comparison"
            ):
                st.caption(
                    "This figure always compares all three SSPs and both periods "
                    "using the corresponding 50 capital-location datasets."
                )

            fig_research, caption_research, research_export_df, research_slug = (
                build_future_research_figure(
                    research_figure_key,
                    future_case_df,
                    future_seasonal_df,
                    future_mapping_df,
                    future_scenario,
                    future_period,
                    state_full,
                )
            )
            # Make all research figure text dark black
            fig_research = make_plot_text_black(fig_research)
            research_height = int(fig_research.layout.height or 650)
            research_chart_config = {
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "CMIP6_" + research_slug,
                    "width": 1800,
                    "height": research_height,
                    "scale": 2,
                },
            }
            st.plotly_chart(
                fig_research,
                use_container_width=True,
                config=research_chart_config,
            )
            st.caption(caption_research)
            st.caption(
                "Use the camera icon in the figure toolbar to export a "
                "high-resolution PNG directly from Plotly (no extra Python "
                "package required)."
            )
            st.download_button(
                "⬇️ Download plotted data (CSV)",
                data=research_export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="CMIP6_" + research_slug + ".csv",
                mime="text/csv",
                key="research_csv_" + research_slug,
            )

            with st.expander("Optional direct PNG and vector SVG downloads"):
                st.caption(
                    "Check this only when you need standalone publication files. "
                    "Static rendering requires Kaleido; the built-in Plotly "
                    "camera export above works without it."
                )
                if st.checkbox(
                    "Prepare publication PNG and SVG files",
                    value=False,
                    key="future_research_prepare_exports",
                ):
                    try:
                        export_png = fig_research.to_image(
                            format="png", width=1800,
                            height=research_height, scale=2,
                        )
                        export_svg = fig_research.to_image(
                            format="svg", width=1800,
                            height=research_height,
                        )
                        dl_png_col, dl_svg_col = st.columns(2)
                        with dl_png_col:
                            st.download_button(
                                "⬇️ Download high-resolution PNG",
                                data=export_png,
                                file_name="CMIP6_" + research_slug + ".png",
                                mime="image/png",
                                key="research_png_" + research_slug,
                            )
                        with dl_svg_col:
                            st.download_button(
                                "⬇️ Download vector SVG",
                                data=export_svg,
                                file_name="CMIP6_" + research_slug + ".svg",
                                mime="image/svg+xml",
                                key="research_svg_" + research_slug,
                            )
                    except Exception:
                        st.warning(
                            "Direct static export needs a working Kaleido/Chrome "
                            "installation. If necessary run `pip install -U kaleido` "
                            "and configure Chrome as required by your Kaleido "
                            "version. The interactive chart, camera PNG export, "
                            "and CSV download are still available."
                        )
            # ============================================================
            # FIGURE 13 — NATIONAL DTR PROBABILITY AND CDF
            # ============================================================

            st.markdown("---")

            with st.expander(
                "📊 Figure 13 — National DTR Probability and CDF",
                expanded=True
            ):

                st.markdown("### National DTR Distribution — 50 State Capitals")

                fig13_scenario = st.selectbox(
                    "Select climate scenario for Figure 13",
                    ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"],
                    index=2,
                    key="fig13_national_scenario"
                )

                # --------------------------------------------------------
                # 1. Extract national histogram data
                # --------------------------------------------------------

                fig13_hist = future_histogram_df[
                    future_histogram_df["Scenario"] == fig13_scenario
                ].copy()

                fig13_periods = ["2026-2050", "2051-2075"]

                fig13_colors = {
                    "2026-2050": "#1f77b4",
                    "2051-2075": "#ff7f0e"
                }

                # Average probability across 50 capital locations
                fig13_national = (
                    fig13_hist
                    .groupby(
                        ["Period", "Dynamic_MVA"],
                        as_index=False
                    )["Probability"]
                    .mean()
                    .sort_values(["Period", "Dynamic_MVA"])
                )

                # Calculate the national cumulative distribution
                fig13_national["National_CDF"] = (
                    fig13_national
                    .groupby("Period")["Probability"]
                    .cumsum()
                )

                # --------------------------------------------------------
                # 2. Determine useful horizontal plotting range
                # --------------------------------------------------------

                active_ratings = fig13_national.loc[
                    fig13_national["Probability"] > 0,
                    "Dynamic_MVA"
                ]

                x_min = float(active_ratings.min()) - 0.4
                x_max = float(active_ratings.max()) + 0.4

                # --------------------------------------------------------
                # 3. Create two vertically arranged plots
                # --------------------------------------------------------

                fig13 = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.09,
                    row_heights=[0.42, 0.58]
                )

                fig13_peak = 0

                below_nameplate = {}

                for period in fig13_periods:

                    # Full distribution for this period
                    period_full = fig13_national[
                        fig13_national["Period"] == period
                    ].copy()

                    # Calculate probability strictly below 22 MVA
                    below_nameplate[period] = (
                        period_full.loc[
                            period_full["Dynamic_MVA"] < 22,
                            "Probability"
                        ].sum() * 100
                    )

                    # Restrict the visual range, not the calculations
                    period_data = period_full[
                        period_full["Dynamic_MVA"].between(
                            x_min, x_max
                        )
                    ].copy()

                    x = period_data["Dynamic_MVA"]

                    probability = (
                        period_data["Probability"] * 100
                    )

                    cumulative = (
                        period_data["National_CDF"] * 100
                    )

                    fig13_peak = max(
                        fig13_peak,
                        float(probability.max())
                    )

                    color = fig13_colors[period]

                    # Upper graph: probability distribution
                    fig13.add_trace(
                        go.Scatter(
                            x=x,
                            y=probability,
                            mode="lines",
                            name=period,
                            line=dict(
                                color=color,
                                width=2.5,
                                shape="hv"
                            ),
                            hovertemplate=(
                                "DTR: %{x:.2f} MVA"
                                "<br>Probability: %{y:.3f}%"
                                "<extra>%{fullData.name}</extra>"
                            )
                        ),
                        row=1,
                        col=1
                    )

                    # Lower graph: cumulative distribution
                    fig13.add_trace(
                        go.Scatter(
                            x=x,
                            y=cumulative,
                            mode="lines",
                            name=period,
                            showlegend=False,
                            line=dict(
                                color=color,
                                width=2.5,
                                shape="hv"
                            ),
                            hovertemplate=(
                                "DTR: %{x:.2f} MVA"
                                "<br>Cumulative: %{y:.2f}%"
                                "<extra>%{fullData.name}</extra>"
                            )
                        ),
                        row=2,
                        col=1
                    )

                # --------------------------------------------------------
                # 4. Add the nameplate reference
                # --------------------------------------------------------

                for plot_row in [1, 2]:

                    fig13.add_vline(
                        x=22,
                        line_color="black",
                        line_dash="dash",
                        line_width=1.5,
                        row=plot_row,
                        col=1
                    )

                # --------------------------------------------------------
                # 5. Publication formatting
                # --------------------------------------------------------

                fig13.update_layout(
                    title=dict(
                        text=(
                            "National DTR Distributions | "
                            + fig13_scenario
                        ),
                        x=0.01,
                        font=dict(
                            color="black",
                            size=18
                        )
                    ),

                    template="plotly_white",

                    height=730,

                    font=dict(
                        color="black",
                        size=13
                    ),

                    legend=dict(
                        orientation="h",
                        x=0.98,
                        xanchor="right",
                        y=0.98,
                        yanchor="top",
                        font=dict(
                            color="black",
                            size=12
                        ),
                        bgcolor="rgba(255,255,255,0.85)"
                    ),

                    margin=dict(
                        l=75,
                        r=25,
                        t=75,
                        b=65
                    )
                )

                fig13.update_yaxes(
                    title_text="Probability (%)",
                    range=[0, fig13_peak * 1.18],
                    row=1,
                    col=1
                )

                fig13.update_yaxes(
                    title_text="Cumulative probability (%)",
                    range=[0, 102],
                    row=2,
                    col=1
                )

                fig13.update_xaxes(
                    title_text="Discrete dynamic rating (MVA)",
                    range=[x_min, x_max],
                    row=2,
                    col=1
                )

                # Dark black labels throughout
                fig13.update_xaxes(
                    title_font=dict(color="black", size=15),
                    tickfont=dict(color="black", size=12),
                    showline=True,
                    linecolor="black"
                )

                fig13.update_yaxes(
                    title_font=dict(color="black", size=15),
                    tickfont=dict(color="black", size=12),
                    showline=True,
                    linecolor="black"
                )

                # --------------------------------------------------------
                # 6. Display the national figure
                # --------------------------------------------------------

                st.plotly_chart(
                    fig13,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": "CMIP6_Figure13_National_DTR",
                            "width": 1800,
                            "height": 1300,
                            "scale": 2
                        }
                    }
                )

                # --------------------------------------------------------
                # 7. Display results below the figure
                # --------------------------------------------------------

                st.markdown(
                    "#### Modeled Time with DTR Below 22 MVA"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "2026–2050",
                    f"{below_nameplate['2026-2050']:.2f}%"
                )

                col2.metric(
                    "2051–2075",
                    f"{below_nameplate['2051-2075']:.2f}%"
                )

                st.caption(
                    "National distribution calculated as the arithmetic "
                    "mean of the 50 capital-location probability distributions. "
                    "The dashed line indicates the 22 MVA nameplate rating. "
                    "The reported below-nameplate percentages use DTR < 22 MVA."
                )

                # --------------------------------------------------------
                # 8. Download plotted numerical data
                # --------------------------------------------------------

                st.download_button(
                    "⬇️ Download Figure 13 National Data (CSV)",
                    data=fig13_national.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="CMIP6_Figure13_National_DTR.csv",
                    mime="text/csv",
                    key="fig13_download"
                )


        # ========================================================
        # IMPORTANT:
        # STOP FUTURE MODE BEFORE NASA POWER
        # ========================================================

        st.stop()


    # ============================================================
    # NORMAL DTR / DCR WORKFLOW
    # ============================================================

    if start_date >= end_date:

        st.error(
            "Start date must be before end date."
        )

        st.stop()


    start_str = start_date.strftime(
        "%Y%m%d"
    )

    end_str = end_date.strftime(
        "%Y%m%d"
    )

    with st.spinner("Fetching weather data from NASA POWER…"):
        try:
            temp_df = fetch_nasa_power(lat, lon, start_str, end_str)
        except Exception as e:
            st.error(f"NASA POWER fetch failed: {e}")
            st.stop()

    st.success(f"Downloaded {len(temp_df):,} hourly records.")

    # ───────────────────────────────
    # 🗺️ US SHAPE MAP
    # ───────────────────────────────

    st.markdown("## 🗺️ Selected Location on US Map")

    state_full = st.session_state.get(
        "state"
    )

    if state_full:
        fig_map = plot_us_highlight_map(state_full)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("State could not be determined for map highlighting.")

    # ── Time axis fix ──
    if temp_df['HR'].iloc[-1] <= 24:
        temp_df['HR_global'] = np.arange(len(temp_df))
    else:
        temp_df['HR_global'] = temp_df['HR']

    time_hr = temp_df['HR_global'].values
    T_ambient_raw = temp_df['T2M'].values
    T_ambient = T_ambient_raw + temp_margin

    # ── Interpolate to minutes ──
    with st.spinner("Interpolating to minute resolution…"):
        total_minutes = int((time_hr.max() - time_hr.min()) * 60) + 60
        time_min = np.arange(0, total_minutes)

        T_amb_profile = interp1d(
            time_hr * 60, T_ambient,
            kind='linear', fill_value='extrapolate', bounds_error=False
        )
        T_amb_min = T_amb_profile(time_min)
        # Use fixed wind speed instead of NASA wind data
        wind_min = np.full_like(time_min, fixed_wind_speed, dtype=float)

    # ============================================================
    # MODE 1: DYNAMIC TRANSFORMER RATING
    # ============================================================
    if calc_mode == "Dynamic Transformer Rating":

        if transformer_params is None:
            st.error("Transformer parameters are not available. Please provide valid transformer input.")
            st.stop()

        with st.spinner("Calculating DTR..."):
            L_dyn = compute_dtr_vectorized(T_amb_min, transformer_params, hs_limit)
            MVA_dyn = L_dyn * transformer_params['MVA_rated']

        results = pd.DataFrame({
            'Time_min': time_min,
            'Time_hr': time_min / 60,
            'Time_day': time_min / 1440,
            'Ambient_C': T_amb_min,
            'PU_Rating': L_dyn,
            'Dynamic_MVA': MVA_dyn,
        })

        results['Day'] = (results['Time_min'] // 1440).astype(int)
        results['Month'] = ((results['Time_day'] // 30.44) + 1).astype(int).clip(1, 12)
        results['Season'] = results['Month'].apply(get_season)
        results['month'] = results['Month']
#===========================================
# Hourly DTR
#===========================================
        hourly_dtr = (
            results.groupby(results['Time_hr'].astype(int))
            .agg({
                'Ambient_C': 'mean',
                'PU_Rating': 'mean',
                'Dynamic_MVA': 'mean'
            })
            .reset_index()
        )

        hourly_dtr.rename(columns={
            'Time_hr': 'Hour',
            'Ambient_C': 'Avg_Ambient_C',
            'PU_Rating': 'Avg_PU_Rating',
            'Dynamic_MVA': 'Avg_DTR_MVA'
        }, inplace=True)

        # Create actual timestamps

        hourly_dtr['Datetime'] = pd.date_range(
            start=pd.Timestamp(start_date),
            periods=len(hourly_dtr),
            freq='h'
        )

        hourly_dtr = hourly_dtr[
            [
                'Datetime',
                'Hour',
                'Avg_Ambient_C',
                'Avg_PU_Rating',
                'Avg_DTR_MVA'
            ]
        ]

        # Daily stats
        daily_stats = results.groupby('Day')['Dynamic_MVA'].agg(['min', 'max'])
        daily_pu = results.groupby('Day')['PU_Rating'].agg(['min', 'max'])
        x_day = daily_stats.index

        # Seasonal stats
        seasonal_summary = results.groupby('Season').agg(
            Min_Temperature_C=('Ambient_C', 'min'),
            Max_Temperature_C=('Ambient_C', 'max'),
            Mean_Temperature_C=('Ambient_C', 'mean'),
            Min_DTR_MVA=('Dynamic_MVA', 'min'),
            Max_DTR_MVA=('Dynamic_MVA', 'max'),
            Mean_DTR_MVA=('Dynamic_MVA', 'mean'),
        ).round(2)

        seasonal_dtr_stats = compute_seasonal_stats(results, 'Dynamic_MVA', 'MVA')
        seasonal_temp_stats = compute_seasonal_stats(results, 'Ambient_C', '°C')

        st.success("✅ Transformer DTR calculation complete!")
        st.markdown("---")

        rated_mva = transformer_params['MVA_rated']
        min_dtr = results['Dynamic_MVA'].min()
        max_dtr = results['Dynamic_MVA'].max()
        avg_dtr = results['Dynamic_MVA'].mean()
        capacity_gain_percent = ((avg_dtr - rated_mva) / rated_mva) * 100
        max_overload_pu = max_dtr / rated_mva
        utilization_factor = avg_dtr / rated_mva
        high_temp_hours = (results['Ambient_C'] > 30).sum() / 60
        worst_season = seasonal_summary['Min_DTR_MVA'].idxmin()
        worst_season_capacity = seasonal_summary['Min_DTR_MVA'].min()
        percent_above_nameplate = ((results['Dynamic_MVA'] > rated_mva).sum() / len(results) * 100)
        thermal_limit_hours = ((results['PU_Rating'] >= 1.2).sum() / 60)
        critical_low_capacity_hours = ((results['Dynamic_MVA'] < rated_mva * 0.9).sum() / 60)
        capacity_headroom_avg = (results['Dynamic_MVA'] - rated_mva).mean()

        if percent_above_nameplate > 30:
            risk_flag = "🔴 HIGH FLEXIBILITY (Frequent overload margin)"
        elif percent_above_nameplate > 10:
            risk_flag = "🟡 MODERATE FLEXIBILITY"
        else:
            risk_flag = "🟢 LIMITED OVERLOAD MARGIN"

        st.write(hourly_dtr.head())

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview",
            "📅 Daily Statistics",
            "🍂 Seasonal Analysis",
            "📊 Statistical Analysis",
            "⏰ Hourly DTR",
        ])

        with tab1:
            st.markdown("## 🏢 Executive Operational Summary")

            col1, col2, col3= st.columns(3)
            col1.metric("Nameplate Rating", f"{rated_mva:.1f} MVA")
            col2.metric("Minimum DTR (MVA)", f"{min_dtr:.1f} MVA")
            col3.metric("Maximum DTR (MVA)", f"{max_dtr:.1f} MVA")

            col4, col5 = st.columns(2)
            col4.metric("Average DTR (MVA)", f"{avg_dtr:.1f} MVA")
            col5.metric("Average Capacity Gain", f"{capacity_gain_percent:.1f}%")

            st.subheader("Transformer Dynamic Capacity Indicator")

            gauge_value = avg_dtr
            max_scale = rated_mva * 1.5

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gauge_value,
                number={'suffix': " MVA"},
                delta={'reference': rated_mva, 'increasing': {'color': "green"}},
                title={'text': "Average Dynamic Transformer Capacity"},
                gauge={
                    'axis': {'range': [0, max_scale]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, rated_mva * 0.8], 'color': "#d6eaf8"},
                        {'range': [rated_mva * 0.8, rated_mva], 'color': "#aed6f1"},
                        {'range': [rated_mva, rated_mva * 1.2], 'color': "#abebc6"},
                        {'range': [rated_mva * 1.2, max_scale], 'color': "#f9e79f"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': rated_mva
                    }
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("Dynamic Transformer Rating (Full Period)")
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(
                go.Scatter(x=results['Time_day'], y=results['Dynamic_MVA'],
                           name='Dynamic Rating (MVA)', line=dict(color='red', width=1.5)),
                secondary_y=False,
            )
            fig1.add_trace(
                go.Scatter(x=results['Time_day'], y=results['PU_Rating'],
                           name='Per Unit Rating', line=dict(color='blue', width=1.5, dash='dash')),
                secondary_y=True,
            )
            fig1.update_xaxes(title_text="Time (days)")
            fig1.update_yaxes(title_text="Dynamic Rating (MVA)", secondary_y=False)
            fig1.update_yaxes(title_text="Per Unit Rating", secondary_y=True)
            fig1.update_layout(title="Dynamic Transformer Rating (IEEE C57.91)",
                               legend=dict(x=0, y=1), hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Ambient Temperature Profile")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=results['Time_day'], y=results['Ambient_C'],
                name='Ambient Temp (°C)', line=dict(color='darkorange', width=1.2)
            ))
            fig2.update_layout(
                title="Ambient Temperature Profile",
                xaxis_title="Time (days)",
                yaxis_title="Ambient Temperature (°C)",
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)


        with tab2:
            st.subheader("Daily Min & Max Dynamic Transformer Rating")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=x_day, y=daily_stats['min'],
                mode='lines+markers', marker_symbol='circle', marker_size=4,
                name='Daily Min DTR (MVA)', line=dict(color='red', width=2)
            ))
            fig3.add_trace(go.Scatter(
                x=x_day, y=daily_stats['max'],
                mode='lines+markers', marker_symbol='square', marker_size=4,
                name='Daily Max DTR (MVA)', line=dict(color='green', width=2)
            ))
            fig3.update_layout(
                title="Daily Minimum and Maximum Dynamic Transformer Rating",
                xaxis_title="Day", yaxis_title="Dynamic Transformer Rating (MVA)",
                hovermode="x unified",
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Daily DTR vs Nameplate Rating")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=x_day, y=daily_stats['min'],
                name='Daily Min DTR (MVA)', line=dict(color='red', width=2)
            ))
            fig4.add_trace(go.Scatter(
                x=x_day, y=daily_stats['max'],
                name='Daily Max DTR (MVA)', line=dict(color='green', width=2)
            ))
            fig4.add_hline(
                y=transformer_params['MVA_rated'],
                line_dash="dash", line_color="black", line_width=1.5,
                annotation_text=f"Nameplate {transformer_params['MVA_rated']} MVA (Static)",
                annotation_position="bottom right",
            )
            fig4.update_layout(
                title="Daily Min & Max DTR vs Static Nameplate Rating",
                xaxis_title="Day", yaxis_title="Rating (MVA)",
                hovermode="x unified",
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader("Daily Per Unit Rating")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=x_day, y=daily_pu['min'],
                name='Daily Min PU', line=dict(color='blue', width=2, dash='dash')
            ))
            fig5.add_trace(go.Scatter(
                x=x_day, y=daily_pu['max'],
                name='Daily Max PU', line=dict(color='purple', width=2, dash='dash')
            ))
            fig5.update_layout(
                title="Daily Min & Max Per Unit Transformer Rating",
                xaxis_title="Day", yaxis_title="Per Unit Rating",
                hovermode="x unified",
            )
            st.plotly_chart(fig5, use_container_width=True)

        with tab3:
            seasons = ['Winter', 'Spring', 'Summer', 'Fall']

            st.subheader("Seasonal Ambient Temperature")
            cols_temp = st.columns(2)
            for idx, season in enumerate(seasons):
                season_data = results[results['Season'] == season]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=season_data['Time_day'], y=season_data['Ambient_C'],
                    name=f'{season} Temp', line=dict(color=SEASON_COLOR[season], width=1.5)
                ))
                fig.update_layout(
                    title=f"Ambient Temperature – {season}",
                    xaxis_title="Time (days)", yaxis_title="Ambient Temperature (°C)",
                    showlegend=False,
                )
                cols_temp[idx % 2].plotly_chart(fig, use_container_width=True)

            st.subheader("Seasonal Dynamic Transformer Rating")
            cols_dtr = st.columns(2)
            for idx, season in enumerate(seasons):
                season_data = results[results['Season'] == season]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=season_data['Time_day'], y=season_data['Dynamic_MVA'],
                    name=f'{season} DTR', line=dict(color=SEASON_COLOR[season], width=1.5)
                ))
                fig.update_layout(
                    title=f"Dynamic Transformer Rating – {season}",
                    xaxis_title="Time (days)", yaxis_title="Dynamic Transformer Rating (MVA)",
                    showlegend=False,
                )
                cols_dtr[idx % 2].plotly_chart(fig, use_container_width=True)

            st.subheader("Seasonal Summary Statistics")
            st.dataframe(seasonal_summary.style.format("{:.2f}"), use_container_width=True)

        with tab4:
            st.subheader("Probability Density of DTR")
            kde_dtr = gaussian_kde(results['Dynamic_MVA'].dropna())
            x_dtr = np.linspace(results['Dynamic_MVA'].min(), results['Dynamic_MVA'].max(), 400)
            fig15 = go.Figure()
            fig15.add_trace(go.Scatter(
                x=x_dtr, y=kde_dtr(x_dtr), fill='tozeroy',
                name='PDF of DTR', line=dict(color='steelblue', width=2)
            ))
            fig15.update_layout(
                title="PDF of Dynamic Transformer Rating",
                xaxis_title="Dynamic Transformer Rating (MVA)",
                yaxis_title="Density",
            )
            st.plotly_chart(fig15, use_container_width=True)

            st.subheader("Probability Density of Ambient Temperature")
            kde_temp = gaussian_kde(results['Ambient_C'].dropna())
            x_temp = np.linspace(results['Ambient_C'].min(), results['Ambient_C'].max(), 400)
            fig16 = go.Figure()
            fig16.add_trace(go.Scatter(
                x=x_temp, y=kde_temp(x_temp), fill='tozeroy',
                name='PDF of Ambient Temp', line=dict(color='orange', width=2)
            ))
            fig16.update_layout(
                title="PDF of Ambient Temperature",
                xaxis_title="Ambient Temperature (°C)",
                yaxis_title="Density",
            )
            st.plotly_chart(fig16, use_container_width=True)

            # ── Helper: seasonal box plot with overlays ──
            def seasonal_boxplot(data_col, y_label, title):
                percentiles = [10, 30, 60, 90]
                fig = go.Figure()
                for season in seasons:
                    vals = results[results['Season'] == season][data_col].dropna()
                    fig.add_trace(go.Box(
                        y=vals, name=season,
                        marker_color=SEASON_COLOR[season],
                        boxpoints=False,
                        whiskerwidth=0.5,
                    ))

                # Overlay scatter points for percentiles, mean, min, max
                for s_idx, season in enumerate(seasons):
                    vals = results[results['Season'] == season][data_col].dropna()
                    perc_vals = np.percentile(vals, percentiles)
                    mean_val = np.mean(vals)
                    min_val = np.min(vals)
                    max_val = np.max(vals)
                    std_val = np.std(vals)

                    # Percentile markers
                    for p, v, c in zip(percentiles, perc_vals, PERCENTILE_COLORS):
                        fig.add_trace(go.Scatter(
                            x=[season], y=[v],
                            mode='markers+text',
                            marker=dict(color=c, size=8, symbol='circle'),
                            text=[f'{p}%'],
                            textposition='middle right',
                            name=f'{p}th % ({season})',
                            showlegend=False,
                        ))

                    # Mean diamond
                    fig.add_trace(go.Scatter(
                        x=[season], y=[mean_val],
                        mode='markers+text',
                        marker=dict(color='black', size=10, symbol='diamond'),
                        text=['Mean'], textposition='middle right',
                        name='Mean' if s_idx == 0 else '',
                        showlegend=(s_idx == 0),
                    ))

                    # Min / Max triangles
                    fig.add_trace(go.Scatter(
                        x=[season], y=[min_val],
                        mode='markers',
                        marker=dict(color='grey', size=8, symbol='triangle-down'),
                        name='Min' if s_idx == 0 else '',
                        showlegend=(s_idx == 0),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[season], y=[max_val],
                        mode='markers',
                        marker=dict(color='magenta', size=8, symbol='triangle-up'),
                        name='Max' if s_idx == 0 else '',
                        showlegend=(s_idx == 0),
                    ))

                    # Std Dev error bar (as separate scatter)
                    fig.add_trace(go.Scatter(
                        x=[season], y=[mean_val],
                        error_y=dict(type='data', array=[std_val], visible=True,
                                     color='red', thickness=2, width=6),
                        mode='lines',
                        marker=dict(color='red', size=14, symbol='line-ns', opacity=1),
                        name='Std Dev' if s_idx == 0 else '',
                        showlegend=(s_idx == 0),
                    ))

                fig.update_layout(
                    title=title,
                    yaxis_title=y_label,
                    hovermode="closest",
                    legend=dict(x=1.01, y=1),
                )
                return fig


            # Plot 17: Seasonal DTR box plot
            st.subheader("Seasonal DTR: Box Plot with Percentiles")
            fig17 = seasonal_boxplot('Dynamic_MVA', 'Dynamic Transformer Rating (MVA)',
                                     'Seasonal DTR')
            fig17.update_layout(
                xaxis_title="Seasons",
                xaxis_title_font = dict(size=14, color="black"),
                yaxis_title_font = dict(size=14, color="black")
            )
            st.plotly_chart(fig17, use_container_width=True)

            # Plot 18: Seasonal Temperature box plot
            st.subheader("Seasonal Temperature: Box Plot with Percentiles")
            fig18 = seasonal_boxplot('Ambient_C', 'Ambient Temperature (°C)',
                                     'Seasonal Temperature')
            fig18.update_layout(
                xaxis_title="Seasons",
                xaxis_title_font = dict(size=14,color="black"),
                yaxis_title_font = dict(size=14,color="black")
            )
            st.plotly_chart(fig18, use_container_width=True)

            # Plot 19: DTR statistics table + download
            st.subheader("Seasonal DTR Statistics Table")
            st.dataframe(seasonal_dtr_stats.style.format(
                {c: "{:.3f}" for c in seasonal_dtr_stats.columns if c != 'Season'}
            ), use_container_width=True)
            st.download_button(
                "⬇️ Download DTR Statistics CSV",
                data=seasonal_dtr_stats.to_csv(index=False),
                file_name="seasonal_DTR_statistics.csv",
                mime="text/csv",
            )

            # Plot 20: Temperature statistics table + download
            st.subheader("Seasonal Temperature Statistics Table")
            st.dataframe(seasonal_temp_stats.style.format(
                {c: "{:.3f}" for c in seasonal_temp_stats.columns if c != 'Season'}
            ), use_container_width=True)
            st.download_button(
                "⬇️ Download Temperature Statistics CSV",
                data=seasonal_temp_stats.to_csv(index=False),
                file_name="seasonal_temperature_statistics.csv",
                mime="text/csv",
            )

            # ─────────────────────────────────────────────────────
            # Dynamic Transformer Rating Percentiles
            # ─────────────────────────────────────────────────────

            st.subheader("Dynamic Transformer Rating Percentiles")

            p10 = np.percentile(results['Dynamic_MVA'], 10)
            p50 = np.percentile(results['Dynamic_MVA'], 50)
            p90 = np.percentile(results['Dynamic_MVA'], 90)

            col1, col2, col3 = st.columns(3)

            col1.metric("10th Percentile Rating", f"{p10:.2f} MVA")
            col2.metric("Median Dynamic Rating", f"{p50:.2f} MVA")
            col3.metric("90th Percentile Rating", f"{p90:.2f} MVA")

        with tab5:

            st.subheader("Hourly Dynamic Transformer Rating")

            fig_hourly = go.Figure()

            fig_hourly.add_trace(
                go.Scatter(
                    # x=hourly_dtr['Hour'],
                    x=hourly_dtr['Datetime'],
                    y=hourly_dtr['Avg_DTR_MVA'],
                    mode='lines',
                    name='Hourly DTR',
                    line=dict(color='red')
                )
            )
            fig_hourly.add_hline(
                y=transformer_params['MVA_rated'],
                line_dash="dash",
                line_color="black",
                annotation_text="Nameplate Rating"
            )

            fig_hourly.update_layout(
                title="Hourly Dynamic Transformer Rating",
                xaxis_title="Date and Time",
                yaxis_title="DTR (MVA)",
                hovermode="x unified"
            )

            st.plotly_chart(fig_hourly, use_container_width=True)

            st.dataframe(hourly_dtr, use_container_width=True)

            st.download_button(
                "⬇️ Download Hourly DTR CSV",
                data=hourly_dtr.to_csv(index=False),
                file_name="Hourly_DTR.csv",
                mime="text/csv"
            )
    # ============================================================
    # MODE 2: DYNAMIC CONDUCTOR RATING
    # ============================================================
    elif calc_mode == "Dynamic Conductor Rating":

        if conductor_params is None:
            st.error("Conductor parameters are not available. Please provide valid conductor input.")
            st.stop()

        with st.spinner("Calculating Conductor DCR..."):
            I_wind = compute_conductor_dcr(T_amb_min, wind_min, conductor_params)

        results = pd.DataFrame({
            'Time_min': time_min,
            'Time_hr': time_min / 60,
            'Time_day': time_min / 1440,
            'Ambient_C': T_amb_min,
            'Wind_mps': wind_min,
            'DCR': I_wind,
        })

        results['Day'] = (results['Time_min'] // 1440).astype(int)
        results['Month'] = ((results['Time_day'] // 30.44) + 1).astype(int).clip(1, 12)
        results['Season'] = results['Month'].apply(get_season)
        results['month'] = results['Month']

        # -----------------------------
        # Seasonal Static Ampacity
        # Summer = May (5) to October (10)
        # Winter = November (11) to April (4)
        # -----------------------------
        results['Static_Ampacity'] = np.where(
            results['Month'].between(5, 10),
            conductor_params['static_amp_summer'],
            conductor_params['static_amp_winter']
        )

        monthly_dcr_stats = compute_monthly_stats(results, 'DCR', 'A')
        monthly_temp_stats = compute_monthly_stats(results, 'Ambient_C', '°C')

        # -----------------------------
        # Seasonal Statistics (NEW)
        # -----------------------------
        seasonal_dcr_stats = compute_seasonal_stats(results, 'DCR', 'A')
        seasonal_temp_stats = compute_seasonal_stats(results, 'Ambient_C', '°C')

        seasonal_summary = results.groupby('Season').agg(
            Min_Temperature_C=('Ambient_C', 'min'),
            Max_Temperature_C=('Ambient_C', 'max'),
            Mean_Temperature_C=('Ambient_C', 'mean'),
            Min_DCR=('DCR', 'min'),
            Max_DCR=('DCR', 'max'),
            Mean_DCR=('DCR', 'mean'),
        ).round(2)

        st.success("✅ Conductor DCR calculation complete!")
        st.markdown("---")

        max_dcr = np.nanmax(results['DCR'])
        min_dcr = np.nanmin(results['DCR'])
        avg_dcr = np.nanmean(results['DCR'])


        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview",
            "📊 Distribution",
            "📅 Monthly Analysis",
            "🍂 Seasonal Analysis"
        ])

        with tab1:

            st.subheader("Conductor KPIs")
            col1, col2, col3 = st.columns(3)
            col1.metric("Max DCR", f"{max_dcr:.1f} A")
            col2.metric("Min DCR", f"{min_dcr:.1f} A")
            col3.metric("Avg DCR", f"{avg_dcr:.1f} A")

            col4, col5= st.columns(2)
            col4.metric("Static Ampacity (Summer)", f"{conductor_params['static_amp_summer']:.1f} A")
            col5.metric("Static Ampacity (Winter)", f"{conductor_params['static_amp_winter']:.1f} A")

            st.subheader("Dynamic Conductor Rating")
            st.caption(f"Selected Conductor: {conductor_params.get('conductor_type', 'Unknown')}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['Time_day'],
                y=results['DCR'],
                name='Dynamic conductor Rating',
                line=dict(color='blue')
            ))
            # Summer static line (only where month is May–Oct)
            summer_static = np.where(
                results['Month'].between(5, 10),
                conductor_params['static_amp_summer'],
                np.nan
            )

            # Winter static line (only where month is Nov–Apr)
            winter_static = np.where(
                ~results['Month'].between(5, 10),
                conductor_params['static_amp_winter'],
                np.nan
            )

            fig.add_trace(go.Scatter(
                x=results['Time_day'],
                y=summer_static,
                name='Static Ampacity (Summer)',
                line=dict(color='red', dash='dash', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=results['Time_day'],
                y=winter_static,
                name='Static Ampacity (Winter)',
                line=dict(color='green', dash='dot', width=2)
            ))

            fig.update_layout(
                title="Dynamic Conductor Rating",
                xaxis_title="Time (days)",
                yaxis_title="Ampacity (Amps)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Probability Density Function of Dynamic Conductor Rating")

            # -----------------------------
            # PDF for Dynamic Rating
            # -----------------------------
            dcr_wind = results['DCR'].dropna()

            if len(dcr_wind) > 1:

                # KDE for Dynamic Rating
                kde_wind = gaussian_kde(dcr_wind)
                x_wind = np.linspace(dcr_wind.min(), dcr_wind.max(), 400)

                fig_pdf_wind = go.Figure()
                fig_pdf_wind.add_trace(go.Scatter(
                    x=x_wind,
                    y=kde_wind(x_wind),
                    fill='tozeroy',
                    name='Dynamic Rating',
                    line=dict(color='blue', width=2)
                ))
                fig_pdf_wind.add_vline(
                    x=conductor_params['static_amp_summer'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Static Ampacity (Summer)",
                    annotation_position="top right"
                )

                fig_pdf_wind.add_vline(
                    x=conductor_params['static_amp_winter'],
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Static Ampacity (Winter)",
                    annotation_position="top right"
                )

                # Dummy traces for legend
                fig_pdf_wind.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    name='Static Ampacity (Summer)',
                    line=dict(color='red', dash='dash')
                ))

                fig_pdf_wind.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    name='Static Ampacity (Winter)',
                    line=dict(color='green', dash='dot')
                ))


                fig_pdf_wind.update_layout(
                    title="PDF of Dynamic Conductor Rating",
                    xaxis_title="Ampacity (Amps)",
                    yaxis_title="Density",
                    template="plotly_white"
                )
                st.plotly_chart(fig_pdf_wind, use_container_width=True)

            else:
                st.warning("Not enough valid DCR data points to compute probability density functions.")


        with tab3:
            st.subheader("Monthly Dynamic Conductor Rating Analysis")

            fig_box_simple = monthly_boxplot_simple(
                results,
                static_amp_summer=conductor_params['static_amp_summer'],
                static_amp_winter=conductor_params['static_amp_winter']
            )
            st.plotly_chart(fig_box_simple, use_container_width=True)

            fig_box_wind = monthly_boxplot_with_stats(
                results,
                data_col='DCR',
                y_label='Ampacity (Amps)',
                title='Monthly Dynamic Conductor Rating \nPercentiles, Mean, Min, Max, Std Dev',
                static_amp_summer = conductor_params['static_amp_summer'],
                static_amp_winter = conductor_params['static_amp_winter']
            )
            st.plotly_chart(fig_box_wind, use_container_width=True)

            st.markdown("---")
            st.subheader("Monthly Ambient Temperature Analysis")

            # -----------------------------
            # Simple Monthly Temperature Box Plot
            # -----------------------------
            fig_temp_box = go.Figure()

            fig_temp_box.add_trace(go.Box(
                x=results['month'].astype(str),
                y=results['Ambient_C'],
                name='Ambient Temperature',
                marker_color='orange',
                boxpoints=False,
                line=dict(width=2),
                width=0.45
            ))

            fig_temp_box.update_layout(
                title="Monthly Ambient Temperature",
                xaxis_title="Month",
                yaxis_title="Ambient Temperature (°C)",
                boxmode='group',
                hovermode="closest",
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01
                )
            )

            fig_temp_box.update_xaxes(
                type='category',
                categoryorder='array',
                categoryarray=[str(i) for i in range(1, 13)]
            )

            st.plotly_chart(fig_temp_box, use_container_width=True)

            # -----------------------------
            # Detailed Monthly Temperature Plot
            # -----------------------------
            fig_temp_stats = monthly_boxplot_with_stats(
                results,
                data_col='Ambient_C',
                y_label='Ambient Temperature (°C)',
                title='Monthly Ambient Temperature \nPercentiles, Mean, Min, Max, Std Dev'
            )

            st.plotly_chart(fig_temp_stats, use_container_width=True)

            st.subheader("Monthly DCR Statistics Table")
            st.dataframe(
                monthly_dcr_stats.style.format(
                    {c: "{:.3f}" for c in monthly_dcr_stats.columns if c != 'Month'}
                ),
                use_container_width=True
            )
            st.download_button(
                "⬇️ Download Monthly DCR Statistics CSV",
                data=monthly_dcr_stats.to_csv(index=False),
                file_name="monthly_DCR_statistics.csv",
                mime="text/csv",
            )

            st.subheader("Monthly Ambient Temperature Statistics Table")
            st.dataframe(
                monthly_temp_stats.style.format(
                    {c: "{:.3f}" for c in monthly_temp_stats.columns if c != 'Month'}
                ),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Monthly Temperature Statistics CSV",
                data=monthly_temp_stats.to_csv(index=False),
                file_name="monthly_temperature_statistics.csv",
                mime="text/csv",
            )

            st.subheader("Monthly Percentile Summary")

            p10_w = np.nanpercentile(results['DCR'].dropna(), 10)
            p50_w = np.nanpercentile(results['DCR'].dropna(), 50)
            p90_w = np.nanpercentile(results['DCR'].dropna(), 90)

            col1, col2, col3 = st.columns(3)
            col1.metric("10th Percentile (DCR)", f"{p10_w:.2f} A")
            col2.metric("Median (DCR)", f"{p50_w:.2f} A")
            col3.metric("90th Percentile (DCR)", f"{p90_w:.2f} A")

            st.subheader("Ambient Temperature Percentile Summary")

            p10_t = np.nanpercentile(results['Ambient_C'].dropna(), 10)
            p50_t = np.nanpercentile(results['Ambient_C'].dropna(), 50)
            p90_t = np.nanpercentile(results['Ambient_C'].dropna(), 90)

            col1, col2, col3 = st.columns(3)
            col1.metric("10th Percentile (Temp)", f"{p10_t:.2f} °C")
            col2.metric("Median (Temp)", f"{p50_t:.2f} °C")
            col3.metric("90th Percentile (Temp)", f"{p90_t:.2f} °C")

        with tab4:

            seasons = ['Winter', 'Spring', 'Summer', 'Fall']

            # -----------------------------
            # Seasonal Temperature Plots
            # -----------------------------
            st.subheader("Seasonal Ambient Temperature")

            cols_temp = st.columns(2)
            for idx, season in enumerate(seasons):
                season_data = results[results['Season'] == season]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=season_data['Time_day'],
                    y=season_data['Ambient_C'],
                    line=dict(color=SEASON_COLOR[season], width=1.5)
                ))

                fig.update_layout(
                    title=f"Ambient Temperature – {season}",
                    xaxis_title="Time (days)",
                    yaxis_title="Ambient Temperature (°C)",
                    showlegend=False
                )

                cols_temp[idx % 2].plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # Seasonal DCR Plots
            # -----------------------------
            st.subheader("Seasonal Dynamic Conductor Rating")

            cols_dcr = st.columns(2)
            for idx, season in enumerate(seasons):
                season_data = results[results['Season'] == season]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=season_data['Time_day'],
                    y=season_data['DCR'],
                    line=dict(color=SEASON_COLOR[season], width=1.5)
                ))

                fig.update_layout(
                    title=f"Dynamic Conductor Rating – {season}",
                    xaxis_title="Time (days)",
                    yaxis_title="Ampacity (A)",
                    showlegend=False
                )

                cols_dcr[idx % 2].plotly_chart(fig, use_container_width=True)
            #----------------------------------------
            # Seasonal Summary Table
            #----------------------------------------
            st.subheader("Seasonal Summary Statistics")

            st.dataframe(
                seasonal_summary.style.format("{:.2f}"),
                use_container_width=True
            )
            # ----------------------------------------
            # Seasonal Box Plot
            # ----------------------------------------

            def seasonal_boxplot(data_col, y_label, title):

                percentiles = [10, 30, 60, 90]
                fig = go.Figure()

                for season in seasons:
                    vals = results[results['Season'] == season][data_col].dropna()

                    fig.add_trace(go.Box(
                        y=vals,
                        name=season,
                        marker_color=SEASON_COLOR[season],
                        boxpoints=False
                    ))

                fig.update_layout(
                    title=title,
                    yaxis_title=y_label
                )

                return fig

            st.subheader("Seasonal DCR Distribution")

            fig_dcr_box = seasonal_boxplot(
                'DCR',
                'Ampacity (A)',
                'Seasonal Dynamic Conductor Rating'
            )
            st.plotly_chart(fig_dcr_box, use_container_width=True)

            #--------------------------------------
            # Seasonal Statistics Tables+Download
            #--------------------------------------

            st.subheader("Seasonal DCR Statistics Table")

            st.dataframe(
                seasonal_dcr_stats.style.format(
                    {c: "{:.3f}" for c in seasonal_dcr_stats.columns if c != 'Season'}
                ),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Seasonal DCR Statistics CSV",
                data=seasonal_dcr_stats.to_csv(index=False),
                file_name="seasonal_DCR_statistics.csv",
                mime="text/csv",
            )

            #------------------------------------------
            # Seasonal Percentiles KPIs
            #------------------------------------------
            st.subheader("Seasonal DCR Percentiles")

            p10 = np.nanpercentile(results['DCR'].dropna(), 10)
            p50 = np.nanpercentile(results['DCR'].dropna(), 50)
            p90 = np.nanpercentile(results['DCR'].dropna(), 90)

            col1, col2, col3 = st.columns(3)
            col1.metric("10th Percentile", f"{p10:.2f} A")
            col2.metric("Median", f"{p50:.2f} A")
            col3.metric("90th Percentile", f"{p90:.2f} A")


else:
    st.info(
        "👈 Configure your location, date range, and calculation mode in the sidebar, "
        "then click **SIMULATE** to run the analysis."
    )