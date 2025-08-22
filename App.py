import streamlit as st
from pathlib import Path

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(page_title="Evaluation of GHM", page_icon=":globe_with_meridians:", layout="wide")
st.title("Global Hydrological Model Evaluation Application")

# App Title
st.title('Evaluation of Global Hydrological Model')

# --- Sidebar ---
st.sidebar.title("Hydrology Research Group") 
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

elif main_selection == "Evaluation setup":
    sub_selection = st.sidebar.radio(
        "Evaluation setup:",
        ("WaterGAP",)
    )
    st.header(sub_selection)
    st.write(f"Details for Evaluation setup: **{sub_selection}**.")

elif main_selection == "Contact Us":
    st.header("Contact Us")
    st.write("📧 Email: hydrology@yahoo.com")
    st.write("🏢 Address: Hydrology Research Group, Example University")

