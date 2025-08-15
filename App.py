import streamlit as st

# App Title
st.title('Evaluation of Global Hydrological Model')

# --- Sidebar ---
st.sidebar.title("Hydrology Research Group")  # Title at the very top
st.sidebar.header("Main Menu")

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

