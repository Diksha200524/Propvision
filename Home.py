import streamlit as st

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------------------------------
# Title
# -----------------------------------------------------
st.title("🏠 Real Estate Analytics Platform")
st.markdown(
    """
    Welcome to the **Real Estate Analytics Platform**.

    This application helps buyers, sellers, and investors make
    informed decisions using Machine Learning and Data Analysis.
    """
)

st.divider()

# -----------------------------------------------------
# Features
# -----------------------------------------------------
st.subheader("🚀 Features")

col1, col2 = st.columns(2)

with col1:

    st.success("Price Prediction")

    st.write("""
    Predict the market price of

    - Flats
    - Independent Houses
    - Residential Plots
    """)

    st.success("Similar Property Recommendation")

    st.write("""
    Get properties similar to your selected one based on
    location, area, amenities and price.
    """)

with col2:

    st.success("EMI Calculator")

    st.write("""
    Estimate

    - Monthly EMI
    - Total Interest
    - Total Payment
    """)

    st.success("Investment Analysis")

    st.write("""
    Analyze

    - Future Property Value
    - CAGR
    - Expected ROI
    """)

st.divider()

# -----------------------------------------------------
# Workflow
# -----------------------------------------------------
st.subheader("📊 Application Workflow")

st.markdown("""
1. Select a property category from the sidebar.
2. Enter property details.
3. Predict the estimated market price.
4. View similar properties.
5. Calculate EMI.
6. Analyze investment potential.
""")

st.divider()

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
st.sidebar.title("Navigation")

st.sidebar.info(
    """
    Select one of the pages:

    • Flats

    • Houses

    • Plots
    """
)

st.sidebar.success("Built with Streamlit & Machine Learning")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.divider()

st.caption("© 2026 Real Estate Analytics Platform")