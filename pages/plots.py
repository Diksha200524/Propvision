import streamlit as st
import pandas as pd

from modules.predict import predict_plot
from modules.emi import calculate_emi
from modules.investment import investment_summary

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------

st.set_page_config(
    page_title="Plot Price Prediction",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Residential Plot Price Prediction")

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATASET_DIR / "cleaned_plot_final.csv")
# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.header("Plot Details")

locality = st.sidebar.selectbox(
    "Locality",
    sorted(df["locality"].dropna().unique())
)

area = st.sidebar.number_input(
    "Plot Area (Sq.ft)",
    min_value=100,
    max_value=100000,
    value=1500
)

open_sides = st.sidebar.slider(
    "Open Sides",
    1,
    4,
    2
)

possession = st.sidebar.selectbox(
    "Possession",
    sorted(df["possession"].dropna().unique())
)

corner_property = st.sidebar.checkbox("Corner Property")

park_facing = st.sidebar.checkbox("Park Facing")

freehold = st.sidebar.checkbox("Freehold")

ready_to_move = st.sidebar.checkbox("Ready To Move")

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

if st.button("Predict Price"):

    sample = pd.DataFrame([{

        "area": area,
        "open_sides": open_sides,
        "possession": possession,
        "locality": locality,
        "corner_property": int(corner_property),
        "park_facing": int(park_facing),
        "freehold": int(freehold),
        "ready_to_move": int(ready_to_move)

    }])

    prediction = predict_plot(sample)

    st.success(
        f"Estimated Plot Price : ₹ {prediction:.2f} Cr"
    )

    st.divider()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Area",
        f"{area} Sq.ft"
    )

    c2.metric(
        "Open Sides",
        open_sides
    )

    c3.metric(
        "Possession",
        possession
    )

    st.divider()

    # -----------------------------------------------------
    # EMI
    # -----------------------------------------------------

    st.subheader("🏦 EMI Calculator")

    # prediction is in Crores
    loan_amount = prediction * 10000000 * 0.80

    emi = calculate_emi(
        loan_amount=loan_amount,
        annual_interest_rate=8.5,
        tenure_years=20
    )
    a, b, c = st.columns(3)

    a.metric(
        "Monthly EMI",
        f"₹ {emi['monthly_emi']:,.0f}"
    )

    b.metric(
        "Total Interest",
        f"₹ {emi['total_interest']:,.0f}"
    )

    c.metric(
        "Total Payment",
        f"₹ {emi['total_payment']:,.0f}"
    )

    st.divider()

    # -----------------------------------------------------
    # Investment
    # -----------------------------------------------------

    st.subheader("📈 Investment Analysis")

    summary = investment_summary(
        purchase_price=prediction,
        annual_growth_rate=8,
        years=10,
        monthly_rent=0
    )

    st.dataframe(
        pd.DataFrame(
            summary.items(),
            columns=["Metric", "Value"]
        ),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # Similar Plots
    # -----------------------------------------------------

    st.subheader("🏘 Similar Plots")

    similar = df[
        (df["locality"] == locality)
    ]

    st.dataframe(
        similar.head(5),
        use_container_width=True
    )