import streamlit as st
import pandas as pd

from modules.features import create_flat_features
from modules.predict import predict_flat
from modules.emi import calculate_emi
from modules.investment import investment_summary

# ------------------------------------------
# Load Dataset
# ------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATASET_DIR / "flat_final.csv")

st.set_page_config(page_title="Flat Price Prediction", layout="wide")

st.title("🏢 Flat Price Prediction")

# ------------------------------------------
# Sidebar
# ------------------------------------------

st.sidebar.header("Property Details")

property_type = st.sidebar.selectbox(
    "Property Type",
    sorted(df["property_type"].dropna().unique())
)

locality = st.sidebar.selectbox(
    "Locality",
    sorted(df["locality"].dropna().unique())
)

area = st.sidebar.number_input(
    "Area (Sq.ft)",
    min_value=100,
    max_value=10000,
    value=1500
)

bedroom = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathroom = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=3
)

balcony = st.sidebar.number_input(
    "Balcony",
    min_value=0,
    max_value=10,
    value=2
)

floor_num = st.sidebar.number_input(
    "Floor Number",
    min_value=0,
    max_value=60,
    value=5
)

facing = st.sidebar.selectbox(
    "Facing",
    sorted(df["facing"].dropna().unique())
)

age = st.sidebar.selectbox(
    "Age of Property",
    sorted(df["agePossession"].dropna().unique())
)

luxury_score = st.sidebar.slider(
    "Luxury Score",
    0,
    100,
    50
)

# ------------------------------------------
# Amenities
# ------------------------------------------

st.sidebar.subheader("Amenities")

servant_room = st.sidebar.checkbox("Servant Room")

store_room = st.sidebar.checkbox("Store Room")

study_room = st.sidebar.checkbox("Study Room")

pooja_room = st.sidebar.checkbox("Pooja Room")

others = st.sidebar.checkbox("Others")

ready = st.sidebar.checkbox("Ready To Move")

corner = st.sidebar.checkbox("Corner Property")

park = st.sidebar.checkbox("Park Facing")

freehold = st.sidebar.checkbox("Freehold")

fully = st.sidebar.checkbox("Fully Furnished")

renovated = st.sidebar.checkbox("Newly Renovated")

# ------------------------------------------
# Prediction
# ------------------------------------------

if st.button("Predict Price"):

    sample = create_flat_features(
        property_type,
        locality,
        area,
        bedroom,
        bathroom,
        balcony,
        floor_num,
        facing,
        age,
        luxury_score,
        servant_room,
        store_room,
        study_room,
        pooja_room,
        others,
        ready,
        corner,
        park,
        freehold,
        fully,
        renovated
    )

    try:
        prediction = predict_flat(sample)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
    price_per_sqft = (prediction * 10000000) / area
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Price", f"₹ {prediction:.2f} Cr")
    c2.metric("Area", f"{area} Sq.ft")
    c3.metric("Bedrooms", bedroom)
    c4.metric("Price/Sq.ft", f"₹ {price_per_sqft:,.0f}")

    st.success(
        f"Predicted Price : ₹ {prediction:.2f} Cr"
    )

    # --------------------------------------
    # Property Summary
    # --------------------------------------



    # --------------------------------------
    # EMI Calculator
    # --------------------------------------

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

    # --------------------------------------
    # --------------------------------------
    # Investment Analysis
    # --------------------------------------

    st.subheader("📈 Investment Analysis")

    summary = investment_summary(
        purchase_price=prediction,
        annual_growth_rate=8,
        years=10,
        monthly_rent=30000
    )

    # Uncomment once to check available keys
    # st.write(summary)

    col1, col2 = st.columns(2)

    future_value = (
        summary.get("Future Property Value")
        or summary.get("Future Value")
        or 0
    )

    capital_gain = (
        summary.get("Capital Gain")
        or summary.get("Profit")
        or 0
    )

    col1.metric(
        "Future Value",
        f"₹ {future_value:,.2f}"
    )

    col2.metric(
        "Capital Gain",
        f"₹ {capital_gain:,.2f}"
    )

    st.divider()

    # --------------------------------------
    # Similar Properties
    # --------------------------------------

    st.subheader("🏠 Similar Properties")

    similar = df[df["locality"] == locality].copy()

    # Calculate similarity
    similar["area_diff"] = abs(similar["area"] - area)
    similar["bedroom_diff"] = abs(similar["bedRoom"] - bedroom)
    similar["bathroom_diff"] = abs(similar["bathroom"] - bathroom)

    # Filter similar price range
    similar = similar[
        (similar["price"] >= prediction * 0.80)
        &
        (similar["price"] <= prediction * 1.20)
    ]

    # Sort by similarity
    similar = similar.sort_values(
        by=[
            "bedroom_diff",
            "bathroom_diff",
            "area_diff"
        ]
    )

    # Remove helper columns
    similar.drop(
        columns=[
            "area_diff",
            "bedroom_diff",
            "bathroom_diff"
        ],
        inplace=True
    )

    display_columns = [
        "society",
        "price",
        "price_per_sqft",
        "area",
        "bedRoom",
        "bathroom",
        "balcony",
        "locality"
    ]

    # Keep only existing columns
    display_columns = [
        col for col in display_columns
        if col in similar.columns
    ]

    if len(similar) > 0:
        st.dataframe(
            similar[display_columns].head(5),
            use_container_width=True
        )
    else:
        st.info("No similar properties found.")