import streamlit as st
import pandas as pd

from modules.features import create_flat_features
from modules.predict import predict_flat
from modules.emi import calculate_emi
from modules.investment import investment_summary

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("datasets/flat_final.csv")

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

    prediction = predict_flat(sample)

    st.success(
        f"Predicted Price : ₹ {prediction:.2f} Cr"
    )

    # --------------------------------------
    # Property Summary
    # --------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Area",
        f"{area} Sq.ft"
    )

    c2.metric(
        "Bedrooms",
        bedroom
    )

    c3.metric(
        "Bathrooms",
        bathroom
    )

    st.divider()

    # --------------------------------------
    # EMI Calculator
    # --------------------------------------

    st.subheader("🏦 EMI Calculator")

    loan = prediction * 0.8

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
    # Investment Analysis
    # --------------------------------------

    st.subheader("📈 Investment Analysis")

    summary = investment_summary(
        purchase_price=prediction,
        annual_growth_rate=8,
        years=10,
        monthly_rent=30000
    )

    st.dataframe(
        pd.DataFrame(
            summary.items(),
            columns=["Metric", "Value"]
        ),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------
    # Similar Properties
    # --------------------------------------

    st.subheader("🏠 Similar Properties")

    similar = df[
        (df["locality"] == locality)
        &
        (df["bedRoom"] == bedroom)
    ]

    st.dataframe(
        similar.head(5),
        use_container_width=True
    )