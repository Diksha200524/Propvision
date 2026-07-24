import streamlit as st
import pandas as pd
from modules.similar import get_similar_properties
from modules.features import create_flat_features
from modules.predict import predict_flat
from modules.emi import (
    calculate_emi,
    amortization_schedule
)
from modules.investment import investment_summary
import plotly.graph_objects as go

import plotly.express as px
from modules.report import generate_report
import tempfile
# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Flat Price Prediction",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Flat Price Prediction")

st.markdown(
    "Predict the estimated market price of a flat using Machine Learning."
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(r"D:\data science\propvision\datasets\flat_final.csv")

df = load_data()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

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
    "Area (sq.ft)",
    min_value=100,
    max_value=10000,
    value=1200
)

bedroom = st.sidebar.selectbox(
    "Bedrooms",
    sorted(df["bedRoom"].dropna().unique())
)

bathroom = st.sidebar.selectbox(
    "Bathrooms",
    sorted(df["bathroom"].dropna().unique())
)

balcony = st.sidebar.selectbox(
    "Balcony",
    sorted(df["balcony"].dropna().unique())
)

floor_num = st.sidebar.number_input(
    "Floor Number",
    0,
    60,
    5
)

facing = st.sidebar.selectbox(
    "Facing",
    sorted(df["facing"].dropna().unique())
)

age = st.sidebar.selectbox(
    "Age Possession",
    sorted(df["agePossession"].dropna().unique())
)

luxury_score = st.sidebar.slider(
    "Luxury Score",
    0,
    200,
    50
)

st.sidebar.divider()

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

furnished = st.sidebar.checkbox("Fully Furnished")

renovated = st.sidebar.checkbox("Newly Renovated")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Price", use_container_width=True):

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

        furnished,

        renovated

    )

    prediction = predict_flat(sample)

    st.success("Prediction Completed")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Price",
            f"₹ {prediction:.2f} Cr"
        )

    with col2:

        st.metric(
            "Area",
            f"{area} sqft"
        )

    st.dataframe(sample)
# ---------------------------------------------------
# Similar Properties
# ---------------------------------------------------

st.divider()

st.header("🏠 Similar Properties")

if st.button("Show Similar Properties"):

    # Find property in dataset with same locality & bedroom
    filtered_df = df[
        (df["locality"] == locality) &
        (df["bedRoom"] == bedroom)
    ]

    if len(filtered_df) == 0:

        st.warning("No similar property found.")

    else:

        index = filtered_df.index[0]

        similar = get_similar_properties(
            property_type="flat",
            selected_index=index,
            top_n=5
        )

        st.success(f"{len(similar)} Similar Properties Found")

        for i, row in similar.iterrows():

            with st.container():

                col1, col2 = st.columns([1,3])

                with col1:

                    st.image(
                        "https://cdn-icons-png.flaticon.com/512/619/619153.png",
                        width=120
                    )

                with col2:

                    st.subheader(row["property_name"])

                    st.write(f"📍 {row['locality']}")

                    c1,c2,c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Price",
                            f"₹ {row['price']:.2f} Cr"
                        )

                    with c2:
                        st.metric(
                            "Area",
                            f"{row['area']} sqft"
                        )

                    with c3:
                        st.metric(
                            "Bedrooms",
                            row["bedRoom"]
                        )

                    st.write("### Features")

                    feature_list = []

                    if row.get("servant room",0)==1:
                        feature_list.append("Servant Room")

                    if row.get("study room",0)==1:
                        feature_list.append("Study Room")

                    if row.get("store room",0)==1:
                        feature_list.append("Store Room")

                    if row.get("pooja room",0)==1:
                        feature_list.append("Pooja Room")

                    if row.get("park_facing",0)==1:
                        feature_list.append("Park Facing")

                    if row.get("fully_furnished",0)==1:
                        feature_list.append("Fully Furnished")

                    if len(feature_list):

                        st.write(", ".join(feature_list))

                    else:

                        st.write("No extra features")

                    st.divider()
# ===========================================================
# EMI CALCULATOR
# ===========================================================

st.divider()

st.header("💰 EMI Calculator")

col1, col2 = st.columns(2)

with col1:

    property_price = st.number_input(
        "Property Price (₹)",
        min_value=100000,
        value=10000000,
        step=50000
    )

    down_payment = st.number_input(
        "Down Payment (₹)",
        min_value=0,
        value=2000000,
        step=50000
    )

with col2:

    interest_rate = st.slider(
        "Interest Rate (%)",
        5.0,
        15.0,
        8.5,
        0.1
    )

    tenure = st.slider(
        "Loan Tenure (Years)",
        1,
        30,
        20
    )

loan_amount = property_price - down_payment

if st.button("Calculate EMI", use_container_width=True):

    result = calculate_emi(
        loan_amount,
        interest_rate,
        tenure
    )

    st.success("EMI Calculated Successfully")
    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Monthly EMI",
            f"₹ {result['monthly_emi']:,.0f}"
        )

    with c2:

        st.metric(
            "Total Interest",
            f"₹ {result['total_interest']:,.0f}"
        )

    with c3:

        st.metric(
            "Total Payment",
            f"₹ {result['total_payment']:,.0f}"
        )
    chart = px.pie(

        names=["Principal","Interest"],

        values=[
            loan_amount,
            result["total_interest"]
        ],

        title="Loan Breakdown"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )
    schedule = amortization_schedule(
        loan_amount,
        interest_rate,
        tenure
    )

    st.subheader("Loan Schedule")

    st.dataframe(
        schedule,
        use_container_width=True
    )
    csv = schedule.to_csv(index=False)

    st.download_button(

        "📥 Download Schedule",

        csv,

        "loan_schedule.csv",

        "text/csv"
    )