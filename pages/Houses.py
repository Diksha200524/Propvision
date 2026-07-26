import streamlit as st
import pandas as pd

from modules.predict import predict_house
from modules.emi import calculate_emi
from modules.investment import investment_summary

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATASET_DIR / "cleaned_house_final.csv")

st.title("🏡 Independent House Price Prediction")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

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
    min_value=200,
    value=1500
)

bedroom = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathroom = st.sidebar.slider(
    "Bathrooms",
    1,
    10,
    3
)

balcony = st.sidebar.slider(
    "Balconies",
    0,
    5,
    2
)

floor_num = st.sidebar.slider(
    "Floor Number",
    0,
    5,
    2
)

facing = st.sidebar.selectbox(
    "Facing",
    sorted(df["facing"].dropna().unique())
)

age = st.sidebar.selectbox(
    "Age Possession",
    sorted(df["agePossession"].dropna().unique())
)

luxury = st.sidebar.slider(
    "Luxury Score",
    0,
    100,
    50
)

# ----------------------------------------------------
# Amenities
# ----------------------------------------------------

servant = st.sidebar.checkbox("Servant Room")
store = st.sidebar.checkbox("Store Room")
study = st.sidebar.checkbox("Study Room")
pooja = st.sidebar.checkbox("Pooja Room")
others = st.sidebar.checkbox("Others")

ready = st.sidebar.checkbox("Ready To Move")
corner = st.sidebar.checkbox("Corner Property")
park = st.sidebar.checkbox("Park Facing")
freehold = st.sidebar.checkbox("Freehold")
furnished = st.sidebar.checkbox("Fully Furnished")
renovated = st.sidebar.checkbox("Newly Renovated")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if st.button("Predict Price"):
    sample = pd.DataFrame([{

        "property_type": property_type,
        "locality": locality,

        # Your model was trained using 'area'
        "area": area,

        "bedRoom": bedroom,
        "bathroom": bathroom,
        "balcony": balcony,

        # Model expects noOfFloor
        "noOfFloor": floor_num,

        "facing": facing,
        "agePossession": age,
        "luxury_score": luxury,

        "study room": int(study),
        "servant room": int(servant),
        "store room": int(store),
        "pooja room": int(pooja),
        "others": int(others),

        "ready_to_move": int(ready),

        # Model expects corner_plot
        "corner_plot": int(corner),

        "park_facing": int(park),
        "freehold": int(freehold),
        "newly_renovated": int(renovated),
        "fully_furnished": int(furnished)

    }])

    # Debug
    print(sample.columns.tolist())

    prediction = predict_house(sample)

    st.success(
        f"Estimated Price : ₹ {prediction:.2f} Cr"
    )

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric(
        "Area",
        f"{area} Sq.ft"
    )



    c2.metric(
        "Bedrooms",
        bedroom
    )

    st.divider()

    # EMI

    st.subheader("🏦 EMI")
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
        "Interest",
        f"₹ {emi['total_interest']:,.0f}"
    )

    c.metric(
        "Total Payment",
        f"₹ {emi['total_payment']:,.0f}"
    )

    st.divider()

    # Investment

    st.subheader("📈 Investment")

    summary = investment_summary(
        purchase_price=prediction,
        annual_growth_rate=8,
        years=10,
        monthly_rent=35000
    )

    st.dataframe(
        pd.DataFrame(
            summary.items(),
            columns=["Metric", "Value"]
        ),
        use_container_width=True
    )

    st.divider()


    # ----------------------------------------------------
    # Similar Houses
    # ----------------------------------------------------

    st.subheader("🏠 Similar Houses")

    similar = df[
        (df["locality"] == locality)
    ].copy()

    # Difference from selected area
    similar["area_diff"] = abs(similar["area"] - area)

    # Difference from selected bedrooms
    similar["bedroom_diff"] = abs(similar["bedRoom"] - bedroom)

    # Sort by similarity
    similar = similar.sort_values(
        by=["bedroom_diff", "area_diff"]
    )

    # Remove helper columns
    similar = similar.drop(
        columns=["area_diff", "bedroom_diff"]
    )

    st.dataframe(
        similar.head(5),
        use_container_width=True
    )