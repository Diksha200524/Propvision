import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Dataset Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "gurugram"

DATASETS = {
    "flat": DATASET_DIR / "flats.csv",
    "house": DATASET_DIR / "independent-house.csv",
    "plot": DATASET_DIR / "residential-land.csv"
}


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def load_dataset(property_type: str):

    property_type = property_type.lower()

    if property_type not in DATASETS:
        raise ValueError("Invalid Property Type")

    return pd.read_csv(DATASETS[property_type])


# ---------------------------------------------------------
# Budget Recommendation
# ---------------------------------------------------------

def recommend_by_budget(
    property_type,
    min_price,
    max_price,
    limit=10
):

    df = load_dataset(property_type)

    if "price" not in df.columns:
        return df.head(limit)

    recommendations = df[
        (df["price"] >= min_price) &
        (df["price"] <= max_price)
    ]

    return recommendations.head(limit)


# ---------------------------------------------------------
# Sector Recommendation
# ---------------------------------------------------------

def recommend_by_sector(
    property_type,
    sector,
    limit=10
):

    df = load_dataset(property_type)

    if "sector" not in df.columns:
        return pd.DataFrame()

    recommendations = df[
        df["sector"].str.lower() == sector.lower()
    ]

    return recommendations.head(limit)


# ---------------------------------------------------------
# Bedroom Recommendation
# ---------------------------------------------------------

def recommend_by_bedroom(
    property_type,
    bedrooms,
    limit=10
):

    df = load_dataset(property_type)

    if "bedRoom" not in df.columns:
        return pd.DataFrame()

    recommendations = df[
        df["bedRoom"] == bedrooms
    ]

    return recommendations.head(limit)


# ---------------------------------------------------------
# Area Recommendation
# ---------------------------------------------------------

def recommend_by_area(
    property_type,
    min_area,
    max_area,
    limit=10
):

    df = load_dataset(property_type)

    if "built_up_area" not in df.columns:
        return pd.DataFrame()

    recommendations = df[
        (df["built_up_area"] >= min_area) &
        (df["built_up_area"] <= max_area)
    ]

    return recommendations.head(limit)


# ---------------------------------------------------------
# Furnishing Recommendation
# ---------------------------------------------------------

def recommend_by_furnishing(
    property_type,
    furnishing,
    limit=10
):

    df = load_dataset(property_type)

    if "furnishing_type" not in df.columns:
        return pd.DataFrame()

    recommendations = df[
        df["furnishing_type"].str.lower() == furnishing.lower()
    ]

    return recommendations.head(limit)


# ---------------------------------------------------------
# Combined Recommendation
# ---------------------------------------------------------

def recommend(
    property_type,
    min_price=None,
    max_price=None,
    sector=None,
    bedrooms=None,
    min_area=None,
    max_area=None,
    furnishing=None,
    limit=10
):

    df = load_dataset(property_type)

    if min_price is not None and "price" in df.columns:
        df = df[df["price"] >= min_price]

    if max_price is not None and "price" in df.columns:
        df = df[df["price"] <= max_price]

    if sector and "sector" in df.columns:
        df = df[df["sector"].str.lower() == sector.lower()]

    if bedrooms is not None and "bedRoom" in df.columns:
        df = df[df["bedRoom"] == bedrooms]

    if min_area is not None and "built_up_area" in df.columns:
        df = df[df["built_up_area"] >= min_area]

    if max_area is not None and "built_up_area" in df.columns:
        df = df[df["built_up_area"] <= max_area]

    if furnishing and "furnishing_type" in df.columns:
        df = df[
            df["furnishing_type"].str.lower() == furnishing.lower()
        ]

    return df.head(limit)