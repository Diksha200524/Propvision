import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"

FLAT_DATA = DATASET_DIR / r"D:\data science\propvision\datasets\flat_final.csv"
HOUSE_DATA = DATASET_DIR / r"D:\data science\propvision\datasets\cleaned_house_final.csv"
PLOT_DATA = DATASET_DIR / r"D:\data science\propvision\datasets\cleaned_plot_final.csv"


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def load_dataset(property_type):

    property_type = property_type.lower()

    if property_type == "flat":
        return pd.read_csv(FLAT_DATA)

    elif property_type == "house":
        return pd.read_csv(HOUSE_DATA)

    elif property_type == "plot":
        return pd.read_csv(PLOT_DATA)

    else:
        raise ValueError("Invalid Property Type")


# ---------------------------------------------------------
# Numeric Columns
# ---------------------------------------------------------

def get_numeric_columns(df):

    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


# ---------------------------------------------------------
# Similar Properties
# ---------------------------------------------------------

def get_similar_properties(
        property_type,
        selected_index,
        top_n=5
):

    df = load_dataset(property_type)

    numeric_columns = get_numeric_columns(df)

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df[numeric_columns])

    similarity = cosine_similarity(scaled)

    similarity_scores = similarity[selected_index]

    indices = similarity_scores.argsort()[::-1]

    indices = indices[1:top_n + 1]

    return df.iloc[indices]


# ---------------------------------------------------------
# Search Property
# ---------------------------------------------------------

def search_property(
        property_type,
        keyword,
        column="property_name"
):

    df = load_dataset(property_type)

    if column not in df.columns:
        return pd.DataFrame()

    result = df[df[column].str.contains(
        keyword,
        case=False,
        na=False
    )]

    return result


# ---------------------------------------------------------
# Filter by Sector
# ---------------------------------------------------------

def filter_sector(
        property_type,
        sector
):

    df = load_dataset(property_type)

    if "sector" not in df.columns:
        return df

    return df[df["sector"] == sector]


# ---------------------------------------------------------
# Budget Filter
# ---------------------------------------------------------

def filter_budget(
        property_type,
        minimum,
        maximum
):

    df = load_dataset(property_type)

    if "price" not in df.columns:
        return df

    return df[
        (df["price"] >= minimum) &
        (df["price"] <= maximum)
    ]


# ---------------------------------------------------------
# Get Property by Index
# ---------------------------------------------------------

def get_property(
        property_type,
        index
):

    df = load_dataset(property_type)

    return df.iloc[index]