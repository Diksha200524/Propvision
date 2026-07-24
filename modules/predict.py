import joblib
import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

FLAT_MODEL_PATH = MODEL_DIR / "flat_model.pkl"
HOUSE_MODEL_PATH = MODEL_DIR / "house_model.pkl"
PLOT_MODEL_PATH = MODEL_DIR / "plot_model.pkl"


# -------------------------------------------------------
# Load Models
# -------------------------------------------------------

flat_model = None
house_model = None
plot_model = None


def load_models():
    """
    Loads all trained models only once.
    """

    global flat_model
    global house_model
    global plot_model

    if flat_model is None:
        flat_model = joblib.load(FLAT_MODEL_PATH)

    if house_model is None:
        house_model = joblib.load(HOUSE_MODEL_PATH)

    if plot_model is None:
        plot_model = joblib.load(PLOT_MODEL_PATH)


# -------------------------------------------------------
# Flat Prediction
# -------------------------------------------------------

def predict_flat(data: pd.DataFrame):

    load_models()

    prediction = flat_model.predict(data)[0]

    return round(float(prediction), 2)


# -------------------------------------------------------
# House Prediction
# -------------------------------------------------------

def predict_house(data: pd.DataFrame):

    load_models()

    prediction = house_model.predict(data)[0]

    return round(float(prediction), 2)


# -------------------------------------------------------
# Plot Prediction
# -------------------------------------------------------

def predict_plot(data: pd.DataFrame):

    load_models()

    prediction = plot_model.predict(data)[0]

    return round(float(prediction), 2)


# -------------------------------------------------------
# Predict by Property Type
# -------------------------------------------------------

def predict(property_type: str, data: pd.DataFrame):

    property_type = property_type.lower()

    if property_type == "flat":
        return predict_flat(data)

    elif property_type == "house":
        return predict_house(data)

    elif property_type == "plot":
        return predict_plot(data)

    else:
        raise ValueError("Invalid Property Type")