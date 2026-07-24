import pandas as pd


# ----------------------------------------------------------
# Create Flat Feature Dictionary
# ----------------------------------------------------------

def create_flat_features(
    property_type,
    locality,
    area,
    bedroom,
    bathroom,
    balcony,
    floor_num,
    facing,
    age_possession,
    luxury_score,
    servant_room,
    store_room,
    study_room,
    pooja_room,
    others,
    ready_to_move,
    corner_property,
    park_facing,
    freehold,
    fully_furnished,
    newly_renovated
):
    """
    Create input dataframe for flat prediction.
    """

    data = {
        "property_type": property_type,
        "locality": locality,
        "area": area,
        "bedRoom": bedroom,
        "bathroom": bathroom,
        "balcony": balcony,
        "floorNum": floor_num,
        "facing": facing,
        "agePossession": age_possession,
        "luxury_score": luxury_score,
        "servant room": int(servant_room),
        "store room": int(store_room),
        "study room": int(study_room),
        "pooja room": int(pooja_room),
        "others": int(others),
        "ready_to_move": int(ready_to_move),
        "corner_property": int(corner_property),
        "park_facing": int(park_facing),
        "freehold": int(freehold),
        "fully_furnished": int(fully_furnished),
        "newly_renovated": int(newly_renovated)
    }

    return pd.DataFrame([data])


# ----------------------------------------------------------
# Create House Feature Dictionary
# ----------------------------------------------------------

def create_house_features(
    property_type,
    sector,
    built_up_area,
    plot_area,
    bedrooms,
    bathrooms,
    balconies,
    furnishing,
    servant_room,
    store_room,
    luxury_score,
    age_of_property
):
    """
    Returns dataframe for house prediction.
    """

    data = {
        "property_type": property_type,
        "sector": sector,
        "built_up_area": built_up_area,
        "plot_area": plot_area,
        "bedRoom": bedrooms,
        "bathroom": bathrooms,
        "balcony": balconies,
        "furnishing_type": furnishing,
        "servant room": servant_room,
        "store room": store_room,
        "luxury_score": luxury_score,
        "agePossession": age_of_property
    }

    return pd.DataFrame([data])


# ----------------------------------------------------------
# Create Plot Feature Dictionary
# ----------------------------------------------------------

def create_plot_features(
    sector,
    plot_area
):
    """
    Returns dataframe for residential plot prediction.
    """

    data = {
        "sector": sector,
        "plot_area": plot_area
    }

    return pd.DataFrame([data])


# ----------------------------------------------------------
# Convert Boolean
# ----------------------------------------------------------

def bool_to_int(value):
    """
    Converts Yes/No or True/False to integer.
    """

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        return 1 if value.lower() == "yes" else 0

    return int(value)


# ----------------------------------------------------------
# Validate Input
# ----------------------------------------------------------

def validate_positive(*values):
    """
    Returns True if every supplied value is positive.
    """

    return all(v > 0 for v in values)


# ----------------------------------------------------------
# Convert Prediction
# ----------------------------------------------------------

def format_price(price):
    """
    Formats price into Indian style.
    """

    if price >= 1e7:
        return f"₹ {price/1e7:.2f} Cr"

    if price >= 1e5:
        return f"₹ {price/1e5:.2f} Lakh"

    return f"₹ {price:,.0f}"


# ----------------------------------------------------------
# Area Conversion
# ----------------------------------------------------------

def sqft_to_sqm(area):
    return area * 0.092903


def sqm_to_sqft(area):
    return area * 10.7639