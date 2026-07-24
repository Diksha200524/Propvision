import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# --------------------------------------------------------
# Load Dataset
# --------------------------------------------------------

df = pd.read_csv(r"D:\data science\propvision\datasets\flat_final.csv")
print(df.dtypes)
print(df["balcony"].dtype)
print(df["balcony"].unique())
numerical_features = [
    "area",
    "bedRoom",
    "bathroom",
    "balcony",   # <-- problem
    "floorNum",
    "luxury_score"
]

for col in numerical_features:
    print(col)
    print(df[col].dtype)
    print(df[col].unique()[:10])
    print("-"*40)

# --------------------------------------------------------
# Selected Features
# --------------------------------------------------------

selected_features = [
    "property_type",
    "locality",
    "area",
    "bedRoom",
    "bathroom",
    "balcony",
    "floorNum",
    "facing",
    "agePossession",
    "luxury_score",
    "servant room",
    "store room",
    "study room",
    "pooja room",
    "others",
    "ready_to_move",
    "corner_property",
    "park_facing",
    "freehold",
    "fully_furnished",
    "newly_renovated"
]

target = "price"

df = df[selected_features + [target]]

# --------------------------------------------------------
# Missing Values
# --------------------------------------------------------

df.dropna(inplace=True)

# --------------------------------------------------------
# X and y
# --------------------------------------------------------

X = df[selected_features]

y = df[target]

# --------------------------------------------------------
# Train Test Split
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------------
# Feature Groups
# --------------------------------------------------------

categorical_features = [
    "property_type",
    "locality",
    "facing",
    "agePossession",
    "balcony"
]

numerical_features = [
    "area",
    "bedRoom",
    "bathroom",
    "floorNum",
    "luxury_score"
]

binary_features = [
    "servant room",
    "store room",
    "study room",
    "pooja room",
    "others",
    "ready_to_move",
    "corner_property",
    "park_facing",
    "freehold",
    "fully_furnished",
    "newly_renovated"
]

# --------------------------------------------------------
# Preprocessor
# --------------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "num",
        StandardScaler(),
        numerical_features
    ),
    (
        "bin",
        "passthrough",
        binary_features
    )
])

# --------------------------------------------------------
# Pipeline
# --------------------------------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ))
])

# --------------------------------------------------------
# Train
# --------------------------------------------------------

pipeline.fit(X_train, y_train)

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

pred = pipeline.predict(X_test)

# --------------------------------------------------------
# Metrics
# --------------------------------------------------------

print("R2 :", round(r2_score(y_test, pred),4))
print("MAE :", round(mean_absolute_error(y_test,pred),2))
print("RMSE :", round(root_mean_squared_error(y_test,pred),2))

# --------------------------------------------------------
# Save
# --------------------------------------------------------

joblib.dump(
    pipeline,
    r"/models/flat_model.pkl"
)

print("Model Saved Successfully.")