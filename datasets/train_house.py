import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# =========================================================
# Load Dataset
# =========================================================

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

df = pd.read_csv(DATASET_DIR / "cleaned_house_final.csv")
print("Dataset Shape :", df.shape)

# =========================================================
# Selected Features
# =========================================================

selected_features = [
    "property_type",
    "locality",
    "area",
    "bedRoom",
    "bathroom",
    "balcony",
    "noOfFloor",
    "facing",
    "agePossession",
    "luxury_score",
    "study room",
    "servant room",
    "store room",
    "pooja room",
    "others",
    "ready_to_move",
    "corner_plot",
    "park_facing",
    "freehold",
    "newly_renovated",
    "fully_furnished"
]

TARGET = "price"

df = df[selected_features + [TARGET]]

# =========================================================
# Clean Numeric Columns
# =========================================================

numeric_cols = [
    "area",
    "bedRoom",
    "bathroom",
    "balcony",
    "noOfFloor",
    "luxury_score"
]

for col in numeric_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================================================
# Binary Columns
# =========================================================

binary_features = [
    "study room",
    "servant room",
    "store room",
    "pooja room",
    "others",
    "ready_to_move",
    "corner_plot",
    "park_facing",
    "freehold",
    "newly_renovated",
    "fully_furnished"
]

for col in binary_features:
    df[col] = df[col].fillna(0).astype(int)

# =========================================================
# Feature Groups
# =========================================================

categorical_features = [
    "property_type",
    "locality",
    "facing",
    "agePossession"
]

numerical_features = [
    "area",
    "bedRoom",
    "bathroom",
    "balcony",
    "noOfFloor",
    "luxury_score"
]

# =========================================================
# Split
# =========================================================
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["price"] >= lower) &
    (df["price"] <= upper)
]
X = df[selected_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================================================
# Preprocessing
# =========================================================

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, categorical_features),
    ("num", num_pipeline, numerical_features),
    ("bin", "passthrough", binary_features)
])

# =========================================================
# Model
# =========================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================================================
# Train
# =========================================================

print("Training...")

pipeline.fit(X_train, y_train)

print("Training Completed")

# =========================================================
# Prediction
# =========================================================

pred = pipeline.predict(X_test)

print("R² :", round(r2_score(y_test, pred), 4))
print("MAE :", round(mean_absolute_error(y_test, pred), 4))
print("RMSE :", round(root_mean_squared_error(y_test, pred), 4))

# =========================================================
# Cross Validation
# =========================================================

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCross Validation R²")

print(scores)

print("Average :", round(scores.mean(), 4))

# =========================================================
# Feature Importance
# =========================================================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")

print(importance_df.head(20))

# =========================================================
# Save Model
# =========================================================
# =========================================================
# Save Model & Artifacts
# =========================================================

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Save Model
joblib.dump(
    pipeline,
    MODEL_DIR / "house_model.pkl"
)

# Save Selected Features
joblib.dump(
    selected_features,
    MODEL_DIR / "house_features.pkl"
)

# Save Localities
localities = sorted(
    df["locality"].dropna().unique()
)

joblib.dump(
    localities,
    MODEL_DIR / "house_localities.pkl"
)

# Save Feature Importance
importance_df.to_csv(
    MODEL_DIR / "house_feature_importance.csv",
    index=False
)

# Save Metrics
metrics = {
    "R2": r2_score(y_test, pred),
    "MAE": mean_absolute_error(y_test, pred),
    "RMSE": root_mean_squared_error(y_test, pred),
    "CV_R2": scores.mean()
}

joblib.dump(
    metrics,
    MODEL_DIR / "house_metrics.pkl"
)

print("\nAll artifacts saved successfully!")
print(MODEL_DIR)
