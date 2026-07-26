import os
import joblib
import numpy as np
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

# =====================================================
# Load Dataset
# =====================================================

DATA_PATH = r"D:\data science\propvision\datasets\flat_final.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Shape :", df.shape)
print("=" * 60)

# =====================================================
# Selected Features
# =====================================================

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

TARGET = "price"

df = df[selected_features + [TARGET]]

# =====================================================
# Clean Numerical Columns
# =====================================================

numeric_columns = [
    "area",
    "bedRoom",
    "bathroom",
    "balcony",
    "floorNum",
    "luxury_score"
]

for col in numeric_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")

# =====================================================
# Binary Columns
# =====================================================

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

for col in binary_features:

    df[col] = df[col].fillna(0)
    df[col] = df[col].astype(int)

# =====================================================
# Feature Groups
# =====================================================

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
    "floorNum",
    "luxury_score"
]

# =====================================================
# Split
# =====================================================

X = df[selected_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# Pipelines
# =====================================================

cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

preprocessor = ColumnTransformer([
    (
        "cat",
        cat_pipeline,
        categorical_features
    ),
    (
        "num",
        num_pipeline,
        numerical_features
    ),
    (
        "bin",
        "passthrough",
        binary_features
    )
])

# =====================================================
# Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])

# =====================================================
# Train
# =====================================================

print("\nTraining Model...\n")

pipeline.fit(X_train, y_train)

print("Training Completed.")

# =====================================================
# Prediction
# =====================================================

pred = pipeline.predict(X_test)

# =====================================================
# Metrics
# =====================================================

print("\nModel Performance")
print("-" * 40)

print("R²   :", round(r2_score(y_test, pred), 4))
print("MAE  :", round(mean_absolute_error(y_test, pred), 4))
print("RMSE :", round(root_mean_squared_error(y_test, pred), 4))

# =====================================================
# Cross Validation
# =====================================================

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCross Validation R² Scores")

print(scores)

print("Average :", round(scores.mean(), 4))

# =====================================================
# Feature Importance
# =====================================================

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

# =====================================================
# Save Model
# =====================================================

MODEL_DIR = r"D:\data science\propvision\models"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "flat_model.pkl"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel Saved Successfully")

print(MODEL_PATH)