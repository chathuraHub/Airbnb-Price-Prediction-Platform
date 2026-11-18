"""Best CatBoost model, trained on log(price).

This one file replaces the five earlier CatBoost experiment files.
It uses the best setup from those experiments (experiment 2: rounded latitude/longitude
"grid cells" as an extra categorical feature) and trains on log_price.

Run from the project root:
    python -m src.Airbnb.pipelines.CatBoost_pipeline

Creates in airbnb-price-prediction/:
    CatBoost_model.cbm            the trained model
    CatBoost_model_metrics.json   the scores
"""
import os
import csv
import io
import json
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.Airbnb.components.Data_ingestion import DataIngestion
from src.Airbnb.logger import logging

# --------------------------------------------------
# Settings
# --------------------------------------------------
OUTPUT_DIR = "airbnb-price-prediction"
MODEL_PATH = os.path.join(OUTPUT_DIR, "CatBoost_model.cbm")
METRICS_PATH = os.path.join(OUTPUT_DIR, "CatBoost_model_metrics.json")

USE_LOCATION_CELL = True   # True = best experiment (grid cells). False = plain CatBoost.
ROUND_DECIMALS = 2         # 2 decimals is roughly a 1 km grid cell
RANDOM_STATE = 42

NUMERICAL_COLS = [
    "amenities", "accommodates", "bathrooms", "latitude", "longitude",
    "host_response_rate", "number_of_reviews", "review_scores_rating",
    "bedrooms", "beds",
]
CATEGORICAL_COLS = [
    "property_type", "room_type", "bed_type", "cancellation_policy",
    "cleaning_fee", "city", "host_identity_verified", "instant_bookable",
    "host_has_profile_pic",
]
DROP_COLUMNS = [
    "log_price", "price", "id", "name", "description", "first_review",
    "host_since", "last_review", "neighbourhood", "thumbnail_url", "zipcode",
]


# --------------------------------------------------
# Data preparation
# --------------------------------------------------
def count_amenities(value):
    """Count items in '{TV,"Cable TV",...}', respecting commas inside quotes."""
    if pd.isna(value) or str(value).strip() in ("", "{}"):
        return np.nan
    value = str(value).strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    items = next(csv.reader(io.StringIO(value), skipinitialspace=True))
    return sum(1 for item in items if item.strip())


def add_location_cell(X):
    """Rounded latitude_longitude text, e.g. '40.73_-73.99', as a categorical feature."""
    ok = X["latitude"].notna() & X["longitude"].notna()
    cell = pd.Series("Unknown", index=X.index, dtype="object")
    cell.loc[ok] = (
        X.loc[ok, "latitude"].round(ROUND_DECIMALS).map(lambda v: f"{v:.{ROUND_DECIMALS}f}")
        + "_"
        + X.loc[ok, "longitude"].round(ROUND_DECIMALS).map(lambda v: f"{v:.{ROUND_DECIMALS}f}")
    )
    X["location_cell"] = cell
    return X


def prepare_data(df):
    """Returns (features, log_price). The target is log(price)."""
    df = df.copy()

    # Target on the log scale (created from price if the CSV does not have it)
    if "log_price" in df.columns:
        y = pd.to_numeric(df["log_price"], errors="coerce")
    elif "price" in df.columns:
        y = np.log(pd.to_numeric(df["price"], errors="coerce"))
    else:
        raise ValueError("The data needs a 'log_price' or 'price' column.")

    df["cleaning_fee"] = df["cleaning_fee"].astype(str).str.upper()
    df["host_response_rate"] = pd.to_numeric(
        df["host_response_rate"].astype(str).str.replace("%", "", regex=False),
        errors="coerce")
    df["amenities"] = df["amenities"].apply(count_amenities)

    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    X = X[NUMERICAL_COLS + CATEGORICAL_COLS].copy()

    for col in NUMERICAL_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")      # CatBoost accepts NaN here
    for col in CATEGORICAL_COLS:                             # but not in categorical columns
        X[col] = X[col].where(X[col].notna(), "Unknown").astype(str)

    if USE_LOCATION_CELL:
        X = add_location_cell(X)

    valid = y.notna() & np.isfinite(y)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------
def scores(y_log_true, y_log_pred):
    """Metrics on log price, and on real price after converting back with exp()."""
    price_true, price_pred = np.exp(y_log_true), np.exp(y_log_pred)
    return {
        "r2_log": float(r2_score(y_log_true, y_log_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_log_true, y_log_pred))),
        "r2_price": float(r2_score(price_true, price_pred)),
        "mae_price": float(mean_absolute_error(price_true, price_pred)),
    }


def main():
    logging.info("Starting CatBoost (log price) training")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Same 80/20 split as the rest of the project (the 20% is the final hold-out)
    train_path, holdout_path = DataIngestion().initiate_data_ingestion()
    X_full, y_full = prepare_data(pd.read_csv(train_path))
    X_test, y_test = prepare_data(pd.read_csv(holdout_path))

    # A slice of the training part is used only to decide when to stop training
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.15, random_state=RANDOM_STATE)

    cat_cols = CATEGORICAL_COLS + (["location_cell"] if USE_LOCATION_CELL else [])

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_cols,
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5,
        early_stopping_rounds=100,
        use_best_model=True,
        allow_writing_files=False,      # no catboost_info folder
        thread_count=-1,
        verbose=200,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=200)

    val_scores = scores(y_val.to_numpy(), model.predict(X_val))
    test_scores = scores(y_test.to_numpy(), model.predict(X_test))

    metrics = {
        "model": "CatBoostRegressor",
        "target": "log_price",
        "location_cell": USE_LOCATION_CELL,
        "best_iteration": int(model.get_best_iteration()),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "validation": val_scores,
        "test": test_scores,
        "features": list(X_train.columns),
    }

    print("\n========== CATBOOST (log price) ==========")
    print(f"Best iteration: {metrics['best_iteration']}")
    for name, s in (("Validation", val_scores), ("Test (hold-out)", test_scores)):
        print(f"{name:16s} R2 log = {s['r2_log']:.4f} | RMSE log = {s['rmse_log']:.4f} | "
              f"R2 price = {s['r2_price']:.4f} | MAE = ${s['mae_price']:.2f}")

    model.save_model(MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved {MODEL_PATH} and {METRICS_PATH}")

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print("\n========== TOP 15 FEATURES ==========")
    print(importance.head(15).to_string(index=False))
    print(f"\nSaved model: {MODEL_PATH}\nSaved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
