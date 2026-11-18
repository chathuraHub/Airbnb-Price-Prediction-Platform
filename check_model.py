"""Re-check the SAVED models on the validation split. No retraining, takes seconds.

Run from the project root:   python check_model.py
Prints, for each saved encoding (ordinal / onehot):
  - which model was saved (the best one from training)
  - R2 and RMSE on log(price)     <- the scale the model is trained on
  - R2 and MAE on the real price  <- converted back with exp()
"""
import os
import csv
import io
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.Airbnb.utils.utils import load_object

D = "airbnb-price-prediction"
VAL_PATH = os.path.join(D, "val_split.csv")

# Same columns that Data_transformation removes before the preprocessor
DROP_COLUMNS = ["log_price", "price", "id", "name", "description", "first_review",
                "host_since", "last_review", "neighbourhood", "thumbnail_url", "zipcode"]


def _count_amenities(value):
    """Same logic as Data_transformation: count items, respecting quotes."""
    if pd.isna(value) or str(value).strip() in ("", "{}"):
        return np.nan
    value = str(value).strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    items = next(csv.reader(io.StringIO(value), skipinitialspace=True))
    return sum(1 for item in items if item.strip())


def load_validation():
    """Returns (features, y_log_price, y_price) from val_split.csv, cleaned like training."""
    df = pd.read_csv(VAL_PATH)

    y_log = df["log_price"] if "log_price" in df.columns else np.log(df["price"])
    y_price = df["price"] if "price" in df.columns else np.exp(y_log)

    df["cleaning_fee"] = df["cleaning_fee"].astype(str).str.upper()
    df["host_response_rate"] = pd.to_numeric(
        df["host_response_rate"].astype(str).str.replace("%", "", regex=False),
        errors="coerce")
    df["amenities"] = df["amenities"].apply(_count_amenities)

    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return X, y_log.to_numpy(), y_price.to_numpy()


def main():
    X, y_log, y_price = load_validation()
    print(f"Validation rows: {len(X)}\n")

    rows = []
    for encoding in ["ordinal", "onehot"]:
        pre_path = os.path.join(D, f"Preprocessor_{encoding}.pkl")
        model_path = os.path.join(D, f"Model_{encoding}.pkl")
        if not (os.path.exists(pre_path) and os.path.exists(model_path)):
            print(f"[{encoding}] saved files not found - run the training pipeline first.")
            continue

        pre = load_object(pre_path)
        model = load_object(model_path)
        pred_log = model.predict(pre.transform(X))
        pred_price = np.exp(pred_log)

        rows.append({
            "encoding": encoding,
            "saved model": type(model).__name__,
            "R2 (log price)": round(r2_score(y_log, pred_log), 4),
            "RMSE (log price)": round(float(np.sqrt(mean_squared_error(y_log, pred_log))), 4),
            "R2 (real price)": round(r2_score(y_price, pred_price), 4),
            "MAE ($)": round(mean_absolute_error(y_price, pred_price), 2),
        })

    if rows:
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        best = table.sort_values("R2 (log price)", ascending=False).iloc[0]
        print(f"\nBest saved pair: {best['saved model']} with {best['encoding']} encoding "
              f"(R2 on log price = {best['R2 (log price)']}).")
        print("To make the web app use it, set the encoding in app.py:  PredictPipeline(encoding=\"%s\")"
              % best["encoding"])


if __name__ == "__main__":
    main()
