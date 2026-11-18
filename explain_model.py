"""SHAP explanation of the saved model.

Run from the project root (after training):   python explain_model.py
Creates:  reports/shap_bar.png       (which inputs matter most)
          reports/shap_beeswarm.png  (how each input pushes the price up or down)

SHAP values are in log-price units: +0.10 means roughly 10% higher price.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # draw straight to files, no window
import matplotlib.pyplot as plt
import shap

from check_model import load_validation, D
from src.Airbnb.utils.utils import load_object

ENCODING = "ordinal"     # "ordinal" or "onehot": which saved pair to explain
N_EXPLAIN = 1000         # rows to explain (more = smoother plots, slower)
OUT = "reports"


def prepare():
    pre = load_object(os.path.join(D, f"Preprocessor_{ENCODING}.pkl"))
    model = load_object(os.path.join(D, f"Model_{ENCODING}.pkl"))

    X, _, _ = load_validation()
    sample = X.sample(min(N_EXPLAIN, len(X)), random_state=0)

    Xt = pre.transform(sample)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    return model, Xt, names


def main():
    model, Xt, names = prepare()
    print(f"Explaining {type(model).__name__} ({ENCODING} encoding) on {len(Xt)} listings...")

    try:
        explainer = shap.Explainer(model)                         # fast path for tree models (CatBoost, forests)
    except Exception:
        explainer = shap.Explainer(model.predict, Xt[:100])       # generic fallback (e.g. linear models)
    raw = explainer(Xt)

    sv = shap.Explanation(values=raw.values, base_values=raw.base_values,
                          data=Xt, feature_names=names)

    os.makedirs(OUT, exist_ok=True)

    shap.plots.bar(sv, max_display=15, show=False)
    plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.savefig(os.path.join(OUT, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.Series(np.abs(sv.values).mean(axis=0), index=names).sort_values(ascending=False)
    print("\nMean |SHAP| (effect on log price):")
    print(importance.head(12).round(4).to_string())
    print(f"\nPlots saved in the '{OUT}' folder.")


if __name__ == "__main__":
    main()
