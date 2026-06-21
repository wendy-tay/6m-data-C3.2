# =============================================================
# L06 — Better, Trustworthy HDB Price Model
# Training logic (shared by the Streamlit app and command line).
#
# What changed from L01-L05:
#   1. We add MORE features (flat_type, town) on top of the original 3.
#   2. We MEASURE how good the model is using MAE (average dollar error).
#   3. We COMPARE two models and keep the better one.
#
# Run standalone to (re)train and save the model file:
#       python model.py
#
# The Streamlit app (app.py) imports load_or_train() so it can train
# automatically on first run if no saved model is present.
# =============================================================

import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_URL = (
    "https://raw.githubusercontent.com/kohjiaxuan/"
    "Predicting-HDB-Price-with-Machine-Learning/master/"
    "resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_model.pkl")

NUMERIC_FEATURES = ["floor_area_sqm", "lease_commence_date", "floor_level"]
CATEGORY_FEATURES = ["flat_type", "town"]


def load_data():
    """Download the live HDB resale dataset and add a numeric floor column."""
    data = pd.read_csv(DATA_URL)
    # Turn 'storey_range' (e.g. "10 TO 12") into a number (the lower bound)
    data["floor_level"] = data["storey_range"].str.split(" ").str[0].astype(float)
    return data


def train_model():
    """Train, compare two models, and return the best one as a bundle dict."""
    data = load_data()

    X = pd.get_dummies(
        data[NUMERIC_FEATURES + CATEGORY_FEATURES], columns=CATEGORY_FEATURES
    )
    y = data["resale_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        scores[name] = (model, mae)

    best_name = min(scores, key=lambda n: scores[n][1])
    best_model, best_mae = scores[best_name]

    return {
        "model": best_model,
        "columns": list(X.columns),
        "model_name": best_name,
        "mae": best_mae,
        "all_scores": {n: s[1] for n, s in scores.items()},
        "flat_types": sorted(data["flat_type"].unique().tolist()),
        "towns": sorted(data["town"].unique().tolist()),
        "n_rows": len(data),
    }


def save_model(bundle, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_or_train(path=MODEL_PATH):
    """Load a saved model, or train and save one if none exists yet.

    This is what makes the app 'deployable when trained': the first time
    the app runs on a fresh server it trains the model, then reuses it.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    bundle = train_model()
    save_model(bundle, path)
    return bundle


if __name__ == "__main__":
    print("Downloading data and training models...")
    bundle = train_model()
    for name, mae in bundle["all_scores"].items():
        print(f"  {name:18s} -> on average off by S${mae:,.0f}")
    print(f"\nWinner: {bundle['model_name']} (lowest dollar error)")
    save_model(bundle)
    print(f"Saved model to {MODEL_PATH}")
