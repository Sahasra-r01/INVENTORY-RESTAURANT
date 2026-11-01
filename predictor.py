"""Prediction logic for the web app (no Streamlit code here, so it can be tested on its own)."""
import os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "telangana_restaurant_inventory_1200.xlsx"
MODEL_FILE = "stockout_leadtime_models.pkl"
LEAD = "Supplier Lead Time (Days)"


def _find(name):
    for p in [os.path.join(HERE, name), name]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"'{name}' not found next to app.py. Run stockout_leadtime_model.py first "
                            f"(it creates the .pkl) and keep the Excel file in the same folder.")


def load_bundle():
    return joblib.load(_find(MODEL_FILE))


def load_data():
    return pd.read_excel(_find(DATA_FILE))


def add_stockout_features(d):
    """Must match stockout_features() in stockout_leadtime_model.py exactly."""
    d = d.copy()
    d["Days of Cover"] = d["Current Stock"] / d["Daily Consumption Rate"].clip(lower=0.01)
    d["Cover minus Lead"] = d["Days of Cover"] - d["Pred Lead Time"]
    d["Demand During Lead"] = d["Daily Consumption Rate"] * d["Pred Lead Time"]
    d["Stock vs Lead Demand"] = d["Current Stock"] / d["Demand During Lead"].clip(lower=0.01)
    return d


def score_frame(df, bundle):
    """Add predicted lead time + stock-out probability to a DataFrame of items."""
    out = df.copy()
    out["Pred Lead Time"] = bundle["lead_time_model"].predict(out[bundle["lead_time_features"]])
    out = add_stockout_features(out)
    out["Stockout Probability"] = bundle["stockout_model"].predict_proba(out[bundle["stockout_features"]])[:, 1]
    return out


def predict_item(bundle, city, cuisine_type, category, unit, item_name, current_stock,
                 reorder_level, unit_price, daily_consumption, shelf_life, threshold=0.5):
    row = pd.DataFrame([{
        "City": city, "Cuisine Type": cuisine_type, "Category": category, "Unit": unit,
        "Item Name": item_name, "Current Stock": current_stock, "Reorder Level": reorder_level,
        "Unit Price (INR)": unit_price, "Daily Consumption Rate": daily_consumption,
        "Shelf Life (Days)": shelf_life}])
    r = score_frame(row, bundle).iloc[0]
    lead, cover, prob = float(r["Pred Lead Time"]), float(r["Days of Cover"]), float(r["Stockout Probability"])
    buffer = bundle["safety_buffer_days"]
    order_qty = float(max(0.0, daily_consumption * (np.ceil(lead) + buffer) - current_stock))
    return {"lead_time": lead, "days_of_cover": cover, "probability": prob,
            "at_risk": prob >= threshold, "order_qty": order_qty,
            "order_cost": order_qty * unit_price}


def item_defaults(df):
    """Per-item defaults (category, unit, typical price/shelf life/consumption) to auto-fill the form."""
    g = df.groupby("Item Name")
    return pd.DataFrame({
        "Category": g["Category"].agg(lambda s: s.mode()[0]),
        "Unit": g["Unit"].agg(lambda s: s.mode()[0]),
        "Unit Price (INR)": g["Unit Price (INR)"].median(),
        "Shelf Life (Days)": g["Shelf Life (Days)"].median(),
        "Daily Consumption Rate": g["Daily Consumption Rate"].median(),
    })
