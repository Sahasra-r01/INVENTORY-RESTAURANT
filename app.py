"""
Restaurant Inventory Assistant - Telangana
Web app for the stock-out & lead-time models.

Run:   streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

import predictor as P

st.set_page_config(page_title="Restaurant Inventory Assistant", page_icon="📦", layout="wide")


@st.cache_resource
def get_bundle():
    return P.load_bundle()


@st.cache_data
def get_data():
    return P.load_data()


try:
    bundle, data = get_bundle(), get_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

defaults = P.item_defaults(data)

st.title("📦 Restaurant Inventory Assistant")
st.caption("Predicts supplier lead time and stock-out risk for restaurants across Telangana.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict an item", "🚨 At-risk dashboard", "📊 Model performance"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Check one inventory item")
    c1, c2 = st.columns(2)
    with c1:
        item = st.selectbox("Item", sorted(defaults.index), index=0)
        d = defaults.loc[item]
        city = st.selectbox("City", sorted(data["City"].unique()))
        cuisine = st.selectbox("Cuisine type", sorted(data["Cuisine Type"].unique()))
        st.text_input("Category (auto)", d["Category"], disabled=True)
        st.text_input("Unit (auto)", d["Unit"], disabled=True)
    with c2:
        stock = st.number_input("Current stock", min_value=0.0, value=float(d["Daily Consumption Rate"] * 3), step=1.0)
        reorder = st.number_input("Reorder level", min_value=0.0, value=float(d["Daily Consumption Rate"] * 2), step=1.0)
        price = st.number_input("Unit price (INR)", min_value=0.0, value=float(d["Unit Price (INR)"]), step=1.0)
        cons = st.number_input("Daily consumption", min_value=0.1, value=float(d["Daily Consumption Rate"]), step=0.5)
        shelf = st.number_input("Shelf life (days)", min_value=1, value=int(d["Shelf Life (Days)"]), step=1)
    threshold = st.slider("Alert threshold (stock-out probability)", 0.1, 0.9, 0.5, 0.05,
                          help="Lower = more cautious (more alerts).")

    if st.button("Predict", type="primary"):
        r = P.predict_item(bundle, city, cuisine, d["Category"], d["Unit"], item, stock, reorder,
                           price, cons, shelf, threshold)
        if r["at_risk"]:
            st.error(f"🚨 STOCK-OUT RISK — {item} may run out before the next delivery arrives.")
        else:
            st.success(f"✅ {item} looks safe.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted lead time", f"{r['lead_time']:.1f} days")
        m2.metric("Days until stock-out", f"{r['days_of_cover']:.1f} days")
        m3.metric("Stock-out probability", f"{r['probability']:.0%}")
        m4.metric("Suggested order", f"{r['order_qty']:.0f} {d['Unit']}",
                  help=f"Covers lead time + {bundle['safety_buffer_days']} safety day(s). Est. cost ₹{r['order_cost']:,.0f}")
        st.progress(min(r["probability"], 1.0), text="Stock-out probability")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Items currently at risk")
    scored = P.score_frame(data, bundle)
    f1, f2, f3 = st.columns(3)
    cities = f1.multiselect("City", sorted(data["City"].unique()))
    cats = f2.multiselect("Category", sorted(data["Category"].unique()))
    thr = f3.slider("Minimum probability", 0.1, 1.0, 0.5, 0.05)

    view = scored[scored["Stockout Probability"] >= thr]
    if cities:
        view = view[view["City"].isin(cities)]
    if cats:
        view = view[view["Category"].isin(cats)]
    view = view.sort_values("Stockout Probability", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Items at risk", len(view))
    k2.metric("Share of all items", f"{len(view) / len(scored):.1%}")
    k3.metric("Restaurants affected", view["Restaurant Name"].nunique())

    if len(view):
        st.write("**At-risk items by category**")
        st.bar_chart(view["Category"].value_counts())
        show = view[["Restaurant Name", "City", "Item Name", "Category", "Current Stock",
                     "Daily Consumption Rate", "Days of Cover", "Pred Lead Time",
                     "Stockout Probability"]].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download as CSV", show.to_csv(index=False).encode(), "items_at_risk.csv", "text/csv")
    else:
        st.info("No items match these filters.")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("How well do the models perform?")
    here = os.path.dirname(os.path.abspath(__file__))
    for title, csv in [("Lead-time model (regression)", "leadtime_model_comparison.csv"),
                       ("Stock-out model (classification)", "stockout_model_comparison.csv")]:
        p = os.path.join(here, csv)
        if os.path.exists(p):
            st.write(f"**{title}**")
            st.dataframe(pd.read_csv(p).round(3), hide_index=True, use_container_width=True)
    cols = st.columns(2)
    for col, img in zip(cols, ["leadtime_results.png", "stockout_results.png"]):
        p = os.path.join(here, img)
        if os.path.exists(p):
            col.image(p, use_container_width=True)
    st.caption("Stock-out risk is derived: Days of Cover < Lead Time + safety buffer. "
               "Lead time is predicted from item, category, price, consumption and shelf life.")
