# =============================================================
# L06 — HDB Resale Price Predictor (deployable Streamlit app)
#
# Highlights:
#   * Auto-trains the model on first run if none is saved (load_or_train).
#   * Caches the trained model so it loads instantly afterwards.
#   * Uses the richer feature set (flat type + town) chosen in model.py.
#
# Run locally:
#       pip install -r requirements.txt
#       streamlit run app.py
# =============================================================

import pandas as pd
import streamlit as st

from model import load_or_train

st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏡", layout="centered")


@st.cache_resource(show_spinner="Training the model (first run only)…")
def get_model():
    """Load the saved model, or train one on first run. Cached across reruns."""
    return load_or_train()


bundle = get_model()
model = bundle["model"]
model_columns = bundle["columns"]

# ---- Header ----
st.title("🏡 Singapore HDB Resale Price Predictor")
st.caption(
    f"Powered by a **{bundle['model_name']}** model trained on "
    f"{bundle['n_rows']:,} real resale transactions (2017 onwards)."
)

# ---- Model report card ----
with st.expander("📊 How accurate is this model?", expanded=False):
    st.metric("Average error (MAE)", f"S${bundle['mae']:,.0f}")
    st.write(
        "On average, the prediction is off by this much. We tested it on flats "
        "the model had never seen, so this is an honest estimate."
    )
    st.write("**Models compared during training:**")
    st.table(
        pd.DataFrame(
            [{"Model": n, "Average error (MAE)": f"S${m:,.0f}"}
             for n, m in bundle["all_scores"].items()]
        )
    )

# ---- Inputs (sidebar) ----
st.sidebar.header("Flat details")
sqm = st.sidebar.slider("Floor area (sqm)", 30, 160, 90, 1)
lease_year = st.sidebar.slider("Lease commencement year", 1970, 2025, 2000, 1)
floor = st.sidebar.slider("Floor level (storey)", 1, 50, 5, 1)
flat_type = st.sidebar.selectbox("Flat type", bundle["flat_types"],
                                 index=min(2, len(bundle["flat_types"]) - 1))
town = st.sidebar.selectbox("Town", bundle["towns"])

st.write("### Your flat")
c1, c2, c3 = st.columns(3)
c1.write(f"**Type:** {flat_type}")
c2.write(f"**Town:** {town}")
c3.write(f"**Size:** {sqm} sqm")

# ---- Prediction ----
if st.button("Predict resale price", type="primary"):
    row = pd.DataFrame([{
        "floor_area_sqm": sqm,
        "lease_commence_date": lease_year,
        "floor_level": floor,
        "flat_type": flat_type,
        "town": town,
    }])
    # Match the exact columns the model was trained on
    row = pd.get_dummies(row, columns=["flat_type", "town"])
    row = row.reindex(columns=model_columns, fill_value=0)

    price = model.predict(row)[0]
    mae = bundle["mae"]

    st.success(f"🇸🇬 Estimated resale price: **S${price:,.0f}**")
    st.caption(
        f"Likely range (±1 average error): "
        f"S${max(0, price - mae):,.0f} — S${price + mae:,.0f}"
    )
    st.info(
        "This is an estimate from a teaching model, not a valuation. "
        "Real prices also depend on renovation, exact location, and market timing."
    )

st.divider()
st.caption("Module 3 · Machine Learning & GenAI · L06 coaching project")
