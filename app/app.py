
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

PAGES = ["🏠 Home", "🔍 Product Recommender", "👤 Customer Segmentation"]
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(base_path)
    models = {}
    models["kmeans"] = joblib.load(os.path.join(parent_path, "models", "kmeans_model.pkl")) if os.path.exists(os.path.join(parent_path, "models", "kmeans_model.pkl")) else None
    models["scaler"] = joblib.load(os.path.join(parent_path, "models", "scaler.pkl")) if os.path.exists(os.path.join(parent_path, "models", "scaler.pkl")) else None
    models["similarity_df"] = joblib.load(os.path.join(parent_path, "models", "product_similarity.pkl")) if os.path.exists(os.path.join(parent_path, "models", "product_similarity.pkl")) else None
    models["product_list"] = joblib.load(os.path.join(parent_path, "models", "product_names.pkl")) if os.path.exists(os.path.join(parent_path, "models", "product_names.pkl")) else None
    seg_path = os.path.join(parent_path, "outputs", "segment_mapping.csv")
    models["segment_mapping"] = pd.read_csv(seg_path) if os.path.exists(seg_path) else None
    return models

models = load_models()

def get_recommendations(product_name, top_n=5):
    similarity_df = models["similarity_df"]
    product_list = models["product_list"]
    if similarity_df is None or product_list is None:
        return None, "Models not loaded."
    if product_name in similarity_df.index:
        matched = product_name
    else:
        matches = [p for p in product_list if product_name.lower() in p.lower()]
        if not matches:
            return None, f"No product found matching '{product_name}'."
        matched = matches[0]
    sim_scores = similarity_df[matched].drop(matched)
    top_products = sim_scores.sort_values(ascending=False).head(top_n)
    return [(prod, round(score, 4)) for prod, score in top_products.items()], matched

def predict_segment(recency, frequency, monetary):
    kmeans = models["kmeans"]
    scaler = models["scaler"]
    seg_map = models["segment_mapping"]
    if kmeans is None or scaler is None or seg_map is None:
        return None, "Models not loaded."
    freq_log = np.log1p(frequency)
    mon_log = np.log1p(monetary)
    X = np.array([[recency, freq_log, mon_log]])
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]
    segment = seg_map[seg_map["Cluster"] == cluster]["Segment"].values[0]
    return segment, cluster

with st.sidebar:
    st.title("🛒 Shopper Spectrum")
    st.markdown("---")
    selected_page = st.radio("Navigate to:", PAGES, index=PAGES.index(st.session_state.current_page))
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

page = st.session_state.current_page

if page == "🏠 Home":
    st.title("🛒 Welcome to Shopper Spectrum")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Product Recommender")
        st.markdown("Enter a product name to get 5 similar recommendations.")
        if st.button("Go to Product Recommender →", key="btn_prod"):
            st.session_state.current_page = "🔍 Product Recommender"
            st.rerun()
    with col2:
        st.subheader("👤 Customer Segmentation")
        st.markdown("Enter RFM values to predict customer segment.")
        if st.button("Go to Customer Segmentation →", key="btn_seg"):
            st.session_state.current_page = "👤 Customer Segmentation"
            st.rerun()

elif page == "🔍 Product Recommender":
    st.title("🔍 Product Recommendation")
    if models["similarity_df"] is None:
        st.error("❌ Product similarity model not found.")
    else:
        product_input = st.text_input("Product Name", placeholder="e.g., WHITE HANGING HEART...")
        if st.button("🎯 Get Recommendations", type="primary"):
            if not product_input.strip():
                st.warning("Please enter a product name.")
            else:
                with st.spinner("Finding similar products..."):
                    recs, matched = get_recommendations(product_input.strip(), top_n=5)
                if recs is None:
                    st.error(f"❌ {matched}")
                else:
                    st.success(f"✅ Recommendations for: {matched}")
                    for i, (prod, score) in enumerate(recs, 1):
                        st.markdown(f"{i}. **{prod}** — Similarity: `{score}`")

elif page == "👤 Customer Segmentation":
    st.title("👤 Customer Segment Predictor")
    if models["kmeans"] is None or models["scaler"] is None:
        st.error("❌ Segmentation models not found.")
    else:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("📊 Input RFM Values")
            recency = st.number_input("Recency (days)", min_value=0, max_value=1000, value=30)
            frequency = st.number_input("Frequency (purchases)", min_value=1, max_value=500, value=5)
            monetary = st.number_input("Monetary (£)", min_value=0.0, max_value=50000.0, value=500.0)
            if st.button("🎯 Predict Segment", type="primary"):
                with st.spinner("Analyzing..."):
                    segment, cluster = predict_segment(recency, frequency, monetary)
                if segment is None:
                    st.error(f"❌ {cluster}")
                else:
                    st.session_state["prediction"] = {"segment": segment, "cluster": cluster, "recency": recency, "frequency": frequency, "monetary": monetary}
        with col2:
            st.subheader("📈 Prediction Result")
            if "prediction" in st.session_state:
                pred = st.session_state["prediction"]
                st.success(f"Predicted Segment: **{pred['segment']}**")
                st.info(f"Cluster ID: {pred['cluster']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Recency", f"{pred['recency']} days")
                c2.metric("Frequency", f"{pred['frequency']} purchases")
                c3.metric("Monetary", f"£{pred['monetary']:,.2f}")
            else:
                st.info("👈 Enter RFM values and click Predict Segment.")

st.markdown("---")
st.caption("🛒 Shopper Spectrum | Customer Segmentation & Product Recommendation")
