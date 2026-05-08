
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ──
st.set_page_config(
    page_title = "Recommendation System | Vineet Prakash",
    page_icon  = "🛒",
    layout     = "wide"
)

# ── Load Models ──
@st.cache_resource
def load_models():
    with open("recommendation_model/user_item_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)
    with open("recommendation_model/predicted_ratings_df.pkl", "rb") as f:
        predicted_ratings_df = pickle.load(f)
    with open("recommendation_model/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("recommendation_model/cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    with open("recommendation_model/product_text.pkl", "rb") as f:
        product_text = pickle.load(f)
    with open("recommendation_model/df_clean.pkl", "rb") as f:
        df_clean = pickle.load(f)
    return user_item_matrix, predicted_ratings_df, tfidf_matrix, cosine_sim, product_text, df_clean

user_item_matrix, predicted_ratings_df, tfidf_matrix, cosine_sim, product_text, df_clean = load_models()

# ── Product Index ──
product_indices = pd.Series(product_text.index, index=product_text["ProductId"])

# ── Recommendation Functions ──
def get_svd_recommendations(user_id, n=10):
    if user_id not in predicted_ratings_df.index:
        return None
    user_predictions = predicted_ratings_df.loc[user_id]
    rated_products   = user_item_matrix.loc[user_id]
    rated_products   = rated_products[rated_products > 0].index
    recs = user_predictions.drop(rated_products).sort_values(ascending=False).head(n)
    return pd.DataFrame({"ProductId": recs.index, "Predicted Rating": recs.values.round(2)})

def get_content_recommendations(product_id, n=10):
    if product_id not in product_indices:
        return None
    idx        = product_indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    product_idx    = [i[0] for i in sim_scores]
    product_scores = [round(i[1], 4) for i in sim_scores]
    return pd.DataFrame({"ProductId": product_text["ProductId"].iloc[product_idx].values,
                         "Similarity Score": product_scores})

def get_hybrid_recommendations(user_id, n=10):
    if user_id not in predicted_ratings_df.index:
        return None
    user_predictions = predicted_ratings_df.loc[user_id]
    rated_products   = user_item_matrix.loc[user_id]
    rated_products   = rated_products[rated_products > 0].index
    svd_recs         = user_predictions.drop(rated_products).sort_values(ascending=False).head(n*2)
    svd_scores       = (svd_recs - svd_recs.min()) / (svd_recs.max() - svd_recs.min())
    top_product      = user_item_matrix.loc[user_id].sort_values(ascending=False).index[0]
    content_recs     = get_content_recommendations(top_product, n=n*2)
    if content_recs is None:
        return get_svd_recommendations(user_id, n)
    content_scores   = content_recs.set_index("ProductId")["Similarity Score"]
    hybrid_scores    = {}
    for product in svd_scores.index:
        hybrid_scores[product] = (0.6 * svd_scores.get(product, 0)) + (0.4 * content_scores.get(product, 0))
    hybrid_df = pd.DataFrame.from_dict(hybrid_scores, orient="index", columns=["Hybrid Score"])
    hybrid_df = hybrid_df.sort_values("Hybrid Score", ascending=False).head(n)
    hybrid_df.index.name = "ProductId"
    return hybrid_df.reset_index()

# ── Sidebar ──
st.sidebar.image("https://avatars.githubusercontent.com/u/vin1bun", width=120)
st.sidebar.title("Vineet Prakash")
st.sidebar.markdown("**Data Scientist**")
st.sidebar.markdown("📍 New Delhi, India")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/vineetprakash03)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/vin1bun)")
st.sidebar.markdown("---")
st.sidebar.markdown("*AI can build, but only humans can innovate*")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Navigate", ["🏠 Home", "🤖 Get Recommendations", "📊 Model Comparison", "📈 Evaluation"])

# ══════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛒 Amazon Product Recommendation System")
    st.markdown("##### Built by **Vineet Prakash** | Data Scientist")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users",    f"{user_item_matrix.shape[0]:,}")
    col2.metric("Total Products", f"{user_item_matrix.shape[1]:,}")
    col3.metric("Total Reviews",  f"{df_clean.shape[0]:,}")
    col4.metric("Models Built",   "3")

    st.markdown("---")
    st.subheader("🧠 About This Project")
    st.markdown("""
    This recommendation system is built on **Amazon Product Reviews** with 3 models:
    - **Collaborative Filtering (SVD)** — Recommends based on similar users
    - **Content Based Filtering** — Recommends based on product similarity
    - **Hybrid Model** — Combines both for best results
    """)

    st.subheader("🛠️ Tech Stack")
    col1, col2, col3 = st.columns(3)
    col1.info("**Data**\nPandas · NumPy · Scipy")
    col2.info("**Models**\nSVD · TF-IDF · Cosine Similarity")
    col3.info("**Deployment**\nStreamlit · Pickle")

    st.markdown("---")
    st.markdown("*Built with ❤️ by Vineet Prakash | [LinkedIn](https://linkedin.com/in/vineetprakash03) | [GitHub](https://github.com/vin1bun)*")

# ══════════════════════════════════════════
# PAGE 2 — GET RECOMMENDATIONS
# ══════════════════════════════════════════
elif page == "🤖 Get Recommendations":
    st.title("🤖 Get Product Recommendations")
    st.markdown("---")

    user_id = st.text_input("Enter User ID", placeholder="e.g. A100WO06OQR8BQ")
    n_recs  = st.slider("Number of Recommendations", 5, 20, 10)
    model   = st.selectbox("Select Model", ["Hybrid (Recommended)", "SVD Only", "Content Based Only"])

    if st.button("🚀 Get Recommendations"):
        if user_id:
            if model == "Hybrid (Recommended)":
                recs = get_hybrid_recommendations(user_id, n=n_recs)
            elif model == "SVD Only":
                recs = get_svd_recommendations(user_id, n=n_recs)
            else:
                rated   = user_item_matrix.loc[user_id] if user_id in user_item_matrix.index else None
                if rated is not None:
                    top_product = rated[rated > 0].sort_values(ascending=False).index[0]
                    recs = get_content_recommendations(top_product, n=n_recs)
                else:
                    recs = None

            if recs is not None:
                st.success(f"Top {n_recs} recommendations for User: **{user_id}**")
                st.dataframe(recs, use_container_width=True)

                # Show user history
                if user_id in user_item_matrix.index:
                    st.subheader("📋 User Rating History")
                    history = user_item_matrix.loc[user_id]
                    history = history[history > 0].sort_values(ascending=False).head(5)
                    st.dataframe(pd.DataFrame({"ProductId": history.index,
                                               "Rating Given": history.values}),
                                 use_container_width=True)
            else:
                st.error("User not found. Please try another User ID.")
        else:
            st.warning("Please enter a User ID.")

# ══════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Compare All 3 Models")
    st.markdown("---")

    user_id = st.text_input("Enter User ID to Compare", placeholder="e.g. A100WO06OQR8BQ")

    if st.button("🔍 Compare Models"):
        if user_id and user_id in predicted_ratings_df.index:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🤖 SVD Model")
                svd_recs = get_svd_recommendations(user_id, n=5)
                if svd_recs is not None:
                    st.dataframe(svd_recs, use_container_width=True)

            with col2:
                st.subheader("📄 Content Based")
                rated       = user_item_matrix.loc[user_id]
                top_product = rated[rated > 0].sort_values(ascending=False).index[0]
                content_recs = get_content_recommendations(top_product, n=5)
                if content_recs is not None:
                    st.dataframe(content_recs, use_container_width=True)

            with col3:
                st.subheader("🔀 Hybrid Model")
                hybrid_recs = get_hybrid_recommendations(user_id, n=5)
                if hybrid_recs is not None:
                    st.dataframe(hybrid_recs, use_container_width=True)
        else:
            st.error("User not found. Please try another User ID.")

# ══════════════════════════════════════════
# PAGE 4 — EVALUATION
# ══════════════════════════════════════════
elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation Dashboard")
    st.markdown("---")

    st.subheader("📊 SVD Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@10", "0.1510")
    col2.metric("Recall@10",    "0.3009")
    col3.metric("NDCG@10",      "0.3249")

    st.markdown("---")
    st.subheader("🧠 What These Metrics Mean")
    st.markdown("""
    | Metric | Score | Meaning |
    |--------|-------|---------|
    | Precision@10 | 0.1510 | 1-2 out of every 10 recommendations are relevant |
    | Recall@10 | 0.3009 | Model finds 30% of all products user would like |
    | NDCG@10 | 0.3249 | Relevant products are ranked near the top |
    """)

    st.markdown("---")
    st.markdown("*Built with ❤️ by Vineet Prakash | [LinkedIn](https://linkedin.com/in/vineetprakash03) | [GitHub](https://github.com/vin1bun)*")
