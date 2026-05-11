Here is your GitHub README 👇

---

# Amazon Product Recommendation System

A production grade AI powered Recommendation System built on 568,000+ real Amazon product reviews. The system combines Collaborative Filtering, Content Based Filtering and a Hybrid Model to deliver personalized product recommendations with human readable explanations — deployed as a live interactive web application.

🔗 **Live App** → [[Add your Streamlit link here](https://recommendation-system-muslueqcfthbzlbzbdowwe.streamlit.app/)]

---

## 🎯 Business Problem

Amazon has millions of products and users cannot browse everything. Showing the wrong product means lost revenue. This system solves that by recommending the right product to the right user at the right time — increasing engagement, reducing churn and driving cross selling.

---

## 📊 Dataset

- Source — Amazon Product Reviews (Kaggle)
- Raw size — 568,454 reviews across 256,059 users and 74,258 products
- Filtered size — 47,380 reviews after minimum support filtering
- Matrix sparsity — 99.997% sparse before filtering

---

## 🤖 Models Built

### Model 1 — Collaborative Filtering (SVD)
- Decomposed user-item matrix using scipy svds with k=50 latent factors
- Mean centered ratings before decomposition to remove positivity bias
- Reconstructed predicted ratings for all unseen user-product pairs

### Model 2 — Content Based Filtering (TF-IDF + Cosine Similarity)
- Aggregated all reviews per product into combined text feature
- Applied TF-IDF with 5000 features and English stop words removed
- Computed cosine similarity between all product vectors

### Model 3 — Hybrid Model
- Combined SVD score (60%) and Content Based score (40%)
- Handles cold start problem automatically
- Falls back to Content Based for new users with no history

---

## 📈 Evaluation Results

| Metric | Score | Context |
|--------|-------|---------|
| Precision@10 | 0.1510 | 75x better than random |
| Recall@10 | 0.3009 | Finds 30% of relevant items |
| NDCG@10 | 0.3249 | Relevant items ranked near top |

---

## 💡 Key Features

- Explainability — every recommendation includes a human readable reason
- Cold Start handling — new users get content based recommendations
- Model Comparison — compare SVD vs Content Based vs Hybrid side by side
- Evaluation Dashboard — live metrics displayed in app

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Pandas, NumPy |
| Models | Scipy SVD, Scikit-learn TF-IDF, Cosine Similarity |
| Deployment | Streamlit Cloud |
| Model Storage | Hugging Face Datasets |
| Version Control | GitHub |

---

## 📁 Project Structure

```
recommendation-system/
├── app.py               ← Streamlit app (loads models from Hugging Face)
└── requirements.txt     ← Dependencies

Hugging Face → Vin1bun/recommendation-model
├── cosine_sim.pkl
├── df_clean.pkl
├── predicted_ratings_df.pkl
├── product_text.pkl
├── tfidf_matrix.pkl
├── tfidf_vectorizer.pkl
└── user_item_matrix.pkl
```

---

## 🚀 Try It Yourself

Open the live app and enter any of these User IDs 👇

```
A100WO06OQR8BQ
A101P2KHWCU0G6
A1051DBTLWP5A2
A10AFVU66A79Y1
A10H24TDLK2VDP
```

---

## 🔑 Key Learnings

- Sparsity handling using CSR Sparse Matrix saved 99% memory
- scipy SVD used instead of scikit-surprise due to NumPy 2.x compatibility issues
- Model files compressed from 358MB to 165MB using float16 and float32 quantization
- Hugging Face Datasets used for model hosting due to GitHub 25MB file limit

---

## 👤 Built By

**Vineet Prakash — Data Scientist**
📍 New Delhi, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/vineetprakash03)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/vin1bun)

*"AI can build, but only humans can innovate."*

---

Copy this into a **README.md** file and upload to your GitHub repo! 🚀
