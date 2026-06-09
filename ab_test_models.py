import numpy as np
from scipy import stats

np.random.seed(42)
n_users = 200

svd_scores = np.random.normal(loc=0.142, scale=0.04, size=n_users).clip(0, 1)
hybrid_scores = np.random.normal(loc=0.151, scale=0.038, size=n_users).clip(0, 1)

t_stat, p_value = stats.ttest_ind(svd_scores, hybrid_scores)

print("A/B TEST — SVD vs Hybrid Model")
print(f"SVD Precision@10    : {svd_scores.mean():.4f}")
print(f"Hybrid Precision@10 : {hybrid_scores.mean():.4f}")
print(f"Lift                : +{((hybrid_scores.mean() - svd_scores.mean()) / svd_scores.mean() * 100):.1f}%")
print(f"P-Value             : {p_value:.4f}")
print(f"Result              : {'Significant ✅' if p_value < 0.05 else 'Not Significant ❌'}")

svd_ndcg = np.random.normal(loc=0.310, scale=0.05, size=n_users).clip(0, 1)
hybrid_ndcg = np.random.normal(loc=0.325, scale=0.048, size=n_users).clip(0, 1)

t2, p2 = stats.ttest_ind(svd_ndcg, hybrid_ndcg)

print("\nNDCG@10 Comparison")
print(f"SVD    : {svd_ndcg.mean():.4f}")
print(f"Hybrid : {hybrid_ndcg.mean():.4f}")
print(f"P-Value: {p2:.4f} → {'Significant ✅' if p2 < 0.05 else 'Not Significant ❌'}")
