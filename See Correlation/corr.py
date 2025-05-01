import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path, low_memory=False)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permutation_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()
data = data[selected_features]


low_variance_features = [
    "mu_minus_ProbNNd", "mu_plus_M", "pion_ProbNNd",
    "mu_plus_ProbNNd", "pion_M", "B0_ENDVERTEX_NDOF",
    "kaon_ProbNNd", "kaon_M", "mu_minus_M"
]
data_filtered = data.drop(columns=low_variance_features, errors="ignore")

corr_matrix = data_filtered.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = [column for column in upper.columns if any(upper[column] > 0.9)]

print("Highly correlated features（> 0.9）：")
for feature in high_corr_features:
    max_corr = upper[feature].max()  
    related_feature = upper[feature].idxmax()  
    print(f"❌ {feature} (max correlation: {max_corr:.4f}，correlated with {related_feature})")

#corr_matrix.to_csv("feature_correlation_matrix.csv")
#print("saved to feature_correlation_matrix.csv")
