import pandas as pd
from sklearn.preprocessing import StandardScaler


data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permutation_feature_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()
data = data[selected_features]

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


feature_variance = data_scaled.var()

low_variance_features = feature_variance[feature_variance < 0.01]

print("low variance features（variance < 0.01）(after normalization)：")
print(low_variance_features)

low_variance_features.to_csv("low_variance_features_scaled.csv", header=True)

