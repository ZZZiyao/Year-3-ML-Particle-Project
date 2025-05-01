#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

selected_features_path=r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'

#selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permu_lowcorr_importance.csv'
feature_importance_list = pd.read_csv(selected_features_path)['Feature'].head(43).tolist()

X = data[feature_importance_list]
y = data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

fixed_params = {
    'max_depth': 8,
    'learning_rate': 0.07,
    'subsample': 1.0,  
    'colsample_bytree': 1.0,  
    'reg_lambda': 0.1,  
    'reg_alpha': 10,  
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}

gamma_values = [0.01, 0.1, 1, 5, 10]  
min_child_weight_values = [1, 3, 5, 7, 10] 

heatmap_data = np.zeros((len(min_child_weight_values), len(gamma_values)))

for i, gamma in enumerate(gamma_values):
    for j, min_child_weight in enumerate(min_child_weight_values):
        print(f"Testing gamma={gamma}, min_child_weight={min_child_weight}")
        params = fixed_params.copy()
        params['gamma'] = gamma
        params['min_child_weight'] = min_child_weight

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain, 
                          num_boost_round=500, early_stopping_rounds=30, 
                          evals=[(dtest,'eval')],
                          verbose_eval=False)

        y_pred = model.predict(dtest)
        auc_score = roc_auc_score(y_val, y_pred)

        heatmap_data[j, i] = auc_score  


df = pd.DataFrame(heatmap_data, index=min_child_weight_values, columns=gamma_values)

#%% 
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="viridis", fmt=".5f")
plt.xlabel("Gamma (Pruning Parameter)")
plt.ylabel("Min_child_weight (Leaf Node Min Weight)")
plt.title("Effect of Gamma and Min_child_weight on XGBoost Performance (AUC)")
plt.show()

#%% 
plt.figure(figsize=(10, 6))

for i, gamma in enumerate(gamma_values):
    plt.plot(min_child_weight_values, heatmap_data[:, i], marker='o', label=f"gamma={gamma}")

idx_gamma = gamma_values.index(0.01)
idx_min_child_weight = min_child_weight_values.index(1)
y_value = heatmap_data[idx_min_child_weight, idx_gamma]  

plt.axhline(y=y_value, color='red', linestyle='--', label=f"optimized: gamma=0.01, min_child_weight=1")
plt.axvline(x=1, color='red', linestyle='--')

plt.xlabel("Min_child_weight (Leaf Node Min Weight)")
plt.ylabel("AUC Score")
plt.title("Effect of Min_child_weight on AUC for Different Gamma Values")
plt.legend(title="Gamma")
plt.grid(True)

plt.show()

# %%
