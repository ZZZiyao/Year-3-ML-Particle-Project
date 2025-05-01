#%% 导入必要库
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
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}

lambda_values = [0, 0.01, 0.1, 1, 5, 10]  
alpha_values = [0, 0.01, 0.1, 1, 5, 10]  

heatmap_data = np.zeros((len(alpha_values), len(lambda_values)))

for i, reg_lambda in enumerate(lambda_values):
    for j, reg_alpha in enumerate(alpha_values):
        print(i,j)
        params = fixed_params.copy()
        params['reg_lambda'] = reg_lambda
        params['reg_alpha'] = reg_alpha

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain, 
                          num_boost_round=500, early_stopping_rounds=30, 
                          evals=[(dtest,'eval')],
                          verbose_eval=False)

        y_pred = model.predict(dtest)
        auc_score = roc_auc_score(y_val, y_pred)

        heatmap_data[j, i] = auc_score 


df = pd.DataFrame(heatmap_data, index=alpha_values, columns=lambda_values)
#%%
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="viridis", fmt=".5f")
plt.xlabel("Reg_lambda (L2 Regularization)")
plt.ylabel("Reg_alpha (L1 Regularization)")
plt.title("Effect of Reg_lambda and Reg_alpha on XGBoost Performance (AUC)")
plt.show()

#%% 
plt.figure(figsize=(10, 6))

for i, reg_lambda in enumerate(lambda_values):
    plt.plot(alpha_values, heatmap_data[:, i], marker='o', label=f"reg_lambda={reg_lambda}")

idx_lambda = lambda_values.index(0.1)
idx_alpha = alpha_values.index(10)
y_value = heatmap_data[idx_alpha, idx_lambda]  

plt.axhline(y=y_value, color='red', linestyle='--', label=f"optimized: reg_lambda=0.1, reg_alpha=10")
plt.axvline(x=10, color='red', linestyle='--')

plt.xlabel("Reg_alpha (L1 Regularization)")
plt.ylabel("AUC Score")
plt.title("Effect of Reg_alpha on AUC for Different Reg_lambda Values")
plt.legend(title="Reg Lambda")
plt.grid(True)

plt.show()

# %%
