import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib  

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data.csv'
data = pd.read_csv(data_path)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].head(52).tolist()

X = data[selected_features]
y = data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


best_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
     'max_depth': 8, 
     'eta': 0.07, 
    'lambda': 0.1, 
    'alpha': 10, 
    'min_child_weight': 1, 
    'gamma': 0.01,
    'n_jobs': -1
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
best_model = None
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Training Fold {fold+1}/5...")

    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

    model = xgb.train(best_params, dtrain, num_boost_round=500, 
                      evals=[(dtrain, 'train'), (dval, 'eval')],
                      early_stopping_rounds=30, verbose_eval=False)

    y_val_pred = model.predict(dval)
    fold_auc = roc_auc_score(y_val_fold, y_val_pred)
    auc_scores.append(fold_auc)

    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    if fold_auc > best_auc:
        best_auc = fold_auc
        best_model = model  

mean_auc = np.mean(auc_scores)
print(f"5-Fold CV AUC: {mean_auc:.5f}")

# 6. 在整个训练集上训练最终模型
print("Training Final Model on Full Training Data...")
dtrain_final = xgb.DMatrix(X_train, label=y_train)
dval_final = xgb.DMatrix(X_val, label=y_val)

final_model = xgb.train(best_params, dtrain_final, num_boost_round=500, 
                        evals=[(dtrain_final, 'train'), (dval_final, 'eval')],
                        early_stopping_rounds=30, verbose_eval=False)

y_val_final_pred = final_model.predict(dval_final)
final_auc = roc_auc_score(y_val, y_val_final_pred)
print(f"Final Model AUC on Validation Set: {final_auc:.5f}")

y_val_final_pred_class = (y_val_final_pred > 0.5).astype(int)  
print("\nClassification Report on Validation Set:")
print(classification_report(y_val, y_val_final_pred_class))

model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\new_model2.pkl"
joblib.dump(final_model, model_path)
