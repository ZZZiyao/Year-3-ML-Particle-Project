from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_data.csv'
data = pd.read_csv(data_path)

selected_features_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\permu_lowcorr_importance.csv'
selected_features = pd.read_csv(selected_features_path)['Feature'].tolist()

X = data[selected_features]
y = data['label']


model = RandomForestClassifier(random_state=42)


def custom_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]  
    auc = roc_auc_score(y, y_pred_proba)  
    loss = log_loss(y, y_pred_proba)  
    return auc, loss

auc_scores = []
log_loss_scores = []
num_features_list = []

# RFECV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selector = RFECV(estimator=model, step=1, cv=cv, scoring='roc_auc')
selector.fit(X, y)

for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    
    model.fit(X_train, y_train)
    
    auc, loss = custom_scorer(model, X_test, y_test)
    
    num_features = selector.n_features_  
    auc_scores.append(auc)
    log_loss_scores.append(loss)
    num_features_list.append(num_features)
    
    print(f"Iteration {i + 1}: Features = {num_features}, AUC = {auc:.4f}, Log Loss = {loss:.4f}")

results = pd.DataFrame({
    'Num Features': num_features_list,
    'AUC': auc_scores,
    'Log Loss': log_loss_scores
})


results.to_csv('rfe_cv_results.csv', index=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(num_features_list, auc_scores, marker='o', color='b', label='AUC')
plt.xlabel('Number of Features')
plt.ylabel('AUC')
plt.title('AUC vs Number of Features')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(num_features_list, log_loss_scores, marker='o', color='r', label='Log Loss')
plt.xlabel('Number of Features')
plt.ylabel('Log Loss')
plt.title('Log Loss vs Number of Features')
plt.grid(True)

plt.tight_layout()
plt.show()