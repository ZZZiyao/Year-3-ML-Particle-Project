#%%
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
import pickle


with open(r"D:\Year3\BSc Project\Particle-Machine-Learning\Models\more_model1.pkl", "rb") as f:
    booster = pickle.load(f)

print(booster.feature_names)
#%%
with open(r"D:\Year3\BSc Project\Particle-Machine-Learning\Models\more_model1.pkl", "rb") as f:
    booster = pickle.load(f)

with open(r"D:\Year3\BSc Project\Particle-Machine-Learning\final_feature_list.txt", "r") as f:
    lines = f.readlines()
feature_list = [line.strip() for line in lines if line.strip() != "Feature"]

data = pd.read_csv(r"D:\Year3\BSc Project\Particle-Machine-Learning\balanced_final.csv")

label_col = "label"  

X = data.drop(columns=[label_col])
y = data[label_col]
#%%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
accuracies = []
f1_scores = []
precisions = []
recalls = []

for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    
    X_val = X_val[feature_list]
    dval = xgb.DMatrix(X_val)

    y_probs = booster.predict(dval)

    auc = roc_auc_score(y_val, y_probs)
    auc_scores.append(auc)

    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]

    y_pred = (y_probs >= best_threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1_sc = f1_score(y_val, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

    accuracies.append(acc)
    f1_scores.append(f1_sc)
    precisions.append(prec)
    recalls.append(rec)
#%%
print(f"AUC Mean: {np.mean(auc_scores):.4f}, Std: {np.std(auc_scores):.4f}")
print(f"Accuracy Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
print(f"F1 Score Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}")
print(f"Precision Mean: {np.mean(precisions):.4f}, Std: {np.std(precisions):.4f}")
print(f"Recall Mean: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f}")

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
accuracies = []
f1_scores = []
precisions = []
recalls = []

first_fold_done = False 

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    X_val = X_val[feature_list]
    dval = xgb.DMatrix(X_val)

    y_probs = booster.predict(dval)

    auc_score = roc_auc_score(y_val, y_probs)
    auc_scores.append(auc)

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_probs)
    f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    best_idx = np.argmax(f1_curve)
    best_threshold = thresholds[best_idx]

    y_pred = (y_probs >= best_threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1_sc = f1_score(y_val, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

    accuracies.append(acc)
    f1_scores.append(f1_sc)
    precisions.append(prec)
    recalls.append(rec)
if not first_fold_done:
    cm = confusion_matrix(y_val, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format=".2f")
    plt.title(f'Normalized Confusion Matrix')
    plt.show()

    prob_0 = y_probs[y_val == 0]
    prob_1 = y_probs[y_val == 1]

    plt.figure(figsize=(8, 5))
    sns.histplot(prob_0, bins=50, kde=True, alpha=0.5, label="True Class 0", color='blue')
    sns.histplot(prob_1, bins=50, kde=True, alpha=0.5, label="True Class 1", color='red')
    plt.axvline(x=best_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Best threshold = {best_threshold:.3f}')

    plt.xlabel("Predicted Probability of Signal",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.title(f'Predicted Probability Distributions',fontsize=16)
    plt.legend()
    plt.show()

    # 3. F1 Score vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_curve[:-1], marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f'F1 Score vs Threshold')
    plt.grid(alpha=0.3)
    plt.show()
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve (After Feature Engineering)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    print(f"Best threshold for Fold: {best_threshold:.4f} with F1 score {f1_curve[best_idx]:.4f}")

    first_fold_done = True

# %%
