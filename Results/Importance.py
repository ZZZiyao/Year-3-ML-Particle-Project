#%%
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

model_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\more_model1.pkl"
loaded_model = joblib.load(model_path)
from xgboost import XGBClassifier
new_model = XGBClassifier()
new_model._Booster = loaded_model  

feature_importance = loaded_model.get_score(importance_type="gain")

feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(20)

print(feature_importance_df)

#%%

plt.figure(figsize=(10, 10))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="blue")
plt.xlabel("Importance (Gain)")
plt.ylabel("Features")
plt.title("Feature Importance for New model")
plt.gca().invert_yaxis() 
plt.show()

# %%
