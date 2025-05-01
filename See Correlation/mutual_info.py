from sklearn.feature_selection import mutual_info_classif
import pandas as pd


data_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\merged_data_small.csv"
data = pd.read_csv(data_path)


X = data.drop(columns=[col for col in data.columns if 'mother' in col.lower()]
              + [col for col in data.columns if 'true' in col.lower()]
             + [col for col in data.columns if 'own' in col.lower()]
            + [col for col in data.columns if 'decision' in col.lower()]
            + [col for col in data.columns if 'endvertex_x' in col.lower()]
            + [col for col in data.columns if 'endvertex_y' in col.lower()]
            + [col for col in data.columns if 'endvertex_z' in col.lower()]
              +['label', 'eventNumber', 'runNumber','kstar_M','Polarity'])

y=data['label']

mi_scores = mutual_info_classif(X, y)

mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

top_20_mi_features = mi_series.head(20).index.tolist()

print(top_20_mi_features)
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(12, 6))
mi_series.head(20).plot(kind='bar', color='royalblue')

plt.title("Top 20 Features Based on Mutual Information", fontsize=14)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Mutual Information Score", fontsize=12)
plt.xticks(rotation=45, ha='right')  

plt.show()
