import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


csv_file = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_final.csv'
features_file = r"D:\Year3\BSc Project\Particle-Machine-Learning\more_feature_names.txt"

df = pd.read_csv(csv_file)

with open(features_file, "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

df['label'] = df['label'].astype(int)  


datasets = [df[df['label'] == i] for i in range(2)]  # 0: Background, 1: Signal

def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)], lower_bound, upper_bound

save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\More_hist_bound"
os.makedirs(save_dir, exist_ok=True)

for selected_feature in feature_names:
    if selected_feature not in df.columns:
        continue
    
    
    _, bg_lower, bg_upper = remove_outliers(datasets[0][selected_feature].dropna())  # Background
    _, sig_lower, sig_upper = remove_outliers(datasets[1][selected_feature].dropna())  # Signal
    
    hist_lower = min(bg_lower, sig_lower)
    hist_upper = max(bg_upper, sig_upper)
    
    plt.figure(figsize=(7, 5))
    plt.hist(datasets[0][selected_feature], bins=500, color="blue", alpha=0.5, density=True, label="Background", 
             range=(hist_lower, hist_upper)
             )
    plt.hist(datasets[1][selected_feature], bins=500, color="red", alpha=0.5, density=True, label="Signal", 
             range=(hist_lower, hist_upper)
             )
    
    plt.title(f"Histogram of {selected_feature}")
    plt.xlabel(selected_feature)
    plt.ylabel("Density")
    plt.legend(title="Class", loc='upper right')
    
    save_path = os.path.join(save_dir, f"{selected_feature}_filtered.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Plot for {selected_feature} saved to: {save_path}")
