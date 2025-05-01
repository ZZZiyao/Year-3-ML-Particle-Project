#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from add_feature import add_feature

save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Carefull_New_Plots"
os.makedirs(save_dir, exist_ok=True)

data_files = [
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg1.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg2.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg3.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg4.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg5.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg6.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig1.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig2.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig3.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig4.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig5.csv',
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig6.csv',

]

datasets = [pd.read_csv(f) for f in data_files]
#%%
selected_feature = 'kstar_P'


def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)], lower_bound, upper_bound

bg_filtered, bg_lower, bg_upper = remove_outliers(datasets[0][selected_feature].dropna())  # BG1
sig_filtered, sig_lower, sig_upper = remove_outliers(datasets[6][selected_feature].dropna())  # SIG1

hist_lower = min(bg_lower, sig_lower)
hist_upper = max(bg_upper, sig_upper)

dataset_labels = [f"BG {i+1}" for i in range(6)] + [f"SIG {i+1}" for i in range(6)]


bg_colors = sns.color_palette("Blues", 6)
sig_colors = sns.color_palette("Reds", 6)
colors = bg_colors + sig_colors

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

data_melted = pd.concat(
    [df[[selected_feature]].assign(dataset=dataset_labels[j]) for j, df in enumerate(datasets)]
).melt(id_vars=["dataset"], var_name="Feature", value_name="Value")

sns.boxplot(x="dataset", y="Value", data=data_melted, width=0.6, palette=colors, ax=axes[0])
axes[0].set_title(f"Boxplot of {selected_feature}")
axes[0].set_xticklabels(dataset_labels, rotation=45, ha="right")

plt.hist(datasets[0][selected_feature], bins=500, color="blue", alpha=0.5, density=True, label="Background",
         range=(0,hist_upper)
         )
plt.hist(datasets[6][selected_feature], bins=500, color="red", alpha=0.5, density=True, label="Signal", 
         range=(0,hist_upper)
         )
#plt.axvline(5279.65, color='red', linestyle='--', linewidth=1.5, label='Theoretical $B^0$ Mass')

axes[1].set_title(f"Distribution of B0_ENDVERTEX_CHI2")

axes[1].set_xlabel('chi2')
axes[1].set_ylabel("Density")
axes[1].legend(title="Class", loc='upper right')

plt.tight_layout()
plt.show()

save_path = os.path.join(save_dir, f"{selected_feature}_filtered.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Plot for {selected_feature} saved to: {save_dir}")


# %%
datasets[0][selected_feature].describe()
#%%
datasets[6][selected_feature].describe()

# %%
import numpy as np

B0_mass_theory = 5279.65

n_over_theory_0 = np.sum(datasets[0][selected_feature] > B0_mass_theory)
n_total_0 = len(datasets[0])
frac_over_theory_0 = n_over_theory_0 / n_total_0

print(f"Dataset 0: {n_over_theory_0} / {n_total_0} ({frac_over_theory_0:.2%}) above theory mass")

n_over_theory_6 = np.sum(datasets[6][selected_feature] > B0_mass_theory)
n_total_6 = len(datasets[6])
frac_over_theory_6 = n_over_theory_6 / n_total_6

print(f"Dataset 6: {n_over_theory_6} / {n_total_6} ({frac_over_theory_6:.2%}) above theory mass")

# %%
