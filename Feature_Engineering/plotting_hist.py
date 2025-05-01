#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from add_feature import add_feature
csv_file = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_final.csv'

df = pd.read_csv(csv_file)


df['label'] = df['label'].astype(int)  

datasets = [df[df['label'] == i] for i in range(2)]  # 0: Background, 1: Signal

#%%
selected_feature = "M_missing_square"  

def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = 0
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)], lower_bound, upper_bound

bg_filtered, bg_lower, bg_upper = remove_outliers(datasets[0][selected_feature].dropna())  # BG1
sig_filtered, sig_lower, sig_upper = remove_outliers(datasets[1][selected_feature].dropna())  # SIG1

hist_lower = min(bg_lower, sig_lower)
hist_upper = max(bg_upper, sig_upper)

plt.figure(figsize=(10,6))
plt.hist(datasets[0][selected_feature], bins=500, alpha=0.5, density=True, color='blue',label="Background",
         range=(-0.2e8,hist_upper)
         )
plt.hist(datasets[1][selected_feature], bins=500, alpha=0.5, density=True, color='red',label="Signal", 
         range=(-0.2e8,hist_upper)
         )
#plt.axvline(x=6955256.451199999, label='Theoretical Starting Point of Background',linestyle='--')
#plt.axvline(x=6314747.220000001, label='Theoretical Starting Point of Signal',linestyle='--',color='red')
#plt.axvline(x=703, label='Theoretical Starting Point of Background',linestyle='--')
#plt.axvline(x=5279.63, label='Theoretical B0 Mass',linestyle='--',color='red')


#plt.title(f"B0 End Vertex CHi2 Distribution")
#plt.title(f"Kaon Total Momentum in Lab Frame")
plt.title(f"B0 Missing Mass Squared",fontsize=16)
#plt.xlabel('chi^2',fontsize=16)
plt.xlabel('Mass^2(MeV^2/c^4)',fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.legend(fontsize=16)

plt.tight_layout()
plt.show()


#%%

plt.figure(figsize=(9,6))

plt.hist(datasets[0]['kaon_P']+datasets[0]['pion_P'], bins=500, color="blue", alpha=0.5, density=True, label="Background",range=(0,200000))
plt.hist(datasets[1]['kaon_P']+datasets[1]['pion_P'], bins=500, color="red", alpha=0.5, density=True, label="Signal", range=(0,200000))
plt.title(f"Kstar Total Momentum in Lab Frame")
plt.xlabel('Momentum(MeV/c)')
plt.ylabel("Density")
plt.legend(title="Class", loc='upper right')

plt.tight_layout()
plt.show()


# %%
sum_series = datasets[1]['Kaon_P_B0'] + datasets[1]['Pion_P_B0']
nan_count = sum_series.isna().sum()
print(f"有 {nan_count} 个 NaN 值")
#%%
nan_count = datasets[1]['M_missing'].isna().sum()
print(f"有 {nan_count} 个 NaN 值")
total = datasets[1]['M_missing'].count()  
ratio = nan_count / total

print(f"占比为: {ratio:.4%}") 


# %%
sum_series.describe()
# %%
gamma_series = (datasets[1]['B0_PE_C'] + datasets[1]['B0_P_C'])/3e8
gamma_series.describe()
# %%
K0mass = np.sqrt(2) * 497.611
print(K0mass)
condition = datasets[0]['M_missing'] < K0mass

count = condition.sum()
total = datasets[0]['M_missing'].count()  
ratio = count / total

print(f"M_missing 中小于 {K0mass} 的数量为: {count}")
print(f"占比为: {ratio:.4%}")  

# %%
Dmass=1864.84
taumass=1776.9
#condition = datasets[0]['Q_square'] < 2*Dmass**2
condition = datasets[1]['Q_square'] < 2*taumass**2
print(2*Dmass**2)
print(2*taumass**2)
#%%
count = condition.sum()
total = datasets[0]['Q_square'].count()
print(total)
ratio = count / total

print(f"数量为: {count}")
print(f"占比为: {ratio:.4%}")  

# %%
datasets[1]['error_m'].describe()
# %%
condition=datasets[1]['error_m']>0.5
count = condition.sum()
total = datasets[0]['error_m'].count()
print(total)
ratio = count / total
print(ratio)
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

selected_features = ["kdR", "mdR"]  
feature_titles = ["Angular Distances Between Kaon and Pion", "Angular distances Between two Muons"]

def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = 0
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)], lower_bound, upper_bound

fig, axes = plt.subplots(2, 1, figsize=(10, 12))  
plt.subplots_adjust(hspace=0.4)  

for idx, selected_feature in enumerate(selected_features):
    bg_filtered, bg_lower, bg_upper = remove_outliers(datasets[0][selected_feature].dropna())  # BG
    sig_filtered, sig_lower, sig_upper = remove_outliers(datasets[1][selected_feature].dropna())  # SIG

    hist_lower = min(bg_lower, sig_lower)
    hist_upper = max(bg_upper, sig_upper)

    ax = axes[idx]
    ax.hist(datasets[0][selected_feature], bins=500, alpha=0.5, density=True, color='blue', label="Background",
            range=(0, hist_upper))
    ax.hist(datasets[1][selected_feature], bins=500, alpha=0.5, density=True, color='red', label="Signal",
            range=(0, hist_upper))

    ax.set_title(feature_titles[idx],fontsize=16)
    ax.set_xlabel('Angle (rad)',fontsize=16)
    ax.set_ylabel('Density',fontsize=16)
    ax.legend(fontsize=16)

plt.tight_layout()
plt.show()

# %%
