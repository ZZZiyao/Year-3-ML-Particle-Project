#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
csv_file = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_final.csv'

df = pd.read_csv(csv_file)


# df['label'] = df['label'].astype(int)  

# datasets = [df[df['label'] == i] for i in range(2)]  # 0: Background, 1: Signal
#%%
# Kaon mass squared before and after boost
df['kaon_m_orig_squared'] = df['kaon_PE']**2 - df['kaon_P']**2
df['kaon_m_boosted_squared'] = df['Kaon_PE_B0']**2 - df['Kaon_P_B0']**2
df['kaon_mass_diff'] = np.abs(df['kaon_m_boosted_squared'] - df['kaon_m_orig_squared'])
plt.hist(df['kaon_mass_diff'], bins=100)
df['kaon_mass_diff'].describe()

# Pion mass squared before and after boost
df['pion_m_orig_squared'] = df['pion_PE']**2 - df['pion_P']**2
df['pion_m_boosted_squared'] = df['Pion_PE_B0']**2 - df['Pion_P_B0']**2
df['pion_mass_diff'] = np.abs(df['pion_m_boosted_squared'] - df['pion_m_orig_squared'])
plt.hist(df['pion_mass_diff'], bins=100)
df['pion_mass_diff'].describe()

# Mu+ mass squared before and after boost
df['mu_plus_m_orig_squared'] = df['mu_plus_PE']**2 - df['mu_plus_P']**2
df['mu_plus_m_boosted_squared'] = df['MP_PE_B0']**2 - df['MP_P_B0']**2
df['mu_plus_mass_diff'] = np.abs(df['mu_plus_m_boosted_squared'] - df['mu_plus_m_orig_squared'])
plt.hist(df['mu_plus_mass_diff'], bins=100)
df['mu_plus_mass_diff'].describe()

# Mu- mass squared before and after boost
df['mu_minus_m_orig_squared'] = df['mu_minus_PE']**2 - df['mu_minus_P']**2
df['mu_minus_m_boosted_squared'] = df['MM_PE_B0']**2 - df['MM_P_B0']**2
df['mu_minus_mass_diff'] = np.abs(df['mu_minus_m_boosted_squared'] - df['mu_minus_m_orig_squared'])
plt.hist(df['mu_minus_mass_diff'], bins=100)
df['mu_minus_mass_diff'].describe()

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(4, 1, figsize=(10, 30))
axs = axs.flatten()

mass_diff_vars = [
    ('kaon_mass_diff', 'Kaon'),
    ('pion_mass_diff', 'Pion'),
    ('mu_plus_mass_diff', r'$\mu^+$'),
    ('mu_minus_mass_diff', r'$\mu^-$')
]

for i, (col, name) in enumerate(mass_diff_vars):
    data = df[col].dropna()
    mean_val = data.mean()
    std_val = data.std()
    max_val = data.max()

    axs[i].hist(data, bins=100, color='lightsteelblue', edgecolor='black')
    axs[i].set_title(f'{name} Mass Squared Difference', fontsize=16)
    axs[i].set_xlabel(r'$|m^2_\mathrm{boost} - m^2_\mathrm{lab}|$ (MeV$^2$)', fontsize=16)
    axs[i].set_ylabel('Counts', fontsize=16)


    textstr = '\n'.join((
        f'Mean = {mean_val:.2e}',
        f'Std = {std_val:.2e}',
        f'Max = {max_val:.2e}'
    ))
    axs[i].text(0.98, 0.95, textstr,
                transform=axs[i].transAxes,
                fontsize=16,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Invariant Mass Squared Differences Before and After Boost', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
