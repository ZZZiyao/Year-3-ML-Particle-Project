#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
csv_file = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_final.csv'

df = pd.read_csv(csv_file)
#%%
phi = np.arctan2(df['pion_PY'], df['pion_PX']) 
pt = np.sqrt(df['pion_PX']**2 + df['pion_PY']**2)

plt.figure(figsize=(8, 5))
plt.scatter(pt, phi, alpha=0.3, s=5)
plt.xlabel(r'$p_T$ (GeV)', fontsize=14)
plt.ylabel(r'$\phi$ (rad)', fontsize=14)
plt.title(r'$\phi$ vs $p_T$', fontsize=16)
plt.grid(True)
plt.show()

# %%
low_pt_mask = pt < 2000
plt.hist(phi[low_pt_mask], bins=100, color='orange', edgecolor='black')
plt.xlabel(r'$\phi$ (rad)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.title(r'$\phi$ Distribution at Low $p_T$', fontsize=16)
plt.grid(True)
plt.show()

# %%
