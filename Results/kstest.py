
import pandas as pd
from scipy.stats import ks_2samp


file_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\balanced_final.csv"

var1 = 'tau_fd_p'
var2 = 'tau_fd_m'
label_col = 'label'
def ks_test_by_label(file_path, var1, var2, label_col='label'):
    df = pd.read_csv(file_path, usecols=[var1, var2, label_col])

    var1_sig = df[df[label_col] == 1][var1].dropna()
    var1_bg  = df[df[label_col] == 0][var1].dropna()

    var2_sig = df[df[label_col] == 1][var2].dropna()
    var2_bg  = df[df[label_col] == 0][var2].dropna()

    stat1, pval1 = ks_2samp(var1_sig, var1_bg)
    print(f"{var1} KS test:")
    print(f"  KS statistics = {stat1}")
    print(f"  p value = {pval1}")
    print("Significant" if pval1 < 0.05 else "Not Significant")
    print()

    stat2, pval2 = ks_2samp(var2_sig, var2_bg)
    print(f"{var2} KS test:")
    print(f"  KS statistics = {stat2}")
    print(f"  p value = {pval2}")
    print("Significant" if pval1 < 0.05 else "Not Significant")
    print()

ks_test_by_label(file_path, var1, var2, label_col)
