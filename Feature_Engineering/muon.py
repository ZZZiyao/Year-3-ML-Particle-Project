import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def muon(df):

    df['delz'] = df['jpsi_ENDVERTEX_Z'] - df['B0_ENDVERTEX_Z']

    df['max_mu_PT'] = df[['mu_plus_PT', 'mu_minus_PT']].max(axis=1)
    df['min_mu_PT'] = df[['mu_plus_PT', 'mu_minus_PT']].min(axis=1)

    df['max_mu_PE'] = df[['mu_plus_PE', 'mu_minus_PE']].max(axis=1)
    df['min_mu_PE'] = df[['mu_plus_PE', 'mu_minus_PE']].min(axis=1)

    df['max_mu_P'] = df[['mu_plus_P', 'mu_minus_P']].max(axis=1)
    df['min_mu_P'] = df[['mu_plus_P', 'mu_minus_P']].min(axis=1)

    df['max_mu_PZ'] = df[['mu_plus_PZ', 'mu_minus_PZ']].max(axis=1)
    df['min_mu_PZ'] = df[['mu_plus_PZ', 'mu_minus_PZ']].min(axis=1)

    df['max_mu_ETA'] = df[['mu_plus_ETA', 'mu_minus_ETA']].max(axis=1)
    df['min_mu_ETA'] = df[['mu_plus_ETA', 'mu_minus_ETA']].min(axis=1)



    return df

    
#     df['B0_flight_length'] = np.sqrt(
#     (df['B0_ENDVERTEX_X'] - df['B0_OWNPV_X'])**2 +
#     (df['B0_ENDVERTEX_Y'] - df['B0_OWNPV_Y'])**2 +
#     (df['B0_ENDVERTEX_Z'] - df['B0_OWNPV_Z'])**2
# )



#     df['cos_plus_theta'] = (
#     (df['B0_PX'] * df['mu_plus_PX'] + df['B0_PY'] * df['mu_plus_PY'] + df['B0_PZ'] * df['mu_plus_PZ'])
#     / (df['B0_PT'] * df['mu_plus_PT'])
# )
#     df['cos_minus_theta'] = (
#     (df['B0_PX'] * df['mu_minus_PX'] + df['B0_PY'] * df['mu_minus_PY'] + df['B0_PZ'] * df['mu_minus_PZ'])
#     / (df['B0_PT'] * df['mu_minus_PT'])

#angle between two muon
# )

if __name__ == "__main__":

    
    save_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Muon_plots"
    os.makedirs(save_dir, exist_ok=True)

    data_files = [
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg2.csv",
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg3.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg4.csv",
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg5.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg6.csv",
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig2.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig3.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig4.csv",
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig5.csv", 
        r"D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig6.csv"
    ]
    datasets = []
    for f in data_files:
        try:
            df = pd.read_csv(f)
            df = muon(df)  
            datasets.append(df)
        except FileNotFoundError:
            print(f"⚠ cannot find: {f}")

    valid_features = ['delz','max_mu_PT','min_mu_PT','mean_mu_PT','max_mu_P','min_mu_P','mean_mu_P',
                      'max_mu_PE','min_mu_PE','mean_mu_PE','max_mu_ETA','min_mu_ETA','mean_mu_ETA',
                      'max_mu_PZ','min_mu_PZ','mean_mu_PZ']


    bg_colors = sns.color_palette("Blues", 6)  
    sig_colors = sns.color_palette("Reds", 6)  
    colors = bg_colors + sig_colors

    dataset_labels = [f"BG {i+1}" for i in range(6)] + [f"SIG {i+1}" for i in range(6)]

    rank = 1
    for feature in valid_features:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        data_melted = pd.concat(
            [df[[feature]].assign(dataset=dataset_labels[j]) for j, df in enumerate(datasets) if feature in df.columns]
        ).melt(id_vars=["dataset"], var_name="Feature", value_name="Value")

        sns.boxplot(x="dataset", y="Value", data=data_melted, width=0.6, palette=colors, ax=axes[0])
        axes[0].set_title(f"Boxplot of {feature}")
        axes[0].set_xticklabels(dataset_labels, rotation=45, ha="right")

        if feature in datasets[0].columns and feature in datasets[6].columns:
            sns.histplot(datasets[0][feature], bins=500, color="blue", alpha=0.5, stat="density", label="Background", ax=axes[1])
            sns.histplot(datasets[6][feature], bins=500, color="red", alpha=0.5, stat="density", label="Signal", ax=axes[1])

            axes[1].set_title(f"Histogram of {feature}")
            axes[1].set_xlabel(feature)
            axes[1].set_ylabel("Density")
            axes[1].legend(title="Class", loc='upper right')

        plt.tight_layout()

        
        save_path = os.path.join(save_dir, f"{rank}_{feature}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()  
        rank += 1

    print(f"All combined plots saved to: {save_dir}")






#b to K tau tau
#b to kstar tau tau
#D is heavier and travels further
