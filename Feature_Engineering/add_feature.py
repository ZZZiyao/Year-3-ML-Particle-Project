from missing_mass_2 import compute_missing_mass
from muon import muon
from tao_fly_dis import compute_tau_fly_distance
from D_and_tau import compute_tau
from Kastar import kstar
from kstar_angle import compute_kangles
from muon_angle import compute_angles
from PT_related import compute_PT
import pandas as pd

def add_feature(df):
    original_columns = set(df.columns)
    compute_missing_mass(df)
    muon(df)
    compute_tau_fly_distance(df)
    compute_tau(df)
    kstar(df)
    compute_kangles(df)
    compute_angles(df)
    compute_PT(df)
    
    new_columns = [col for col in df.columns if col not in original_columns] 
    return df, new_columns 

if __name__ == "__main__":
    data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_test.csv'
    data = pd.read_csv(data_path)

    added_data, new_feature_names = add_feature(data)

    output_path_full = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_test_added.csv'
    added_data.to_csv(output_path_full, index=False)
    
    feature_list_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\new_feature_names.txt'
    with open(feature_list_path, 'w') as f:
        for feature in new_feature_names:
            f.write(feature + '\n')

    print(f"Feature-enhanced data saved to {output_path_full}")
    print(f"New feature names saved to {feature_list_path}")
