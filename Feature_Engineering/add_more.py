import Inpaper
import pandas as pd
from Inpaper import compute_paper
from combine import compute_combine

def add_feature(df):
    original_columns = set(df.columns)  # record original columns
    compute_combine(df)
    new_columns = [col for col in df.columns if col not in original_columns]  #find new columns
    return df, new_columns  

if __name__ == "__main__":
    data_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_3data_more.csv'
    data = pd.read_csv(data_path)

    added_data, new_feature_names = add_feature(data)

    output_path_full = r'D:\Year3\BSc Project\Particle-Machine-Learning\balanced_more2.csv'
    added_data.to_csv(output_path_full, index=False)
    

    feature_list_path = r'D:\Year3\BSc Project\Particle-Machine-Learning\more_feature_names2.txt'
    with open(feature_list_path, 'w') as f:
        for feature in new_feature_names:
            f.write(feature + '\n')

    print(f"Feature-enhanced data saved to {output_path_full}")
    print(f"New feature names saved to {feature_list_path}")

