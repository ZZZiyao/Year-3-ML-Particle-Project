import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_bg1.csv')
# bg2 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg2.csv')
# bg3 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg3.csv')
# bg4= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg4.csv')
# bg5= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg5.csv')
# bg6= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg6.csv')

sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Data2\filtered_sig1.csv')
# sig2=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig2.csv')
# sig3=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig3.csv')
# sig4=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig4.csv')
# sig5=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig5.csv')
# sig6=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig6.csv')


#print(list(bg.columns))
#print(bg1.shape)
#print(sig1.shape)

k_columns = [col for col in bg1.columns if 'prob' in col.lower()]

print(k_columns)

