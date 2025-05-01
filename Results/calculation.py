import numpy as np

labframe=np.array([0.027,0.035,0.033,0.083,0.066,0.113,0.065,0.114,0.015,0.068,0.078,0.136])
B0frame=np.array([0.134,0.108,0.185,0.155,0.171,0.158,0.168,0.155,0.321,0.250,0.236,0.219])

percentage_increase=(B0frame-labframe)/labframe*100
print(percentage_increase)
