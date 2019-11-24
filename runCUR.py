import cur
import pandas as pd
import numpy as np
from time import time

# M = pd.DataFrame([
#         [1,1,1,0,0],
#         [3,3,3,0,0],
#         [4,4,4,0,0],
#         [5,5,5,0,0],
#         [0,0,0,4,4],
#         [0,0,0,5,5],
#         [0,0,0,2,2]
#     ])

st_time = time()
M = pd.read_pickle("utility_matrix.pickle").fillna(0)
print(f"Time taken to load : {time()-st_time}")

st_time = time()
C, U, R = cur.get_cur(M, r=1000, energy=0.8)
print(f"Time taken for CUR decom : {time()-st_time}")
MAE = abs(M.values - C@U@R).sum()/(M.shape[0]*M.shape[1])
MSE = np.square(M.values - C@U@R).sum()/(M.shape[0]*M.shape[1])
RMSE = MSE**0.5
print("MAE:", MAE)
print("RMSE: ", RMSE)