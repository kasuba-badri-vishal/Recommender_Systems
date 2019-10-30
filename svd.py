# from numpy import array
from scipy.linalg import svd
import time
import numpy as np

class SVD:
  def __init__(self,utility_matrix):
    self.utility_matrix = utility_matrix
    self.utility_matrix = utility_matrix.fillna(0)
    # self.utility_matrix[~self.utility_matrix.isin([np.nan, np.inf, -np.inf]).any(1)]
    print("From svd.py")
    # print(utility_matrix)
    # self.utility_matrix = array([[1, 2], [3, 4], [5, 6]])
    start_time = time.time()
    U,s,VT = svd(utility_matrix)
    print(U)
    print(s)
    print(VT)
    end_time = time.time()
    print("Time taken for svd : ", (end_time-start_time))