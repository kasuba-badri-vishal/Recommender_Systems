import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import warnings


class Collaborative:

  def __init__(self, utility_matrix):
    self.utility_matrix = utility_matrix
  
  def calculate_error(self,similarity_matrix):
    self.utility_matrix = self.utility_matrix.fillna(0)
    similarity_matrix = similarity_matrix.fillna(-100)
    utility_matrix = self.utility_matrix
    error_matrix =  pd.DataFrame(0,index = self.utility_matrix.index,columns = self.utility_matrix.columns,dtype=float)
    for i in tqdm(utility_matrix.index):
      for j in utility_matrix.columns:
        if utility_matrix.at[i,j]:
          error_matrix.at[i,j] = self.get_rating_error(i,j,similarity_matrix,20)

   
    print("saving generated error_matrix to pickle file...")
    error_matrix.to_pickle("error_matrix.pickle")
    print("saved to error_matrix.pickle")

  def get_rating_error(self,query_user,query_movie,sim_matrix,N=20):
    predicted_rating = 0
    utility_matrix = self.utility_matrix 
    query_useri = list(utility_matrix.index).index(query_user)
    query_moviei = list(utility_matrix.columns).index(query_movie)
    utility_matrix.at[query_user,query_movie]=0 
    arr_of_user = utility_matrix.loc[[query_user]].to_numpy()
    movies_rated_by_user = np.nonzero(arr_of_user)
    movies_rated_by_user = movies_rated_by_user[1].tolist()
    sim_movies = np.array(sim_matrix.iloc[query_moviei,movies_rated_by_user])
    if sim_movies.shape[0]<N:
      N = sim_movies.shape[0]
    indices = np.argpartition(sim_movies,-1*N)[-1*N:]
    sum,val=0,0
    for i in indices:
      temp = movies_rated_by_user[i]
      sum += sim_matrix.iloc[query_moviei,temp]
      val += sim_matrix.iloc[query_moviei,temp]*utility_matrix.iloc[query_useri,temp]
    if sum:
      predicted_rating = val/sum
    else:
      predicted_rating = 0
    return predicted_rating

  def get_normalized_cosine_similarity(self):
    if os.path.exists("sim_matrix.pickle"):
      cosine_similarity = pd.read_pickle("sim_matrix.pickle")
      print("Using already created sim_matrix.pickle file")
      return cosine_similarity
    else:
      # np.seterr(divide='ignore', invalid='ignore')
      start_time = time.time()
      utility_matrix = self.utility_matrix
      print("Normalizing the Utility_matrix with subtracting mean")
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in utility_matrix.columns:
            x = np.nanmean(utility_matrix[i],dtype=np.float)
            if x==np.nan:
              x=0
            utility_matrix.loc[:,i] = utility_matrix.loc[:,i] - x

      print("Calculated avg_item_rating")
      utility_matrix = utility_matrix.fillna(0)
      print("Added Bias element")
      self.utility_matrix = utility_matrix
      print("Calculating Cosine Similarity , Finding Similarity_matrix")
      cosine_similarity = pd.DataFrame(0,index=utility_matrix.columns,columns=utility_matrix.columns,dtype=np.float)
      

      for i in tqdm(utility_matrix.columns):
        for j in utility_matrix.columns:
          if (i==j):
            cosine_similarity.at[i,j] = 1
          elif (i<j):
            val = np.dot(utility_matrix[i],utility_matrix[j])
            val1 = np.linalg.norm(utility_matrix[i])
            val2 = np.linalg.norm(utility_matrix[j])
            if (val1==0) or (val2==0):
              cosine_similarity.at[i,j] = -100
              cosine_similarity.at[j,i] = -100
            else:
              cosine_similarity.at[i,j] = (val/(val1*val2))
              cosine_similarity.at[j,i] = (val/(val1*val2))

      end_time = time.time()
      print("Time Taken for generating sim_matrix : ",(end_time-start_time))
      print("saving generated cosine_similarity_matrix to pickle file...")
      cosine_similarity.to_pickle("sim_matrix.pickle")
      print("saved to sim_matrix.pickle")
      return cosine_similarity