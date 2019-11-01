import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time

class Collaborative:

  def __init__(self, utility_matrix):
    self.utility_matrix = utility_matrix

  def get_rating(self,query_user,query_movie,sim_matrix,N=20):
    predicted_rating = 0
    indices = np.argpartition(sim_matrix[query_movie],-1*N)[-1*N:]
    sum = np.sum(sim_matrix[query_movie][indices])
    val = 0
    indices = indices.tolist()

    print(indices)
    for i in indices:
      val += sim_matrix.at[query_movie,i]*(self.utility_matrix.at[query_user,i])
    predicted_rating = val/sum; 
    return predicted_rating

  

  def get_normalized_cosine_similarity(self):
    if os.path.exists("sim_matrix.pickle"):
      cosine_similarity = pd.read_pickle("sim_matrix.pickle")
      print("Using already created sim_matrix.pickle file")
      # print(cosine_similarity.shape)
      # print(cosine_similarity)
      return cosine_similarity
    else:
      np.seterr(divide='ignore', invalid='ignore')
      start_time = time.time()
      utility_matrix = self.utility_matrix
      # print(utility_matrix)
      print("Normalizing the Utility_matrix with subtracting mean")


      avg_item_rating = [0]*(utility_matrix.shape[1])
      x=0
      # print(utility_matrix.columns)
      
      for i in utility_matrix.columns:
        avg_item_rating[x] = np.nanmean(utility_matrix[i],dtype=np.float)
        x += 1
      print("Calculated avg_item_rating")
      avg_item_rating = np.nan_to_num(avg_item_rating)
     
      
      utility_matrix = utility_matrix.subtract(avg_item_rating,axis=1)
      utility_matrix = utility_matrix.fillna(0)
      # print(utility_matrix)
      # for col in tqdm(range(utility_matrix.shape[1])):
      #   utility_matrix.iloc[:,col]-=avg_item_rating[col]
      print("Added Bias element")
      
      self.utility_matrix = utility_matrix
      print("Calculating Cosine Similarity , Finding S matrix")
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
              cosine_similarity.at[i,j] = -10
              cosine_similarity.at[j,i] = -10
            else:
              cosine_similarity.at[i,j] = (val/(val1*val2))
              cosine_similarity.at[j,i] = (val/(val1*val2))

      end_time = time.time()
      print("Time Taken for generating sim_matrix : ",(end_time-start_time))
      print("saving generated cosine_similarity_matrix to pickle file...")
      cosine_similarity.to_pickle("sim_matrix.pickle")
      print("saved to sim_matrix.pickle")
      return cosine_similarity