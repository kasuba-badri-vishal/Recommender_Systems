import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import warnings

#  print(self.utility_matrix.iloc[query_user,query_movie])
#     if self.utility_matrix.iloc[query_user,query_movie] is not np.nan:
#       print(self.utility_matrix.iloc[query_user,])
#       self.utility_matrix[query_user,].fillna(0,inplace=True)
#       print(self.utility_matrix.iloc[query_user,])
#       # print(self.utility_matrix)
#       movies_rated_by_user = np.argpartition(self.utility_matrix.iloc[query_user,],1*40)[40:]
#       movies_rated_by_user = movies_rated_by_user.index.values.tolist()
#       for i in movies_rated_by_user:
#         print(self.utility_matrix.at[query_user,i])
      # movies_rated_by_user = movies_rated_by_user.tolist()
      # print(movies_rated_by_user)
      # sim_movies = []
      

      # indices = np.argpartition(sim_matrix[query_movie],-1*N)[-1*N:] 
      # sum = np.sum(sim_matrix[query_movie][indices])
      # print(sum)
      # val = 0
      # indices = indices.tolist()
      # for i in indices:
      #   print(sim_matrix[query_movie][i])
      # print(indices)
      # for i in indices:
      #   val += sim_matrix.at[query_movie,i]*(self.utility_matrix.at[query_user,i])
      # predicted_rating = val/sum; 

class Collaborative:

  def __init__(self, utility_matrix):
    self.utility_matrix = utility_matrix

  def get_rating(self,query_user,query_movie,sim_matrix,N=20):
    predicted_rating = 0
    query_useri = query_user-1
    utility_matrix = self.utility_matrix.fillna(0)
    sim_matrix = sim_matrix.fillna(-100)
    query_moviei = list(utility_matrix.columns).index(query_movie)
    arr_of_user = utility_matrix.loc[[query_user]].to_numpy()
    if utility_matrix.loc[query_user,query_movie]: 
      print("Query user already rated query movie in dataset")
      predicted_rating = self.utility_matrix.loc[query_user,query_movie]
    else :
      movies_rated_by_user = np.nonzero(arr_of_user)
      print("Total number of movies rated by Query_User are : ",np.count_nonzero(utility_matrix.loc[[query_user]]))
      movies_rated_by_user = movies_rated_by_user[1].tolist()
      # print(movies_rated_by_user)
      ratings_of_user = utility_matrix.iloc[query_useri,movies_rated_by_user].tolist()
      mean_rating = np.mean(ratings_of_user)
      # print(ratings_of_user)
      print("User average rating : ",mean_rating)
      indices = np.argpartition(sim_matrix.loc[query_movie],-1*500)[-1*500:] 
      # indices = indices.tolist()
      indices = list(indices.values.flatten())
      # print(indices)
      ans = np.intersect1d(indices,movies_rated_by_user).flatten()
      print("Number of common movies rated by query user : ",ans.shape[0])
      # print(ans)
      # print(type(ans))
      sum,val=0,0
      for i in ans:
        temp = sim_matrix.iloc[query_moviei,i]
        val += temp*utility_matrix.iloc[query_useri,i]
        sum += temp
      # print(sum)
      # print(val)
      predicted_rating = val/sum
    return predicted_rating


  def get_normalized_cosine_similarity(self):
    if os.path.exists("sim_matrix.pickle"):
      cosine_similarity = pd.read_pickle("sim_matrix.pickle")
      print("Using already created sim_matrix.pickle file")
      # print(np.sum(cosine_similarity.values==np.nan))
      # print(cosine_similarity)
      return cosine_similarity
    else:
      np.seterr(divide='ignore', invalid='ignore')
      start_time = time.time()
      utility_matrix = self.utility_matrix
      print("Normalizing the Utility_matrix with subtracting mean")
      for i in utility_matrix.columns:
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          x = np.nanmean(utility_matrix[i],dtype=np.float)
          if x==np.nan:
            x=0
          utility_matrix.loc[:,i] = utility_matrix.loc[:,i] - x

      print("Calculated avg_item_rating")
     
      print(utility_matrix)
      utility_matrix = utility_matrix.fillna(0)
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