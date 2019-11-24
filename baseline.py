import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

class Baseline:

  def __init__(self, utility_matrix):
    self.utility_matrix = utility_matrix


  def get_rating_deviation(self):
    avg_movie_rating = self.utility_matrix.mean(axis=0)
    avg_user_rating = self.utility_matrix.mean(axis=1)
    print("calculated average user and movie ratings")
    mu = np.mean(avg_movie_rating)
    print("Calculated mu")
    print(mu)
    baseline_matrix = pd.DataFrame(0,index = self.utility_matrix.index,columns = self.utility_matrix.columns)
    # baseline_matrix = np.add(baseline_matrix,avg_movie_rating)
    baseline_matrix = baseline_matrix.add(avg_movie_rating,axis=1) + baseline_matrix.add(avg_user_rating,axis=0) - mu
    print(baseline_matrix)
    print("saving generated cosine_baseline_matrix to pickle file...")
    baseline_matrix.to_pickle("baseline_matrix.pickle")
    print("saved to baseline_matrix.pickle")
    return baseline_matrix

  def calculate_error(self,similarity_matrix,baseline_matrix):
    utility_matrix = self.utility_matrix.fillna(0)
    similarity_matrix = similarity_matrix.fillna(-100)
    # utility_matrix = self.utility_matrix
    error_matrix =  pd.DataFrame(0,index = self.utility_matrix.index,columns = self.utility_matrix.columns,dtype=float)
    size=0
    for i in tqdm(range(500)):
      for j in range(500):
        if utility_matrix.iloc[i,j]:
          size+=1
          error_matrix.iloc[i,j] = self.get_baseline_rating_prediction_error(i,j,similarity_matrix,utility_matrix,baseline_matrix,20)
        else:
          error_matrix.iloc[i,j]=0
   
    print("saving generated baseline_error_matrix to pickle file...")
    error_matrix.to_pickle("baseline_error_matrix.pickle")
    print("saved to baseline_error_matrix.pickle")
    return size

  def get_baseline_rating_prediction_error(self,query_user,query_movie,utility_matrix,sim_matrix,baseline_matrix,N):

    predicted_rating = baseline_matrix.iloc[query_user,query_movie]
    arr_of_user = utility_matrix.iloc[[query_user]].to_numpy()
    movies_rated_by_user = np.nonzero(arr_of_user)
    movies_rated_by_user = movies_rated_by_user[1].tolist()
    # query_useri = list(utility_matrix.index).index(query_user)
    # query_moviei = list(utility_matrix.columns).index(query_movie)
    sim_movies = np.array(sim_matrix.iloc[query_movie,movies_rated_by_user])     
    indices = np.argpartition(sim_movies,-1*N)[-1*N:]
    sum,val=0,0
    for i in indices:
      temp = movies_rated_by_user[i]
      sum += sim_matrix.iloc[query_movie,temp]
      val += sim_matrix.iloc[query_movie,temp]*(utility_matrix.iloc[query_user,temp]-baseline_matrix.iloc[query_user,temp])
    if sum:
      predicted_rating += (val)/(sum)
    # print(predicted_rating)
    return predicted_rating

