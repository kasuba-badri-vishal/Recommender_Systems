import numpy as np
import pandas as pd
import warnings

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

  def get_baseline_rating_prediction(self,query_user,query_movie,sim_matrix,baseline_matrix,N):
    utility_matrix = self.utility_matrix.fillna(0)
    if utility_matrix.loc[query_user,query_movie]:   
      print("Query user already rated query movie in dataset")
      predicted_rating = utility_matrix.loc[query_user,query_movie]
    else:
      predicted_rating = baseline_matrix.at[query_user,query_movie]
      arr_of_user = utility_matrix.loc[[query_user]].to_numpy()
      movies_rated_by_user = np.nonzero(arr_of_user)
      print("Total number of movies rated by Query_User are : ",np.count_nonzero(utility_matrix.loc[[query_user]]))
      movies_rated_by_user = movies_rated_by_user[1].tolist()
      query_useri = query_user-1
      query_moviei = list(utility_matrix.columns).index(query_movie)
      sim_movies = np.array(sim_matrix.iloc[query_moviei,movies_rated_by_user])
      
      indices = np.argpartition(sim_movies,-1*N)[-1*N:]
      print(indices)
      sum,val=0,0
      for i in indices:
        temp = movies_rated_by_user[i]
        # print(temp)
        sum += sim_matrix.iloc[query_moviei,temp]
        val += sim_matrix.iloc[query_moviei,temp]*(utility_matrix.iloc[query_useri,temp]-baseline_matrix.iloc[query_useri,temp])
      predicted_rating += (val)/(sum)
    return predicted_rating
