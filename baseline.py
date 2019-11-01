import numpy as np

class Baseline:

  def __init__(self, utility_matrix):
    self.utility_matrix = utility_matrix

  def get_avg_movie_rating(self):
    avg_movie_rating = {}
    for movieID in utility_matrix.columns:
      avg_movie_rating[movieID] = np.nanmean(self.utility_matrix[movieID])
    return avg_movie_rating

  def get_avg_user_rating(self):
    avg_user_rating = {} 
    for userID in utility_matrix.index:
      avg_user_rating[userID] = np.nanmean(self.utility_matrix[userID])
    return avg_user_rating

  def get_rating_deviation(avg_user_rating,avg_movie_rating):
    mu = np.array([avg_movie_rating[movieID] for movieID in avg_movie_rating]).mean()
    baseline_matrix = pd.DataFrame(index = self.utility_matrix.index,columns = self.utility_matrix.columns)
    for i in baseline_matrix.index:
      for j in baseline_matrix.columns:
        baseline_matrix.at(i,j) = avg_movie_rating[j]+avg_user_rating[i]-mu

    return baseline_matrix

def baseline_rating_prediction(self,utility_matrix,query_user,query_movie,sim_matrix,baseline_matrix,N):
  predicted_rating = baseline_matrix.at[query_user,query_movie]

  indices = np.argpartition(sim_matrix[query_movie],-1*N)[-1*N:]
  for i in indices:
    sum += sim_matrix.at[query_movie,i]
    val += sim_matrix.at[query_movie,i]*(utility_matrix.at[query_user,i]-baseline_matrix.at[query_user,i])
  predicted_rating += (val)/(sum)
  return predicted_rating
