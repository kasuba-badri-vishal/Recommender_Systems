# from svd import SVD
from collaborative import Collaborative
from baseline import Baseline

import pandas as pd
import os
from time import time
from math import sqrt
import numpy as np

from scipy.sparse import csc_matrix
import pickle


def load_sparse_pickle():
  users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
  movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])
  utility_matrix_sparse = pd.read_pickle("utility_matrix_sparse.pickle")
  utility_matrix = pd.DataFrame(utility_matrix_sparse.todense(),index=users["UserID"],columns=movies["MovieID"], dtype=np.float)
  utility_matrix.replace(0, np.nan, inplace=True)
  return utility_matrix

def item_item_collaborative_model(colab,utility_matrix):
  #Initiation
  similarity_matrix = colab.get_normalized_cosine_similarity()
  print("Calculated Similarity Matrix")

  # #Query Input
  print("Take Input of user ID and Movie ID to predict the Rating")
  query_user = int(input("Query User ID : "))
  query_movie = int(input("Query Movie ID : "))
  n = 20

  #Print predicted answer
  predicted_rating = colab.get_rating(query_user,query_movie,similarity_matrix,n)
  print("Predicted Rating for the given query is ",predicted_rating)

  #Return sim_matrix for further usage
  return similarity_matrix


def baseline_model(utility_matrix,similarity_matrix):
  if os.path.exists("baseline_matrix.pickle"):
    baseline = Baseline(utility_matrix)
    baseline_matrix = pd.read_pickle("baseline_matrix.pickle")
    print("Using already created baseline_matrix.pickle file")
  else:
    print("Baseline Method Started")
    baseline = Baseline(utility_matrix)
    baseline_matrix = baseline.get_rating_deviation()
    
  #Query Input
  print("Take Input of user ID and Movie ID to predict the Rating")
  query_user = int(input("Query User ID : "))
  query_movie = int(input("Query Movie ID : "))
  n = 20
  predicted_rating = baseline.get_baseline_rating_prediction(query_user,query_movie,similarity_matrix,baseline_matrix,n)
  print("Predicted Rating using baseline approach : ",predicted_rating)

  if os.path.exists("baseline_error_matrix.pickle"):
    error_matrix = pd.read_pickle("baseline_error_matrix.pickle")
    print("Using already created baseline_error_matrix.pickle file")
    utility_matrix = utility_matrix.fillna(0)
    # matrix = pd.DataFrame(0,index=np.arrange(100),columns=np.arrange(100),dtype=np.float)
    # print(error_matrix)
    for i in range(100):
      for j in range(100):
        error_matrix.iloc[i,j] = utility_matrix.iloc[i,j]-error_matrix.iloc[i,j]
    

    # print(error_matrix)
    ans2 = error_matrix.sum().sum()
    matrix = error_matrix.pow(2)
    ans = matrix.sum().sum()
    ans = ans/10000
    ans2 = ans/10000
    print("Calculated RMSE is : ",sqrt(ans))
    print("Calculated MAE is ",ans2)
  else:
    baseline.calculate_error(similarity_matrix,baseline_matrix)



def calculate_error(colab,utility_matrix,similarity_matrix):
  if os.path.exists("error_matrix.pickle"):
    error_matrix = pd.read_pickle("error_matrix.pickle")
    print("Using already created error_matrix.pickle file")
    utility_matrix = utility_matrix.fillna(0)
    print(error_matrix)
    matrix = utility_matrix-error_matrix
    print(matrix)
    ans2 = matrix.sum().sum()
    matrix = matrix.pow(2)
    ans = matrix.sum().sum()
    ans = ans/(utility_matrix.shape[0]*utility_matrix.shape[1])
    ans2 = ans/(utility_matrix.shape[0]*utility_matrix.shape[1])
    print("Calculated RMSE is : ",sqrt(ans))
    print("Calculated MAE is ",ans2)
  else:
    colab.calculate_error(similarity_matrix)

def main():
  utility_matrix = None
  users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
  movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])
  if os.path.exists("utility_matrix_sparse.pickle"):
    # utility_matrix = pd.read_pickle("utility_matrix.pickle")
    # print("Using already created utility_matrix.pickle file")
    utility_matrix = load_sparse_pickle()
    start_time = time()
    print("Using already created utility_matrix_sparse.pickle file. Time taken: ", time()-starttime)
    start_time = time()
    colab = Collaborative(utility_matrix)
    similarity_matrix = item_item_collaborative_model(colab,utility_matrix)
    end_time = time()
    baseline_model(utility_matrix,similarity_matrix)
    end_time2 = time()
    print(end_time-start_time)
    print(end_time2-end_time)
    # calculate_error(colab,utility_matrix,similarity_matrix)
  
  else:
    start_time = time()
    
    ratings = pd.read_csv('dataset/ratings.dat',delimiter='::',engine='python',names = ["UserID","MovieID", "Rating","Timestamp"],usecols = ["UserID","MovieID","Rating"])
    

    utility_matrix = pd.DataFrame(index=users["UserID"],columns=movies["MovieID"])
    for i in ratings.index:
      utility_matrix.at[ratings.at[i,"UserID"],ratings.at[i,"MovieID"]] = ratings.at[i,"Rating"]
    
    start_time = time()
    utility_matrix_sparse = csc_matrix(utility_matrix.fillna(0).values.astype(np.int))
    print("saving generated utility_matrix to pickle file...")
    with open("utility_matrix_sparse.pickle", 'wb') as file_l:
      pickle.dump(utility_matrix_sparse, file_l)
    print("saved to utility_matrix_sparse.pickle")
    end_time = time()
    print("Total Time Taken :",(end_time - start_time))



if __name__=="__main__":
  main()