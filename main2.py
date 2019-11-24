# from svd import SVD
from collaborative2 import Collaborative
from baseline import Baseline

import pandas as pd
import os
import time
from scipy.sparse import csc_matrix
import numpy as np
import pickle

def load_sparse_pickle():
  users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
  movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])
  utility_matrix_sparse = pd.read_pickle("utility_matrix_sparse.pickle")
  utility_matrix = pd.DataFrame(utility_matrix_sparse.todense(),index=users["UserID"],columns=movies["MovieID"], dtype=np.float)
  utility_matrix.replace(0, np.nan, inplace=True)
  return utility_matrix

if __name__=='__main__':
  utility_matrix = None
  users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
  movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])

  if os.path.exists("utility_matrix_sparse.pickle"):
    
    starttime = time.time()
    
    # utility_matrix_sparse = pd.read_pickle("utility_matrix_sparse.pickle")
    # utility_matrix = pd.DataFrame(utility_matrix_sparse.todense(),index=users["UserID"],columns=movies["MovieID"])
    utility_matrix = load_sparse_pickle()
    print("Using already created utility_matrix_sparse.pickle file. Time taken: ", time.time()-starttime)
    # print(utility_matrix)
    colab = Collaborative(utility_matrix)
    similarity_matrix = colab.get_normalized_cosine_similarity()
    print("Calculated Similarity Matrix")
    print("Take Input of user ID and Movie ID to predict the Rating")
    query_user = int(input("Query User ID : "))
    query_movie = int(input("Query Movie ID : "))
    n = 20
    predicted_rating = colab.get_rating(query_user,query_movie,similarity_matrix,N=20)
    print(predicted_rating)
    # if os.path.exists("baseline_matrix.pickle"):
    #   baseline = Baseline(utility_matrix)
    #   baseline_matrix = pd.read_pickle("baseline_matrix.pickle")
    #   print("Using already created baseline_matrix.pickle file")
    #   # print(baseline_matrix)
    #   predicted_rating = baseline.get_baseline_rating_prediction(query_user,query_movie,similarity_matrix,baseline_matrix,n)
    #   print("Predicted Rating using baseline approach : ",predicted_rating)
    # else:
    #   print("Baseline Method Started")
    #   start_time = time.time()
    #   baseline = Baseline(utility_matrix)
    #   baseline_matrix = baseline.get_rating_deviation()
    #   end_time = time.time()
    #   print("Time Taken for generating sim_matrix : ",(end_time-start_time))
    #   predicted_rating = baseline.get_baseline_rating_prediction(query_user,query_movie,similarity_matrix,baseline_matrix,n)
    #   print("Predicted Rating using baseline approach : ",predicted_rating)
    

  else:
    start_time = time.time()
    users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
    movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])
    ratings = pd.read_csv('dataset/ratings.dat',delimiter='::',engine='python',names = ["UserID","MovieID", "Rating","Timestamp"],usecols = ["UserID","MovieID","Rating"])


    utility_matrix = pd.DataFrame(index=users["UserID"],columns=movies["MovieID"])
    for i in ratings.index:
      utility_matrix.at[ratings.at[i,"UserID"],ratings.at[i,"MovieID"]] = ratings.at[i,"Rating"]
    
    start_time = time.time()
    utility_matrix_sparse = csc_matrix(utility_matrix.fillna(0).values.astype(np.int))
    print("saving generated utility_matrix to pickle file...")
    # with open('')
    with open("utility_matrix_sparse.pickle", 'wb') as file_l:
      pickle.dump(utility_matrix_sparse, file_l)
    # utility_matrix_sparse.to_pickle("utility_matrix_sparse.pickle")
    print("saved to utility_matrix_sparse.pickle")
    end_time = time.time()
    print("Total Time Taken :",(end_time - start_time))

