from svd import SVD
from collaborative import Collaborative

import pandas as pd
import os
import time

utility_matrix = None
if os.path.exists("utility_matrix.pickle"):
  utility_matrix = pd.read_pickle("utility_matrix.pickle")
  print("Using already created utility_matrix.pickle file")
  print(utility_matrix.shape)

  colab = Collaborative(utility_matrix)
  similarity_matrix = colab.get_normalized_cosine_similarity()
  print("Calculated Similarity Matrix")
  print("Take Input of user ID and Movie ID to predict the Rating")
  query_user = int(input("Query User ID : "))
  query_movie = int(input("Query Movie ID : "))
  n = 20
  predicted_rating = colab.get_rating(query_user,query_movie,similarity_matrix,n)
  print("Predicted Rating : ",predicted_rating)
  

else:
  start_time = time.time()
  users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
  ratings = pd.read_csv('dataset/ratings.dat',delimiter='::',engine='python',names = ["UserID","MovieID", "Rating","Timestamp"],usecols = ["UserID","MovieID","Rating"])
  movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])


  utility_matrix = pd.DataFrame(index=users["UserID"],columns=movies["MovieID"])
  # utility_matrix = utility_matrix.fillna()
  for i in ratings.index:
    utility_matrix.at[ratings.at[i,"UserID"],ratings.at[i,"MovieID"]] = ratings.at[i,"Rating"]
  
  print("saving generated utility_matrix to pickle file...")
  utility_matrix.to_pickle("utility_matrix.pickle")
  print("saved to utility_matrix.pickle")
  end_time = time.time()
  print("Total Time Taken :",(end_time - start_time))

