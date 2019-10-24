import pandas as pd
#from pandas import read_pickle
import os
import time

utility_matrix = None
if os.path.exists("utility_matrix.pickle"):
  utility_matrix = pd.read_pickle("utility_matrix.pickle")
  print("Using already created utility_matrix.pickle file")
  print(utility_matrix.shape)
  print(utility_matrix)

else:
  start_time = time.time()
  users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
  ratings = pd.read_csv('dataset/ratings.dat',delimiter='::',engine='python',names = ["UserID","MovieID", "Rating","Timestamp"],usecols = ["UserID","MovieID","Rating"])
  movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])


  utility_matrix = pd.DataFrame(index=users["UserID"],columns=movies["MovieID"])
  # utility_matrix = utility_matrix.fillna()
  for i in ratings.index:
    utility_matrix.loc[ratings["UserID"][i], ratings["MovieID"][i]] = ratings["Rating"][i]
  
  print("saving generated utility_matrix to pickle file...")
  utility_matrix.to_pickle("utility_matrix.pickle")
  print("saved to utility_matrix.pickle")
  end_time = time.time()
  print("Total Time Taken :",(end_time - start_time))

