import pandas as pd


users = pd.read_csv('dataset/users.dat',delimiter='::',engine='python',names = ["UserID","Gender","Age","Occupation","Zip-code"],usecols = ["UserID"])
ratings = pd.read_csv('dataset/ratings.dat',delimiter='::',engine='python',names = ["UserID","MovieID", "Rating","Timestamp"],usecols = ["UserID","MovieID","Rating"])
movies = pd.read_csv('dataset/movies.dat',delimiter='::',engine='python',names = ["MovieID","Title","Genres"])


# print(users)
# print(ratings)
# print(movies)
utility_matrix = pd.DataFrame(index=users["UserID"],columns=movies["MovieID"])
utility_matrix = utility_matrix.fillna(0)
print(utility_matrix.shape)
# utility_matrix[ratings["UserID"][0]][ratings["MovieID"][0]] = ratings["Rating"][0]
for i in range(ratings.shape[0]):
  utility_matrix[ratings["UserID"][i]][ratings["MovieID"][i]] = ratings["Rating"][i]

# #     utility_matrix[ratings["UserID"][i]][ratings["MovieID"][i]] = ratings["Rating"][i]
# print(utility_matrix[1][48])