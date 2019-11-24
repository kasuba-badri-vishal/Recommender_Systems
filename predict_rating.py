import numpy as np
import pandas as pd
import warnings
from time import time



utility_matrix = pd.read_pickle("utility_matrix.pickle")

M = utility_matrix.to_numpy()

usr_mean = utility_matrix.mean(axis=1)
item_mean = utility_matrix.mean(axis=0)
mu = np.mean(item_mean)
baseline_matrix = pd.DataFrame(0,index =utility_matrix.index,columns = utility_matrix.columns,dtype=np.float)
baseline_matrix = baseline_matrix.add(item_mean,axis=1) + baseline_matrix.add(usr_mean,axis=0) - mu
usr_mean = usr_mean.to_numpy().reshape(-1,1)
item_mean = item_mean.to_numpy().reshape(-1,1)

U = np.nan_to_num(M - usr_mean)

temp = pd.DataFrame(U).fillna(0).to_numpy()

norm_users = temp / np.sqrt(np.square(temp).sum(axis=1))[:,None]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    norm_items = temp / np.sqrt(np.square(temp).sum(axis=0))
norm_users = np.nan_to_num(norm_users)
norm_items = np.nan_to_num(norm_items)

user_sim_mat = norm_users.dot(norm_users.T)
item_sim_mat = norm_items.T.dot(norm_items)



def main(user=None, item=None):
    if user is None:
        user = int(input("Input user ID: "))
    if item is None:
        item = int(input("Input movie ID: "))



    # ===========================================================================
    # Performing user-user similarity analysis
    k = 20 # neighbours

    # people who have seen the movie
    seen_users = utility_matrix[item].dropna().index
    indices = np.argsort(user_sim_mat[user])[::-1]
    top_k_neigh = list()
    for i in range(len(indices)):
        if indices[i] in seen_users and i != user:
            top_k_neigh.append(indices[i])
        if len(top_k_neigh) == k: break

    pred = np.sum(user_sim_mat[user, top_k_neigh]*utility_matrix.loc[top_k_neigh,item].to_numpy())/np.sum(user_sim_mat[user, top_k_neigh])
    s = 0
    for i in top_k_neigh:
        s += user_sim_mat[user, i]*(utility_matrix.at[i,item] - usr_mean[i][0])
    s = s/np.sum(user_sim_mat[user, top_k_neigh])
    s += usr_mean[user][0]
    print("Query user mean rating:", usr_mean[user])
    print("Query item mean rating:", item_mean[item])
    print()
    print("Using user-user Collabarative Approach")
    print("normal Predicted Rating:", pred)
    print("Unbiased Predicted Rating:", s)
    print("top k similar user items mean:", utility_matrix.loc[top_k_neigh, item].mean())


    # ===========================================================================
    #Performing item-item similarity analysis
    k = 20 # neighbours

    # items which have been seen by the user
    seen_items = utility_matrix.loc[user].dropna().index
    indices = np.argsort(item_sim_mat[item])[::-1]
    top_k_neigh = list()
    for i in range(len(indices)):
        if indices[i] in seen_items and i != item:
            top_k_neigh.append(indices[i])
        if len(top_k_neigh) == k: break
    
    if len(top_k_neigh) == 0: return 0
    test = np.sum(item_sim_mat[item, top_k_neigh])
    if test == 0: return 0
    pred = np.sum(item_sim_mat[item, top_k_neigh]*utility_matrix.loc[user,top_k_neigh].to_numpy())/test
    print()
    print("Using item-item Collaborative Approach")
    print("predicted Rating:", pred)
    print("top k similar movie item mean:", utility_matrix.loc[user, top_k_neigh].mean())
    
    #Baseline Approach
    pred = baseline_matrix.at[user,item]
    temp3 = utility_matrix.loc[user,top_k_neigh] - baseline_matrix.loc[user,top_k_neigh]
    pred += np.sum(item_sim_mat[item, top_k_neigh]*temp3.to_numpy())/test
    print()
    print("Rating using Baseline method", pred)



   


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print("Time taken", end_time-start_time)