"""
This module has functions required to perform CUR decomposition

CUR decomposition is a sparse matrix approximation of SVD decomposition. CUR 
tries to maintain as much data as possible using sparse matrices as opposed to SVD
"""

import pandas as pd
import numpy as np
import random

from svd import get_svd


def get_rand_selection(M, r):
    """Return random selection of rows and cols based of probabilities of frobenius norm
    TODO: edit this accordingly

    Parameters
    ----------
    M: pandas.DataFrame
        input matrix
    r: int
        no of rows and cols to select(features)

    Returns
    -------
    tuple
        tuple containing row and col lists with selected rows and cols
    """

    # frobenius norm of total matrix
    f = (M**2).sum().sum()

    # find probability of each row being selected using the frobenius norm
    # row_prob: list of probabilites for each row -> row_prob[i]: prob of ith row
    row_prob = [ (M.loc[i,:]**2).sum()/f for i in range(M.shape[0]) ]
    # col_prob: list of probabilites for each col -> col_prob[i]: prob of ith col
    col_prob = [ (M[i]**2).sum()/f for i in range(M.shape[1]) ]

    # return [np.random.randint(0, M.shape[0]) for i in range(r)] # just randomly return rows irrespective of prob_dist
    return (random.choices([i for i in range(M.shape[0])], row_prob, k=r), 
            random.choices([i for i in range(M.shape[1])], col_prob, k=r),
            row_prob,
            col_prob )


def moore_penrose_psuedoinverse(E):
    """Return the Moore Penrose Psuedoinverse of the given Matrix

    Parameters
    ----------
    E: np.ndarray
        The sigma matrix returned after performing SVD on matrix W
    
    Returns
    -------
    np.ndarray
        returns Moore Penrose Psuedoinverse of the input matrix
    """

    E_plus = E.copy()
    for i in range(min(E_plus.shape[0], E_plus.shape[1])):
        if E_plus[i][i] != 0:
            E_plus[i][i] = (1/E_plus[i][i])
    return E_plus.transpose()


def get_cur(M, energy=1):
    """This function performs CUR decomposition on the input matrix

    Parameters
    ----------
    M: pandas.DataFrame
        input matrix
    energy: float, optional
        The threshold to perform CUR decomposition

    Returns
    -------
    (C, U, R): tuple
        tuple containing CUR decompostion matrices
    """

    r = 2 # TODO: think about this later

    rows, cols, row_prob, col_prob = get_rand_selection(M, r)
    # DEBUG: to set custom rows and cols if required(Debugging)
    # rows = [3,1]
    # cols = [1,0]

    # check for repeated rows and cols
    row_count = dict()
    for i in rows:
        if i not in row_count: row_count[i] = 1
        else: row_count[i] += 1
    col_count = dict()
    for i in cols:
        if i not in col_count: col_count[i] = 1
        else: col_count[i] += 1
    
    rows = list(row_count.keys())   # get unique rows
    cols = list(col_count.keys())   # get unique cols

    # get selected rows/cols
    # divide with their prob of selection to normalize
    # if multiple same k no of rows are selected, then multiply the row by sqrt(k)
    C = M.filter(cols, axis=1).copy().astype(np.float)
    for j in C.columns:
        C.loc[:,j] = ( C.loc[:,j]/( (r*col_prob[j])**0.5 ) ) * (col_count[j]**0.5)
    
    R = M.filter(rows, axis=0).copy().astype(np.float)
    for i in R.index:
        R.loc[i] = ( R.loc[i]/((r*row_prob[i])**0.5) ) * (row_count[i]**0.5)

    C = C.values    # convert to numpy array
    R = R.values    # convert to numpy array

    W = M.filter(rows, axis=0).filter(cols, axis=1)

    # our custom svd module
    # X, E, Y_t = get_svd(W, energy=1.0)

    ###### Using numpy for svd #####
    # comment this whole section to remove numpy svd
    X, s, Y_t = np.linalg.svd(W)
    E = np.zeros((X.shape[1], Y_t.shape[0])).astype(np.float)
    E[:s.shape[0], :s.shape[0]] = np.diag(s)
    ###### end of numpy for svd ####

    E_plus = moore_penrose_psuedoinverse(E)
    Y = Y_t.transpose()
    U = Y.dot(E_plus**2).dot(X.transpose())

    print("Original: \n", M, end='\n\n')
    print("Cols:", cols, end='\n')
    print("Rows:", rows, end='\n\n')
    # print(row_prob)
    # print(col_prob)
    print("C:\n", C, end='\n\n')
    print("U:\n", U, end='\n\n')
    print("R:\n", R, end='\n\n')
    print("CUR:\n", C@U@R, end='\n\n')
    print("Error:\n", C@U@R - M.values, end='\n\n')