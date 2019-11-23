import time
import numpy as np
import pandas as pd

def get_svd(M, energy=1):
	"""
	Performs singular value decomposition

    Parameters
    ----------
    M: numpy.ndarray
        The matrix that needs to be decomposed.
    energy: float, optional
        The energy threshold for performing dimensionality
		reduction.
    
    Returns
    -------
    U numpy.ndarray, sigma numpy.ndarray, V.T numpy.ndarray
        U is the left singular matrix
		sigma is a diagonal matrix
		V.T is the right singular matrix
    """

	m = M.shape[0]
	n = M.shape[1]
	# shape of MMT = m x m
	MMT = np.matmul(M, M.T)
	# shape of MTM = n x n
	MTM = np.matmul(M.T, M)
	_, U = np.linalg.eigh(MMT)
	eigenvals, V = np.linalg.eigh(MTM)

	for i in range(eigenvals.shape[0]):
		eigenvals[i] = 0 if eigenvals[i]<1e-6 else eigenvals[i]

	num_eig = eigenvals.shape[0]
	eigenvals = eigenvals[::-1]

	sigma = np.zeros((max(num_eig, U.shape[1]), max(num_eig, V.shape[1])))
	sigma[:num_eig][:num_eig] = np.diag(np.sqrt(eigenvals))
	sigma = sigma[:U.shape[1], :V.shape[1]]

	# arranging eigen vectors in decreasing order
	U = U[:, ::-1]
	V = V[:, ::-1]

	# print(U.shape, sigma.shape, V.shape)
	# print("error before reduction: ", np.linalg.norm(M-U@sigma@(V.T)))
	if energy<1:
		U,sigma,V = reduce_dim(U, sigma, V, eigenvals, energy)

	print(U.shape,sigma.shape, V.shape)
	print("error after reduction: ", np.linalg.norm(M-U@sigma@(V.T)))

	return U,sigma,V.T

def reduce_dim(U, sigma, V, eigenvals, energy):
	'''
	This function reduces the dimensions of 
	the decomposition according to the given energy.
	'''
	i = 0
	while i<min(sigma.shape[0],sigma.shape[1]):
		# removing columns from U and rows from sigma
		if sigma[i][i]==0:
			U = np.delete(U, i, 1)
			V = np.delete(V, i, 1)
			sigma = np.delete(sigma, i, 0)
			sigma = np.delete(sigma, i, 1)
			i-=1
		i+=1

	cur_energy = 0
	tot_energy = sum(eigenvals)
	# number of eigenvalues with energy greater than threshold
	num = 0
	for i in range(eigenvals.shape[0]):
		cur_energy += eigenvals[i]
		if (cur_energy/tot_energy)>=energy:
			num = i
			break
	U = U[:,:num+1]
	V = V[:,:num+1]
	sigma = sigma[:num+1, :num+1]
	return U,sigma,V

def libsvd(M, energy=1):
	'''
	SVD using numpy library function.
	'''
	U,d,VT = np.linalg.svd(M)
	s = np.zeros((U.shape[1], VT.shape[0]))
	s[:d.shape[0],:d.shape[0]] = np.diag(d)
	if energy<1:
		U, s, V = reduce_dim(U, s, VT.T, d, energy)
		VT = V.T
	print(U.shape, s.shape, VT.shape)
	reconstr = U@s@VT
	print("SVD Mean Squared error: ", np.linalg.norm(M-reconstr)/(M.shape[0]*M.shape[1]))
	print("SVD Mean Average error: ", np.linalg.norm(M-reconstr, 1)/(M.shape[0]*M.shape[1]))
	return U, s, VT

def get_new_ratings(userRatings, decomp):
	'''
	This function, when given ratings of new users
	atleast one movie, gives ratings for other movies.
	'''
	# U, sigma, VT = libsvd(M, energy)
	V = decomp[2].T
	res = userRatings @ V @ V.T
	return res

if __name__=='__main__':
	starttime = time.time()
	utility_matrix = pd.read_pickle("utility_matrix.pickle").fillna(0)
	print("got pickle in ", time.time() - starttime)

	starttime = time.time()
	decomp = libsvd(utility_matrix.values, 0.9)
	print("got svd in ", time.time() - starttime)

	starttime = time.time()
	newRatings = get_new_ratings(utility_matrix.values, decomp)
	print("got prediction in ", time.time() - starttime)
	