import time
import numpy as np

def get_svd(M, energy=0.9):
	"""performs singular valu decomposition

    Parameters
    ----------
    M: numpy.ndarray
        The matrix that needs to be decomposed.
    energy: float, optional
        The energy threshold for performing dimensionality
		reduction.
    
    Returns
    -------
    U numpy.ndarray, sigma numpy.ndarray, VT numpy.ndarray
        U is the left singular matrix
		sigma is a diagonal matrix
		VT is the right singular matrix
    """

	m = M.shape[0]
	n = M.shape[1]
	# shape of MMT = m x m
	MMT = np.matmul(M, M.T)
	# shape of MMT = n x n
	MTM = np.matmul(M.T, M)
	_, U = np.linalg.eig(MMT)
	eigenvals, V = np.linalg.eig(MTM)
	sigma = np.zeros(shape = (m, n))
	for i in range(eigenvals.shape[0]):
		temp = 0 if eigenvals[i]<1e-10 else eigenvals[i]
		sigma[i][i] = np.sqrt(temp)

	cur_energy = 0
	tot_energy = sum(eigenvals)
	# number of eigenvalues with energy greater than threshold
	num = 0
	for i in range(eigenvals.shape[0]):
		cur_energy += eigenvals[i]
		if (cur_energy/tot_energy)>=energy:
			num = i
			break

	for i in range(m-1, num, -1):
		# removing columns from U and rows from sigma
		U = np.delete(U, i, 1)
		sigma = np.delete(sigma, i, 0)
	for i in range(n-1, num, -1):
		# removing columns from V and sigma
		V = np.delete(V, i, 1)
		sigma = np.delete(sigma, i, 1)

	return U,sigma,V.T