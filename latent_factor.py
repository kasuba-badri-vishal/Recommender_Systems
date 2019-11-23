from svd import *
import time
from main import load_sparse_pickle

def gradient_descent(M, p, q, alpha, lmbda, max_iter=100):
    '''
    This function runs stochastic gradient descent to optimize
    the factors p and q.
    returns cost array, new p and new q as a tuple.
    '''
    cost_arr = []
    for iterat in range(max_iter):
        E = M - (p@q)
        q = q + alpha * (p.T @ E - lmbda*q)
        p = p + alpha * (E @ q.T - lmbda*p)
        error = np.square(E).sum()/(M.shape[0]*M.shape[1])
        cost_arr.append(np.sqrt(error))
        print(f"iter={iterat}  error={error}")
    return (cost_arr,p, q)

if __name__=='__main__':
    # loading utility matrix from pickle file
    starttime = time.time()
    # utility_matrix = pd.read_pickle("utility_matrix.pickle").fillna(0)
    utility_matrix = load_sparse_pickle().fillna(0)
    print("got pickle in  ", time.time() - starttime)

    m = utility_matrix.shape[0]
    n = utility_matrix.shape[1]

    # number of latent factors
    r = 2000
    p = np.random.randn(m, r)
    q = np.random.randn(r, n)
    print("started gradient descent")
    starttime = time.time()
    cost_arr, p, q = gradient_descent(utility_matrix.values, p, q, 8e-5, 0.1, 400)
    print("gradient descent finished in ", time.time() - starttime)

    M = utility_matrix.values
    reconstr = p@q
    print("Root Mean Squared error: ", np.sqrt(np.square(M-reconstr).sum()/(M.shape[0]*M.shape[1])))
    print("Mean Average error: ", np.abs(M-reconstr).sum()/(M.shape[0]*M.shape[1]))

    num = 60
    import matplotlib.pyplot as plt
    plt.title("rmse vs iteration plot")
    plt.xlabel("iterations")
    plt.ylabel("reconstruction rmse")
    plt.plot(cost_arr[:num])
    plt.savefig(f"./temp/latent_{r}_{num}.png")
