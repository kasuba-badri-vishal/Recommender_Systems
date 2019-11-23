from svd import *
import time

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
        error = np.linalg.norm(E)/(M.shape[0]*M.shape[1])
        cost_arr.append(error)
        print(f"iter={iterat}  error={error}")
    return (cost_arr,p, q)

if __name__=='__main__':
    # loading utility matrix from pickle file
    starttime = time.time()
    utility_matrix = pd.read_pickle("utility_matrix.pickle").fillna(0)
    print("got pickle in ", time.time() - starttime)

    m = utility_matrix.shape[0]
    n = utility_matrix.shape[1]

    # number of latent factors
    r = 1000
    p = np.random.randn(m, r)
    q = np.random.randn(r, n)
    print("started gradient descent")
    starttime = time.time()
    cost_arr, p, q = gradient_descent(utility_matrix.values, p, q, 8e-5, 0.1, 400)
    print("gradient descent finished in ", time.time() - starttime)

    M = utility_matrix.values
    reconstr = p@q
    print("Mean Squared error: ", np.linalg.norm(M-reconstr)/(M.shape[0]*M.shape[1]))
    print("Mean Average error: ", np.linalg.norm(M-reconstr, 1)/(M.shape[0]*M.shape[1]))
    np.linalg.norm(M - p@q)
    import matplotlib.pyplot as plt
    plt.title("mse vs iteration plot")
    plt.xlabel("iterations")
    plt.ylabel("reconstruction mse")
    plt.plot(cost_arr[:60])
