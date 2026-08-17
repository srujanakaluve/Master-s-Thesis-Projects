import numpy as np

def sign(x):
    """
    Implements the sign function.
    x can be a scalar, 1d list or any dimensional np array.
    """
    y = np.sign(x)
    y = np.where(y == 0, 1, y)
    return y

def relu(x):
    """Relu function vectorized"""
    return np.where(x > 0, x, 0)

def gridtogrid(exc, inh, lambdas):
    """
    Makes the weight matrix for grid CAN dynamics with self excitation and lateral inhibition
    """
    lambdas = np.asarray(lambdas)
    Ng = np.sum(lambdas*lambdas)
    W_gg = np.zeros((Ng, Ng))
    i = 0

    for lam in lambdas:
        size = lam**2
        W_gg[i:i+size, i:i+size] = inh
        i += size

    np.fill_diagonal(W_gg, exc)
    return W_gg


def scaffold_layers(Ng, Nh, Npatts, lambdas, gamma, theta=0.5):
    """
    Makes the scaffold layers: grid states matrix, the corresponding hippocampal layer patterns, with associated weights

    Inputs
    Ng: Number of grid units (sum of lambdas squared)
    Nh: Number of hippocampal units
    Npatts: Number of patterns, should be total number of patterns = product of lambdas squared
    lambdas: list of grid periods
    gamma: sparsity
    theta: bias 

    Returns
    grid: grid state matrix of shape (Ng, patts_total)
    W_hg: grid to hc weights (random, sparse) of shape (Nh, Ng)
    hc: hc activations corresponding to each grid state, shape (Nh, patts_total)
    W_gh: hc to grid weights of shape (Ng, Nh)

    """

    # g tp hc weights are random sparse
    W_hg = np.random.normal(0, 1, size=(Nh, Ng))

    if gamma != 0:
        prune = int((1 - gamma) * Nh * Ng)
        a, b = np.random.randint(low=0, high=Nh, size=prune), np.random.randint(low=0, high=Ng, size=prune)
        W_hg[a, b] = 0

    # get grid fixed points
    lambda_sq = lambdas * lambdas
    grid = generate_grid(lambda_sq)

    # get hc layer fixed points
    hc = relu(W_hg @ grid - theta)

    # learn hc to g weights
    W_gh = (1/Npatts) * (grid @ hc.T)

    return grid, W_hg, hc, W_gh

def generate_grid(lambda_sq):
    """
    Makes a matrix of all possible grid states accounting for modules
    lambda_sq: squared list of lambdas (grid periods). Each element is the total number of units within that module
    """
    Ng = np.sum(lambda_sq)
    patts_total = np.prod(lambda_sq)
    grid = np.zeros((Ng, patts_total))
    jumps = [0] +list(np.cumsum(lambda_sq))[:-1]

    for i in range(patts_total):
        a = np.mod(i, lambda_sq)
        grid[a+jumps, i] = 1

    return grid

def sensory_weights(sensory, hc):
    """
    Makes the weight matrices from sensory to hc layer using pseudoinverse
    sensory: matrix of random (-1 or 1) patterns of shape (Npatts, Ns)
    hc: matrix of all hc states corresponding to grid states, shape (Nh, patts_total)

    Returns
    W_hs: sensory to hc weights of shape (Nh, Ns)
    W_sh: hc to sensory weights of shape (Ns, Nh)

    """
    N_patts = sensory.shape[1]

    #scaffold has n choose k points but we are only storing N_patts patterns, so learn weights with N_patts learned patterns only
    hc_till_Npatts = hc[:, :N_patts]

    W_hs = hc_till_Npatts @ np.linalg.pinv(sensory)
    W_sh = sensory @ np.linalg.pinv(hc_till_Npatts)

    return W_hs, W_sh