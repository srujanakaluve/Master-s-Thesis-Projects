import numpy as np
import matplotlib.pyplot as plt

"""
Helper functions

"""

def sign(x):
    """
    Implements the sign function.
    x can be a scalar, 1d list or any dimensional np array.
    """
    y = np.sign(x)
    y = np.where(y == 0, 1, y)
    return y

def topk(x, k):
    """
    Top-k non-linearity: the top k units are set to 1, rest are 0
    x can be a 1D or 2D numpy array
    k is the number of units to activate
    """
    topk_x = np.zeros_like(x)

    if x.ndim == 1:
        k_indices = np.argsort(x)[-k:]
        topk_x[k_indices] = 1

    elif x.ndim == 2:
        k_indices = np.argsort(x, axis=0)[-k:, :]
        col_indices = np.arange(x.shape[1])
        topk_x[k_indices, col_indices] = 1

    return topk_x

def relu(x):
    """Relu function vectorized"""
    return np.where(x > 0, x, 0)

def get_mutual_info(state, pattern):
    """Returns mutual information between two states, vectorized"""
    m = np.array(get_overlap(state, pattern))
    a, b = (1 + m)/2, (1 - m)/2
    s = a * np.log2(np.where(a > 0, a, 1)) + b* np.log2(np.where(b > 0, b, 1))
    return 1 + s

def get_overlap(state, pattern):
    """Returns overlap between two states, vectorized"""
    return np.mean(state * pattern, axis=0)

""" 
vectorHASH specific functions
"""

def grid_CAN(g, lambdas):
    """
    Module-wise top-k non-linearity
    g: Can be 1d array activity vector or a matrix of activity vectors
    lambdas: list of grid periods
    """
    g_in = g
    if g.ndim == 1:
        g = g[:, None]

    Ng, ncols = g.shape
    g_out = np.zeros_like(g)
    i = 0
    for lam in lambdas:
        size = lam**2
        module = g[i:i+size, :]  # shape (size, ncols)
        winners = np.argmax(module, axis=0)  # shape (ncols,)
        rows = i + winners
        g_out[rows, np.arange(ncols)] = 1
        i += size

    if g_in.ndim == 1:
        return g_out[:, 0]
    return g_out

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

def recall(s_cued, W_hs, W_sh, W_gh, W_hg, lambdas, theta=0.5, Niter=5):
    """
    Runs the recall dynamics from sensory-hc-grid and back
    s_cued: input pattern in sensory layer - can be a single pattern or a matrix of patterns
    theta: bias for hc layer
    Niter: number of grid-grid iterations to run

    Returns s_recalled: (hopefully) correct sensory patterns recalled
    """

    h = relu(W_hs @ s_cued - theta)

    for _ in range(Niter):
        g = grid_CAN(W_gh @ h, lambdas)
        h = relu(W_hg @ g - theta)

    s_recalled = np.sign(W_sh @ h)
    return s_recalled

