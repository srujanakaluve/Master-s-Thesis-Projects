import numpy as np

def relu(x):
    """Relu function vectorized"""
    return np.where(x > 0, x, 0)

def tanh(x, beta):
    return np.tanh(beta * x)

def softmax(x, lambdas, beta=10):
    x1 = x[:, None] if x.ndim==1 else x

    lambdas = np.asarray(lambdas)
    y = np.zeros_like(x1)
    i = 0
    for lam in lambdas:
        size = lam**2
        module = x1[i:i+size, :]
        exp = np.exp(beta * module)
        y[i:i+size, :] = exp / np.sum(exp, axis=0, keepdims=True)
        i += size

    return y[:,0] if x.ndim==1 else y

def glob_inh(x, lambdas, inh_stength):
    x1 = x.copy()
    if x.ndim == 1:
        x1 = x[:, None]

    lambdas = np.asarray(lambdas)
    y = np.zeros_like(x1)
    i = 0
    for lam in lambdas:
        size = lam**2
        module = x1[i:i+size, :]
        module_pos = np.where(module > 0, module, 0)
        sq_activity = module_pos*module_pos
        y[i:i+size, :] = sq_activity / (1 + inh_stength * np.sum(sq_activity, axis=0, keepdims=True))
        i += size

    if x.ndim == 1:
        return y[:, 0]
    return y

def get_overlap(state, pattern):
    """Returns overlap between two states, vectorized"""
    return np.mean(state * pattern, axis=0)

def get_mutual_info(state, pattern):
    """Returns mutual information between two states, vectorized"""
    m = np.array(get_overlap(state, pattern))
    a, b = (1 + m)/2, (1 - m)/2
    s = a * np.log2(np.where(a > 0, a, 1)) + b* np.log2(np.where(b > 0, b, 1))
    return 1 + s

def noisy_state(state, noise):
    """
    Adds gaussian (0,1) noise to hc state.
    noise: scaling parameter
    """
    
    noisy = state + noise * np.random.randn(*state.shape)
    return noisy
