import numpy as np
import jax
import jax.numpy as jnp

def get_overlap(state, pattern):
    """Returns overlap between two states, vectorized"""
    return np.mean(state * pattern, axis=0)

def get_mutual_info(state, pattern):
    """Returns mutual information between two states, vectorized"""
    m = np.array(get_overlap(state, pattern))
    a, b = (1 + m)/2, (1 - m)/2
    s = a * np.log2(np.where(a > 0, a, 1)) + b* np.log2(np.where(b > 0, b, 1))
    return 1 + s

def softmax(x, lambdas, beta=10.0):
    x = jnp.asarray(x)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    lambdas = tuple(int(l) for l in lambdas)
    out = []
    start = 0

    for lam in lambdas:
        size = int(lam ** 2)
        block = x[start:start + size]
        out.append(jax.nn.softmax(beta * block, axis=0))
        start += size

    y = jnp.concatenate(out, axis=0)
    if was_1d:
        return y[:, 0]
    return y


def sigmoid(x, lambdas=None, beta=10):
    return jax.nn.sigmoid(beta * x)

def glob_inh(x, lambdas, inh_strength=1):
    x = jnp.asarray(x)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]

    lambdas = tuple(int(l) for l in lambdas)
    outputs = []
    start = 0

    for lam in lambdas:
        size = lam ** 2
        module = x[start:start + size, :]
        module_pos = jnp.where(module > 0, module, 0)
        sq_activity = module_pos * module_pos
        denominator = 1 + inh_strength * jnp.sum(
            sq_activity, axis=0, keepdims=True
        )
        outputs.append(sq_activity / denominator)
        start += size

    y = jnp.concatenate(outputs, axis=0)
    if was_1d:
        return y[:, 0]
    return y

def gg_space(Ng, lambdas, exc_range, inh_range, file_path=None):
    """run this once and store in a file"""

    exc_range = np.asarray(exc_range)
    inh_range = np.asarray(inh_range)
    matrices = np.empty((len(exc_range), len(inh_range), Ng, Ng))

    for exc_idx, exc in enumerate(exc_range):
        for inh_idx, inh in enumerate(inh_range):
            matrix = np.zeros((Ng, Ng))
            i = 0
            for lam in lambdas:
                size = lam ** 2
                matrix[i:i + size, i:i + size] = inh
                i += size
            np.fill_diagonal(matrix, exc)
            matrices[exc_idx, inh_idx] = matrix

    if file_path is not None:
        np.save(file_path, matrices)

    return matrices
