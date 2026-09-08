import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

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

def sigmoid(x, lambdas, beta=10):
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
        outputs.append(jax.nn.sigmoid(beta * module))
        start += size

    y = jnp.concatenate(outputs, axis=0)
    if was_1d:
        return y[:, 0]
    return y

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

class Grid:

    def __init__(self, lambdas, activation, gg_exc=None, gg_inh=None,
                 tau_g=1., dt=0.1, gg_default=True, W_gg=None):
        self.lambdas = tuple(int(l) for l in lambdas)
        self.Ng = sum(l * l for l in self.lambdas)
        self.patts_total = np.prod([l * l for l in self.lambdas])
        if gg_exc is not None and gg_inh is not None:
            self.gg_exc, self.gg_inh = gg_exc, gg_inh
        self.tau_g = tau_g
        self.dt = dt

        if gg_default:
            self.W_gg = self.gridtogrid()
        elif gg_default is False and W_gg is not None:
            self.W_gg = W_gg

        if activation=='softmax':
            self.activation = softmax
        if activation=='glob_inh':
            self.activation = glob_inh
        if activation=='sigmoid':
            self.activation = sigmoid

        self.grid = self.generate_grid()

        self.weights = {'W_gg': jnp.array(self.W_gg)}

    def gridtogrid(self):
        """
        Makes the weight matrix for grid CAN dynamics with self excitation and lateral inhibition.
        Returns a bp array
        """
        W_gg = np.zeros((self.Ng, self.Ng))
        i = 0

        for lam in self.lambdas:
            size = lam**2
            W_gg[i:i+size, i:i+size] = self.gg_inh
            i += size

        np.fill_diagonal(W_gg, self.gg_exc)
        return W_gg   

    def generate_grid(self):
        """
        Makes a matrix of all possible grid states accounting for modules
        """
        lambda_sq = np.array([l * l for l in self.lambdas])
        grid = np.zeros((self.Ng, self.patts_total))
        jumps = [0] +list(np.cumsum(lambda_sq))[:-1]

        for i in range(self.patts_total):
            a = np.mod(i, lambda_sq)
            grid[a+jumps, i] = 1

        return grid

    @staticmethod
    @partial(jax.jit, static_argnames=("lambdas", "activation"))
    def simulate_run(g0, weights, lambdas, tau_g, activation, dt=0.1):
        state0 = (g0)

        def rk4_step(state, _):
            g = state

            def derivative(g_t):
                dg = (-g_t + activation(weights['W_gg'] @ g_t, lambdas)) / tau_g
                return dg

            dg1 = derivative(g)
            dg2 = derivative(g + 0.5 * dt * dg1)
            dg3 = derivative(g + 0.5 * dt * dg2)
            dg4 = derivative(g + dt * dg3)
            g_next = g + (dt / 6.0) * (dg1 + 2*dg2 + 2*dg3 + dg4)
            return g_next, None

        final_state, _ = jax.lax.scan(rk4_step, state0, None, length=500)
        return final_state

    def run(self, g0):
            return self.simulate_run(jnp.array(g0), self.weights, self.lambdas, self.tau_g, self.activation, dt=self.dt)

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    lambdas = np.array([3,4,5])
    scaffold = Grid(lambdas, 'sigmoid', 4, -1)
    g0 = scaffold.grid
    exc_range = np.arange(0, 101)
    inh_range = np.arange(0, -101, -1)

    #wggs = gg_space(scaffold.Ng, lambdas, exc_range, inh_range, 'g2g_space.npy')
    final = scaffold.run(g0)

    n_correct = sum(np.allclose(final[:, p], g0[:, p], atol=1e-1) for p in range(3600))
    print(n_correct)
    print(final[:,0])