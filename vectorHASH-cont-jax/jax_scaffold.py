import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax_helpers import sigmoid, glob_inh, softmax

class Scaffold:

    def __init__(self, Nh, lambdas, activation, gg_exc=None, gg_inh=None, gamma=0, b=0.5,
                 tau_g=1., tau_h=1., dt=0.1, gg_default=True, W_gg=None):
        self.lambdas = tuple(int(l) for l in lambdas)
        self.Ng, self.Nh = sum(l * l for l in self.lambdas), Nh
        self.patts_total = np.prod([l * l for l in self.lambdas])
        if gg_exc is not None and gg_inh is not None:
            self.gg_exc, self.gg_inh = gg_exc, gg_inh
        self.b = b
        self.gamma = gamma
        self.tau_g, self.tau_h = tau_g, tau_h
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
        self.grid, self.W_hg, self.hc, self.W_gh = self.scaffold_layers()

        self.weights = {
            'W_gg': jnp.array(self.W_gg),
            'W_gh': jnp.array(self.W_gh),
            'W_hg': jnp.array(self.W_hg),}

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

    def scaffold_layers(self):
        # g tp hc weights are random sparse
        W_hg = np.random.normal(0, 1, size=(self.Nh, self.Ng))

        if self.gamma != 0:
            prune = int((1 - self.gamma) * self.Nh * self.Ng)
            a, b = np.random.randint(low=0, high=self.Nh, size=prune), np.random.randint(low=0, high=self.Ng, size=prune)
            W_hg[a, b] = 0

        # get grid fixed points
        grid = self.generate_grid()

        # get hc layer fixed points
        hc = jax.nn.relu(W_hg @ grid - self.b)

        # learn hc to g weights
        W_gh = (1/self.patts_total) * (grid @ hc.T)

        return grid, W_hg, hc, W_gh

    @staticmethod
    @partial(jax.jit, static_argnames=("lambdas", "activation"))
    def simulate_run(g0, h0, weights, lambdas, b, tau_g, tau_h, activation, dt=0.1):
        state0 = (g0, h0)

        def rk4_step(state, _):
            g, h = state

            def derivative(g_t, h_t):
                dg = (-g_t + activation(weights['W_gg'] @ g_t + weights['W_gh'] @ h_t, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t - b)) / tau_h
                return dg, dh

            dg1, dh1 = derivative(g, h)
            dg2, dh2 = derivative(g + 0.5 * dt * dg1, h + 0.5 * dt * dh1)
            dg3, dh3 = derivative(g + 0.5 * dt * dg2, h + 0.5 * dt * dh2)
            dg4, dh4 = derivative(g + dt * dg3, h + dt * dh3)
            g_next = g + (dt / 6.0) * (dg1 + 2*dg2 + 2*dg3 + dg4)
            h_next = h + (dt / 6.0) * (dh1 + 2*dh2 + 2*dh3 + dh4)

            return(g_next, h_next), None

        final_state, _ = jax.lax.scan(rk4_step, state0, None, length=250)
        return final_state

    def run(self, g0, h0):
            return self.simulate_run(jnp.array(g0),
                jnp.array(h0), self.weights, self.lambdas, self.b, self.tau_g,
                self.tau_h, self.activation, dt=self.dt)

    def run_and_error(self, g0, h0):
        final_state = self.simulate_run(g0, h0, self.weights, self.lambdas, self.b, self.tau_g, self.tau_h, dt=self.dt)
        error = jnp.linalg.norm(final_state[1] - h0, axis=0)
        return error

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    lambdas = np.array([3,4,5])
    gg = np.load('/home/srujana/VSCode Projects/Thesis/vectorHASH-cont-jax/g2g_space.npy')
    scaffold = Scaffold(400, lambdas, 'glob_inh', gamma=0.6, gg_default=False, W_gg=gg[40, 40, :, :])
    h0 = scaffold.hc
    g0 = jnp.zeros_like(scaffold.grid)
    final = scaffold.run(g0, h0)

    n_correct = sum(np.allclose(final[0][:, p], scaffold.grid[:, p], atol=1e-1) for p in range(3600))
    """plt.imshow(n_correct, aspect='auto')
    plt.colorbar()
    plt.show()"""
    print(n_correct)
    print(final[0][:,0])
    print(gg[40, 40, :, :])
    

"""    mean_hc_norm = np.mean(np.linalg.norm(h0, axis=0))
    noise_vals = np.arange(0, 10, 0.5)
    runs = 50
    correct = np.zeros((runs, len(noise_vals), scaffold.patts_total))

    for i in tqdm(range(runs)):
        for nidx, noise_val in enumerate(noise_vals):
            noise = np.random.normal(0, 1, h0.shape) / np.sqrt(scaffold.Nh)
            h0_noisy = h0 + noise_val * noise * mean_hc_norm

            # Single call to class run method

            err = scaffold.run_and_error(g0, h0_noisy)
            correct[i, nidx] = err

    np.save('correct.npy', correct)
    print("Saved!")

    correct_valid = correct<1
    # average across runs (axis 0)
    correct_avg = correct_valid.mean(axis=0)
    plt.plot(noise_vals, correct_avg.mean(axis=1),'k',lw=2.)
    plt.xlabel(r'|noise|/|hpc|')
    plt.ylabel('p(correct)')
    plt.show()"""