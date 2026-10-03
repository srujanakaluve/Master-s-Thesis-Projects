import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax_helpers import sigmoid, glob_inh, softmax

class VectorHaSH:

    def __init__(self, Nh, Ns, Npatts, lambdas, activation, gg_exc=None, gg_inh=None, beta=10, gamma=0, b=0.5,
                 tau_g=1., tau_h=1., tau_s=1., dt=0.01, gg_default=True, W_gg=None, seed=None):
        
        self.lambdas = tuple(int(l) for l in lambdas)
        self.Ng, self.Nh, self.Ns = sum(l * l for l in self.lambdas), Nh, Ns
        self.patts_total = np.prod([l * l for l in self.lambdas])
        if gg_exc is not None and gg_inh is not None:
            self.gg_exc, self.gg_inh = gg_exc, gg_inh
        self.b = b
        self.gamma = gamma
        self.beta = beta
        self.tau_g, self.tau_h, self.tau_s= tau_g, tau_h, tau_s
        self.dt = dt
        self.rng = np.random.default_rng(seed)

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
        self.sensory = np.sign(self.rng.standard_normal((self.Ns, Npatts)))
        self.W_hs, self.W_sh = self.sensory_weights()

        self.weights = {
            'W_gg': jnp.array(self.W_gg),
            'W_gh': jnp.array(self.W_gh),
            'W_hg': jnp.array(self.W_hg),
            'W_hs': jnp.array(self.W_hs),
            'W_sh': jnp.array(self.W_sh)}

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
        W_hg = self.rng.normal(0, 1, size=(self.Nh, self.Ng))
        if self.gamma != 0:
            prune = int((1 - self.gamma) * self.Nh * self.Ng)
            a = self.rng.integers(0, self.Nh, size=prune)
            b = self.rng.integers(0, self.Ng, size=prune)
            W_hg[a, b] = 0
        grid = self.generate_grid()
        hc = jax.nn.relu(W_hg @ grid - self.b)
        W_gh = (1 / self.patts_total) * (grid @ hc.T)
        return grid, W_hg, hc, W_gh

    def sensory_weights(self):
        N_patts = self.sensory.shape[1]

        #scaffold has n choose k points but we are only storing N_patts patterns, so learn weights with N_patts learned patterns only
        hc_till_Npatts = self.hc[:, :N_patts]

        W_hs = hc_till_Npatts @ np.linalg.pinv(self.sensory)
        W_sh = self.sensory @ np.linalg.pinv(hc_till_Npatts)

        return W_hs, W_sh

    @staticmethod
    @partial(jax.jit, static_argnames=("lambdas", "activation", "n_steps"))
    def simulate_run(g0, h0, s0, weights, lambdas, beta, b, tau_g, tau_h, tau_s,
                     activation, dt=0.01, n_steps=1000):
        state0 = (g0, h0, s0)

        def rk4_step(state, _):
            g, h, s = state

            def derivative(g_t, h_t, s_t):
                dg = (-g_t + activation(weights['W_gg'] @ g_t + weights['W_gh'] @ h_t, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t + weights['W_hs'] @ s_t - b)) / tau_h
                ds = (-s_t + jax.nn.tanh(beta * weights['W_sh'] @ h_t)) / tau_s
                return dg, dh, ds

            dg1, dh1, ds1 = derivative(g, h, s)
            dg2, dh2, ds2 = derivative(
                g + 0.5 * dt * dg1, h + 0.5 * dt * dh1, s + 0.5 * dt * ds1
            )
            dg3, dh3, ds3 = derivative(
                g + 0.5 * dt * dg2, h + 0.5 * dt * dh2, s + 0.5 * dt * ds2
            )
            dg4, dh4, ds4 = derivative(g + dt * dg3, h + dt * dh3, s + dt * ds3)
            g_next = g + (dt / 6.0) * (dg1 + 2*dg2 + 2*dg3 + dg4)
            h_next = h + (dt / 6.0) * (dh1 + 2*dh2 + 2*dh3 + dh4)
            s_next = s + (dt / 6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)

            return (g_next, h_next, s_next), None

        final_state, _ = jax.lax.scan(rk4_step, state0, None, length=n_steps)
        return final_state

    @staticmethod
    @partial(jax.jit, static_argnames=('lambdas', 'activation', 'n_steps'))
    def simulate_run_traj(g0, h0, s0, weights, lambdas, beta, b, tau_g, tau_h,
                             tau_s, activation, dt=0.01, n_steps=1000):
        state0 = (g0, h0, s0)

        def rk4_step(state, _):
            g, h, s = state

            def derivative(g_t, h_t, s_t):
                dg = (-g_t + activation(weights['W_gg'] @ g_t + weights['W_gh'] @ h_t, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t + weights['W_hs'] @ s_t - b)) / tau_h
                ds = (-s_t + jax.nn.tanh(beta * weights['W_sh'] @ h_t)) / tau_s
                return dg, dh, ds

            dg1, dh1, ds1 = derivative(g, h, s)
            dg2, dh2, ds2 = derivative(
                g + 0.5 * dt * dg1, h + 0.5 * dt * dh1, s + 0.5 * dt * ds1
            )
            dg3, dh3, ds3 = derivative(
                g + 0.5 * dt * dg2, h + 0.5 * dt * dh2, s + 0.5 * dt * ds2
            )
            dg4, dh4, ds4 = derivative(g + dt * dg3, h + dt * dh3, s + dt * ds3)
            g_next = g + (dt / 6.0) * (dg1 + 2*dg2 + 2*dg3 + dg4)
            h_next = h + (dt / 6.0) * (dh1 + 2*dh2 + 2*dh3 + dh4)
            s_next = s + (dt / 6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)

            return (g_next, h_next, s_next), (g_next, h_next, s_next)
        
        _, (g_traj, h_traj, s_traj) = jax.lax.scan(
            rk4_step, state0, None, length=n_steps
        )
        return g_traj, h_traj, s_traj

    def run(self, g0, h0, s0, n_steps=1000):
        return self.simulate_run(
            jnp.asarray(g0), jnp.asarray(h0), jnp.asarray(s0), self.weights,
            self.lambdas, self.beta, self.b, self.tau_g, self.tau_h, self.tau_s,
            self.activation, dt=self.dt, n_steps=n_steps
        )

    def run_and_error(self, g0, h0, s0, n_steps=1000):
        final_state = self.run(g0, h0, s0, n_steps=n_steps)
        error = jnp.linalg.norm(final_state[1] - h0, axis=0)
        return error

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    lambdas = np.array([3,4,5])
    #gg = np.load('/home/srujana/VSCode Projects/Thesis/vectorHASH-cont-jax/g2g_space.npy')
    scaffold = VectorHaSH(
        Nh=400, Ns=3600, Npatts=200, lambdas=lambdas, activation='sigmoid',
        gg_exc=20, gg_inh=-30, gamma=0.6)
    
    patt = 0
    h0 = scaffold.hc[:, :200]
    g0 = scaffold.grid[:, :200]
    s0 = scaffold.sensory

    g_traj, _, _ = scaffold.simulate_run_traj(
        g0, h0, s0, scaffold.weights, scaffold.lambdas, scaffold.beta,
        scaffold.b, scaffold.tau_g, scaffold.tau_h, scaffold.tau_s,
        scaffold.activation, dt=scaffold.dt
    )

    g_err = np.max(np.abs(g_traj[-1] - g0), axis=0)
    frac_grid_sig = np.mean(g_err < 0.3)
    print(frac_grid_sig)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.asarray(g_traj[:, :, patt]).T, aspect='auto', origin='upper', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('time step')
    ax.set_ylabel('grid unit')
    ax.set_title(f'grid trajectory, pattern {patt}')
    fig.colorbar(im, label='activation')
    plt.show()

    #n_correct = sum(np.allclose(final[0][:, p], scaffold.grid[:, p], atol=1e-1) for p in range(3600))
    """plt.imshow(n_correct, aspect='auto')
    plt.colorbar()
    plt.show()"""
    
    #print(final[0][:,0])
    #print(gg[20, 6, :, :])
    

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