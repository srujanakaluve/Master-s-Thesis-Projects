import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax_helpers import sigmoid, glob_inh, softmax

def compute_sample_step(freq, duty, phase, dt, n_steps, safety_steps=1):
    t_final = (n_steps - 1) * dt

    def freq_zero_case(_):
        return jnp.array(n_steps - 1, dtype=jnp.int32)

    def freq_pos_case(_):
        period = 1.0 / freq
        on_duration = duty * period
        m_max = jnp.floor((t_final - phase - on_duration) / period)
        on_end_time = phase + m_max * period + on_duration
        sample_time = jnp.clip(on_end_time - safety_steps * dt, 0.0, t_final)
        return jnp.round(sample_time / dt).astype(jnp.int32)

    return jax.lax.cond(freq <= 0, freq_zero_case, freq_pos_case, operand=None)

class Scaffold:

    def __init__(self, Nh, lambdas, activation, gg_exc=None, gg_inh=None, gamma=0, b=0.5,
                 tau_g=1., tau_h=1., dt=0.1, freq=0, duty=0.5, phase=0, gg_default=True, W_gg=None):
        
        self.lambdas = tuple(int(l) for l in lambdas)
        self.Ng, self.Nh = sum(l * l for l in self.lambdas), Nh
        self.patts_total = np.prod([l * l for l in self.lambdas])
        if gg_exc is not None and gg_inh is not None:
            self.gg_exc, self.gg_inh = gg_exc, gg_inh
        self.b = b
        self.gamma = gamma
        self.tau_g, self.tau_h = tau_g, tau_h
        self.dt = dt
        self.freq, self.duty, self.phase = freq, duty, phase

        if gg_default:
            self.W_gg = self.gridtogrid()
        elif gg_default is False and W_gg is not None:
            self.W_gg = W_gg

        if activation == 'softmax':
            self.activation = softmax
        if activation == 'glob_inh':
            self.activation = glob_inh
        if activation == 'sigmoid':
            self.activation = sigmoid
        self.grid, self.W_hg, self.hc, self.W_gh = self.scaffold_layers()

        self.weights = {
            'W_gg': jnp.array(self.W_gg),
            'W_gh': jnp.array(self.W_gh),
            'W_hg': jnp.array(self.W_hg),
        }

    def gridtogrid(self):
        W_gg = np.zeros((self.Ng, self.Ng))
        i = 0
        for lam in self.lambdas:
            size = lam ** 2
            W_gg[i:i + size, i:i + size] = self.gg_inh
            i += size
        np.fill_diagonal(W_gg, self.gg_exc)
        return W_gg

    def generate_grid(self):
        lambda_sq = np.array([l * l for l in self.lambdas])
        grid = np.zeros((self.Ng, self.patts_total))
        jumps = [0] + list(np.cumsum(lambda_sq))[:-1]
        for i in range(self.patts_total):
            a = np.mod(i, lambda_sq)
            grid[a + jumps, i] = 1
        return grid

    def scaffold_layers(self):
        W_hg = np.random.normal(0, 1, size=(self.Nh, self.Ng))
        if self.gamma != 0:
            prune = int((1 - self.gamma) * self.Nh * self.Ng)
            a, b = np.random.randint(0, self.Nh, size=prune), np.random.randint(0, self.Ng, size=prune)
            W_hg[a, b] = 0
        grid = self.generate_grid()
        hc = jax.nn.relu(W_hg @ grid - self.b)
        W_gh = (1 / self.patts_total) * (grid @ hc.T)
        return grid, W_hg, hc, W_gh

    @staticmethod
    def square_wave(t, freq, duty, phase):
        """Return a SciPy-compatible +/-1 square wave using JAX operations."""
        cycle_position = jnp.mod(freq * (t - phase), 1.0)
        return jnp.where(cycle_position < duty, 1.0, -1.0)

    @staticmethod
    @partial(jax.jit, static_argnames=('lambdas', 'activation'))
    def simulate_run(g0, h0, weights, lambdas, b, tau_g, tau_h, activation,
                     dt=0.1, freq=0.0, duty=0.5, phase=0.0):
        state0 = (g0, h0)

        def rk4_step(state, step):
            g, h = state
            t = step * dt

            def derivative(g_t, h_t, t_t):
                theta_h = Scaffold.square_wave(t_t, freq, duty, phase)
                dg = (-g_t + activation(
                    weights['W_gg'] @ g_t + theta_h * weights['W_gh'] @ h_t,
                    lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t - b)) / tau_h
                return dg, dh

            dg1, dh1 = derivative(g, h, t)
            dg2, dh2 = derivative(g + 0.5 * dt * dg1, h + 0.5 * dt * dh1, t + 0.5 * dt)
            dg3, dh3 = derivative(g + 0.5 * dt * dg2, h + 0.5 * dt * dh2, t + 0.5 * dt)
            dg4, dh4 = derivative(g + dt * dg3, h + dt * dh3, t + dt)
            g_next = g + (dt / 6.0) * (dg1 + 2 * dg2 + 2 * dg3 + dg4)
            h_next = h + (dt / 6.0) * (dh1 + 2 * dh2 + 2 * dh3 + dh4)
            return (g_next, h_next), None

        final_state, _ = jax.lax.scan(rk4_step, state0, jnp.arange(250))
        return final_state

    @staticmethod
    @partial(jax.jit, static_argnames=('lambdas', 'activation', 'n_steps'))
    def simulate_run_sampled(g0, h0, weights, lambdas, b, tau_g, tau_h, activation,
                            dt=0.1, freq=0.0, duty=0.5, phase=0.0, n_steps=250):
        sample_step = compute_sample_step(freq, duty, phase, dt, n_steps)

        state0 = (g0, h0, g0, h0)  # (g, h, g_sampled, h_sampled)

        def rk4_step(state, step):
            g, h, g_sampled, h_sampled = state
            t = step * dt

            def derivative(g_t, h_t, t_t):
                theta_h = Scaffold.square_wave(t_t, freq, duty, phase)
                dg = (-g_t + activation(weights['W_gg'] @ g_t + theta_h * weights['W_gh'] @ h_t, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t - b)) / tau_h
                return dg, dh

            dg1, dh1 = derivative(g, h, t)
            dg2, dh2 = derivative(g + 0.5*dt*dg1, h + 0.5*dt*dh1, t + 0.5*dt)
            dg3, dh3 = derivative(g + 0.5*dt*dg2, h + 0.5*dt*dh2, t + 0.5*dt)
            dg4, dh4 = derivative(g + dt*dg3, h + dt*dh3, t + dt)
            g_next = g + (dt/6.0)*(dg1 + 2*dg2 + 2*dg3 + dg4)
            h_next = h + (dt/6.0)*(dh1 + 2*dh2 + 2*dh3 + dh4)

            hit = (step == sample_step)
            g_sampled = jnp.where(hit, g_next, g_sampled)
            h_sampled = jnp.where(hit, h_next, h_sampled)

            return (g_next, h_next, g_sampled, h_sampled), None

        # scan still returns the full carry internally — we just only keep what we want
        (_, _, g_sampled, h_sampled), _ = jax.lax.scan(
            rk4_step, state0, jnp.arange(n_steps), length=n_steps
        )
        return g_sampled, h_sampled

    def run(self, g0, h0, freq=0.0, duty=0.5, phase=0.0):
        return self.simulate_run(
            jnp.asarray(g0), jnp.asarray(h0), self.weights, self.lambdas,
            self.b, self.tau_g, self.tau_h, self.activation, dt=self.dt,
            freq=freq, duty=duty, phase=phase,
        )

    def run_and_sample(self, g0, h0, freq=0.0, duty=0.5, phase=0.0):
        g_sampled, h_sampled = self.simulate_run_sampled(
            jnp.asarray(g0), jnp.asarray(h0), self.weights, self.lambdas,
            self.b, self.tau_g, self.tau_h, self.activation, dt=self.dt,
            freq=freq, duty=duty, phase=phase,
        )
        return g_sampled, h_sampled

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt


    lambdas = np.array([3,4,5])
    gg = np.load('/home/srujana/VSCode Projects/Thesis/vectorHASH-cont-jax/g2g_space.npy')
    scaffold = Scaffold(400, lambdas, 'sigmoid', gg_exc=1, gg_inh=-5, gamma=0.6, freq=0.2, duty=0.4)
    h0 = scaffold.hc
    g0 = jnp.zeros_like(scaffold.grid)
    g, h = scaffold.run_and_sample(g0, h0, scaffold.freq, scaffold.duty)

    g_err = np.max(np.abs(g - scaffold.grid), axis=0)
    frac_grid_sig = np.mean(g_err < 0.2, axis=-1)
    #n_correct = sum(np.allclose(final[0][:, p], scaffold.grid[:, p], atol=1e-1) for p in range(3600))
    #plt.imshow(n_correct, aspect='auto')
    #plt.colorbar()
    #plt.show()
    print(frac_grid_sig)
    print(g[:,0])
    print(np.max(np.abs(g[:, 0] - scaffold.grid[:, 0])) < 0.2)
    #print(gg[20, 6, :, :])
    
