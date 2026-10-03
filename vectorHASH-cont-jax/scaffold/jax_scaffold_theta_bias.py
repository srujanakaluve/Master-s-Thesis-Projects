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

def time_averaged_activation(traj, steps_after):
    return jnp.mean(traj[steps_after:], axis=0)

class Scaffold_theta_bias:
    """
    Scaffold with theta waves as bias (subthreshold gating). Frequency, duty and phase are passed as tuples with order (theta_h, theta_g).
    theta_h is in the equation for the hc layer and theta_g is the grid layer equation.

    if you want only one theta term (the other is always ON), set the duty cycle of the other term to 1.0 with any arbitrary frquency or phase. 
    Setting duty of both to 0.0 reduces this to the simple case without theta waves.
    """

    def __init__(self, Nh, lambdas, activation, gg_exc=None, gg_inh=None, gamma=0,
                 tau_g=1., tau_h=1., dt=0.01, freq=(0.1, 0.1), duty=(0.5, 0.5), phase=(0.0, 0.0), gg_default=True, W_gg=None, seed=None):
        
        self.lambdas = tuple(int(l) for l in lambdas)
        self.Ng, self.Nh = sum(l * l for l in self.lambdas), Nh
        self.patts_total = np.prod([l * l for l in self.lambdas])
        if gg_exc is not None and gg_inh is not None:
            self.gg_exc, self.gg_inh = gg_exc, gg_inh
        #self.b = b
        self.gamma = gamma
        self.tau_g, self.tau_h = tau_g, tau_h
        self.dt = dt
        self.freq = freq
        self.duty = duty
        self.phase = phase
        self.rng = np.random.default_rng(seed)

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
        W_hg = self.rng.normal(0, 1, size=(self.Nh, self.Ng))
        if self.gamma != 0:
            prune = int((1 - self.gamma) * self.Nh * self.Ng)
            a = self.rng.integers(0, self.Nh, size=prune)
            b = self.rng.integers(0, self.Ng, size=prune)
            W_hg[a, b] = 0
        grid = self.generate_grid()
        hc = jax.nn.relu(W_hg @ grid)
        W_gh = (1 / self.patts_total) * (grid @ hc.T)
        return grid, W_hg, hc, W_gh

    @staticmethod
    def square_wave(t, freq, duty, phase):
        cycle_position = jnp.mod(freq * t - phase, 1.0)
        return jnp.where(cycle_position < duty, 1.0, 0.0)

    @staticmethod
    @partial(jax.jit, static_argnames=('lambdas', 'activation'))
    def simulate_run(g0, h0, weights, lambdas, tau_g, tau_h, activation,
                     dt=0.01, freq=(0.1, 0.1), duty=(0.5, 0.5), phase=(0.0, 0.0)):
        
        freq_h, freq_g = freq
        duty_h, duty_g = duty
        phase_h, phase_g = phase

        state0 = (g0, h0)

        def rk4_step(state, step):
            g, h = state
            t = step * dt

            def derivative(g_t, h_t, t_t):
                theta_h = Scaffold_theta_bias.square_wave(t_t, freq_h, duty_h, phase_h)
                theta_g = Scaffold_theta_bias.square_wave(t_t, freq_g, duty_g, phase_g)
                dg = (-g_t + activation(weights['W_gg'] @ g_t +  weights['W_gh'] @ h_t - theta_h, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t - theta_g)) / tau_h
                return dg, dh

            dg1, dh1 = derivative(g, h, t)
            dg2, dh2 = derivative(g + 0.5 * dt * dg1, h + 0.5 * dt * dh1, t + 0.5 * dt)
            dg3, dh3 = derivative(g + 0.5 * dt * dg2, h + 0.5 * dt * dh2, t + 0.5 * dt)
            dg4, dh4 = derivative(g + dt * dg3, h + dt * dh3, t + dt)
            g_next = g + (dt / 6.0) * (dg1 + 2 * dg2 + 2 * dg3 + dg4)
            h_next = h + (dt / 6.0) * (dh1 + 2 * dh2 + 2 * dh3 + dh4)
            return (g_next, h_next), None

        final_state, _ = jax.lax.scan(rk4_step, state0, jnp.arange(1000))
        return final_state

    @staticmethod
    @partial(jax.jit, static_argnames=('lambdas', 'activation', 'n_steps'))
    def simulate_run_traj(g0, h0, weights, lambdas, tau_g, tau_h, activation,
                        dt=0.01, freq=(0.1, 0.1), duty=(0.5, 0.5), phase=(0.0, 0.0), n_steps=1000):
        freq_h, freq_g = freq
        duty_h, duty_g = duty
        phase_h, phase_g = phase

        state0 = (g0, h0)

        def rk4_step(state, step):
            g, h = state
            t = step * dt

            def derivative(g_t, h_t, t_t):
                theta_h = Scaffold_theta_bias.square_wave(t_t, freq_h, duty_h, phase_h)
                theta_g = Scaffold_theta_bias.square_wave(t_t, freq_g, duty_g, phase_g)
                dg = (-g_t + activation(weights['W_gg'] @ g_t +  weights['W_gh'] @ h_t - theta_h, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t - theta_g)) / tau_h
                return dg, dh

            dg1, dh1 = derivative(g, h, t)
            dg2, dh2 = derivative(g + 0.5*dt*dg1, h + 0.5*dt*dh1, t + 0.5*dt)
            dg3, dh3 = derivative(g + 0.5*dt*dg2, h + 0.5*dt*dh2, t + 0.5*dt)
            dg4, dh4 = derivative(g + dt*dg3, h + dt*dh3, t + dt)
            g_next = g + (dt/6.0)*(dg1 + 2*dg2 + 2*dg3 + dg4)
            h_next = h + (dt/6.0)*(dh1 + 2*dh2 + 2*dh3 + dh4)

            # carry moves the sim forward; the second element becomes the per-step output ("ys")
            return (g_next, h_next), (g_next, h_next)

        _, (g_traj, h_traj) = jax.lax.scan(
            rk4_step, state0, jnp.arange(n_steps), length=n_steps
        )
        # g_traj: (n_steps, *g0.shape), h_traj: (n_steps, *h0.shape)
        return g_traj, h_traj
    
    @staticmethod
    @partial(jax.jit, static_argnames=('lambdas', 'activation', 'n_steps'))
    def simulate_run_sampled(g0, h0, weights, lambdas, tau_g, tau_h, activation,
                            dt=0.01, freq=(0.1, 0.1), duty=(0.5, 0.5), phase=(0.0, 0.0), n_steps=1000):
        freq_h, freq_g = freq
        duty_h, duty_g = duty
        phase_h, phase_g = phase

        sample_step_g = compute_sample_step(freq_h, duty_h, phase_h, dt, n_steps)
        sample_step_h = compute_sample_step(freq_g, duty_g, phase_g, dt, n_steps)

        state0 = (g0, h0, g0, h0)  # (g, h, g_sampled, h_sampled)

        def rk4_step(state, step):
            g, h, g_sampled, h_sampled = state
            t = step * dt

            def derivative(g_t, h_t, t_t):
                theta_h = Scaffold_theta_bias.square_wave(t_t, freq_h, duty_h, phase_h)
                theta_g = Scaffold_theta_bias.square_wave(t_t, freq_g, duty_g, phase_g)
                dg = (-g_t + activation(weights['W_gg'] @ g_t +  weights['W_gh'] @ h_t - theta_h, lambdas)) / tau_g
                dh = (-h_t + jax.nn.relu(weights['W_hg'] @ g_t - theta_g)) / tau_h
                return dg, dh

            dg1, dh1 = derivative(g, h, t)
            dg2, dh2 = derivative(g + 0.5*dt*dg1, h + 0.5*dt*dh1, t + 0.5*dt)
            dg3, dh3 = derivative(g + 0.5*dt*dg2, h + 0.5*dt*dh2, t + 0.5*dt)
            dg4, dh4 = derivative(g + dt*dg3, h + dt*dh3, t + dt)
            g_next = g + (dt/6.0)*(dg1 + 2*dg2 + 2*dg3 + dg4)
            h_next = h + (dt/6.0)*(dh1 + 2*dh2 + 2*dh3 + dh4)

            hit_g = (step == sample_step_g)
            hit_h = (step == sample_step_h)
            g_sampled = jnp.where(hit_g, g_next, g_sampled)
            h_sampled = jnp.where(hit_h, h_next, h_sampled)

            return (g_next, h_next, g_sampled, h_sampled), None

        # scan still returns the full carry but we only keep what we want
        (_, _, g_sampled, h_sampled), _ = jax.lax.scan(
            rk4_step, state0, jnp.arange(n_steps), length=n_steps
        )
        return g_sampled, h_sampled

    def run(self, g0, h0, freq, duty, phase):
        return self.simulate_run(
            jnp.asarray(g0), jnp.asarray(h0), self.weights, self.lambdas,
            self.tau_g, self.tau_h, self.activation, dt=self.dt,
            freq=freq, duty=duty, phase=phase,
        )

    def run_and_sample(self, g0, h0, freq, duty, phase):
        g_sampled, h_sampled = self.simulate_run_sampled(
            jnp.asarray(g0), jnp.asarray(h0), self.weights, self.lambdas,
            self.tau_g, self.tau_h, self.activation, dt=self.dt,
            freq=freq, duty=duty, phase=phase,
        )
        return g_sampled, h_sampled

    @staticmethod
    def run_and_score(g0, h0, weights, lambdas, tau_g, tau_h, activation, dt,
                  freq, duty, phase, n_steps, g_target, h_target, h_target_norm,
                  grid_thresh, hc_thresh):
        """ this is a static method so that it can be JIT compiled and be used for jax.vmap. """

        g_s, h_s = Scaffold_theta_bias.simulate_run_sampled(
            g0, h0, weights, lambdas, tau_g, tau_h, activation,
            dt=dt, freq=freq, duty=duty, phase=phase, n_steps=n_steps,
        )
        grid_err = jnp.max(jnp.abs(g_s - g_target), axis=0)              # (patts,)
        hc_err = jnp.linalg.norm(h_s - h_target, axis=0) / h_target_norm  # (patts,)
        frac_grid = jnp.mean(grid_err < grid_thresh)
        frac_hc = jnp.mean(hc_err < hc_thresh)
        return frac_grid, frac_hc

    @staticmethod
    def run_and_score_avg(g0, h0, weights, lambdas, tau_g, tau_h, activation, dt,
                       freq, duty, phase, n_steps, steps_after,
                       g_target, h_target, h_target_norm, grid_thresh, hc_thresh):
        g_traj, h_traj = Scaffold_theta_bias.simulate_run_traj(g0, h0, weights, lambdas, tau_g, tau_h,
                                                    activation, dt=dt, freq=freq, duty=duty,
                                                    phase=phase, n_steps=n_steps)
        g_avg = time_averaged_activation(g_traj, steps_after)
        h_avg = time_averaged_activation(h_traj, steps_after)

        grid_err = jnp.max(jnp.abs(g_avg - g_target), axis=0)
        hc_err = jnp.linalg.norm(h_avg - h_target, axis=0) / h_target_norm
        return jnp.mean(grid_err < grid_thresh), jnp.mean(hc_err < hc_thresh)


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    freq, duty, phase = (0.5, 1.0), (0.5, 1.0), (0.5, 0.5)

    """
    t = np.arange(0, 10, 0.05)
    plt.plot(t, Scaffold_theta_bias.square_wave(t, freq[0], duty[0], phase[0]), label='theta_h') #grid
    plt.plot(t, Scaffold_theta_bias.square_wave(t, freq[1], duty[1], phase[1]), label='theta_g')  #hc
    plt.legend()
    plt.show()
    """

    #"""
    lambdas = np.array([3,4,5])
    #gg = np.load('/home/srujana/VSCode Projects/Thesis/vectorHASH-cont-jax/g2g_space.npy')
    scaffold = Scaffold_theta_bias(400, lambdas, 'sigmoid', gg_exc=5, gg_inh=-8.75, gamma=0.6, freq=freq, duty=duty, phase=phase, seed=0)
    h0 = scaffold.hc
    g0 = jnp.zeros_like(scaffold.grid)


    g_traj, _ = Scaffold_theta_bias.simulate_run_traj(g0, h0,
                                                scaffold.weights, scaffold.lambdas, scaffold.tau_g, scaffold.tau_h,
                                                sigmoid, dt=scaffold.dt, freq=scaffold.freq, duty=scaffold.duty, phase=scaffold.phase, n_steps=250,
                                                )
    
    patt = 5
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.asarray(g_traj[:, :, patt]).T, aspect='auto', origin='upper', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('time step')
    ax.set_ylabel('grid unit')
    ax.set_title(f'grid trajectory, pattern {patt}')
    fig.colorbar(im, label='activation')
    plt.show()

    """
    g_err = np.max(np.abs(g_traj[-1, :, :] - scaffold.grid[:, patt]), axis=0)
    frac_grid_sig = np.mean(g_err < 0.2, axis=-1)
    #n_correct = sum(np.allclose(final[0][:, p], scaffold.grid[:, p], atol=1e-1) for p in range(3600))
    #plt.imshow(n_correct, aspect='auto')
    #plt.colorbar()
    #plt.show()
    print(frac_grid_sig)
    #print(g[:,0])
    print(np.max(np.abs(g[:, 0] - scaffold.grid[:, 0])) < 0.2)
    """
    #print(gg[20, 6, :, :])