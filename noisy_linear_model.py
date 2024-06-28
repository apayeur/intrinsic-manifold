from nonlinear_model import NonlinearDeterministicNetwork
import numpy as np
import matplotlib.pyplot as plt
from utils import units_convert, target_colors
plt.style.use('rnn4bci_plot_params.dms')


class NoisyLinearNetwork(NonlinearDeterministicNetwork):
    def __init__(self, network_size=100, nb_inputs=6, exponent_W=0.55, noise=0.,
                 global_mean_input_is_zero=False, do_z_score=False, rng_seed=1):
        super().__init__(network_size=network_size, nb_inputs=nb_inputs,
                         exponent_W=exponent_W,
                         global_mean_input_is_zero=global_mean_input_is_zero,
                         do_z_score=do_z_score, rng_seed=rng_seed,
                         activation_function='linear')
        self.noise = noise

    # ==============  Statistics  =================
    def conditioned_activities(self):
        return self.conditioned_potentials()

    def activity_covariance(self):
        ac = np.zeros((self.network_size, self.network_size))
        cma = self.conditioned_activities()
        for k in range(self.nb_inputs):
            ac += np.outer(cma[k] - self.mean_activity(), cma[k] - self.mean_activity())
        private_cov = self.conditioned_covariance() if self.noise > 1e-8 else 0
        return ac / self.nb_inputs + private_cov

    def conditioned_covariance(self):
        return self.noise * self.inv_I_minus_W() @ self.inv_I_minus_W().T

    # ==============  Loss ================
    def task_loss(self):
        rates = self.conditioned_activities()
        L = 0.
        for k in range(self.nb_inputs):
            error = self.V @ (rates[k] - self.ma_0) + self.intercept - self.targets[k]
            L += np.dot(error, error)
        L = 0.5 * L / self.nb_inputs
        return (L + 0.5 * np.trace(self.V @ self.conditioned_covariance() @ self.V.T)) if self.noise > 1e-8 else L

    def loss_for_each_target(self):
        losses = np.zeros(self.nb_inputs)
        T = self.V @ self.inv_I_minus_W()
        L_private_noise = 0.5 * self.noise * np.trace(T @ T.T) if self.noise > 1e-8 else 0.
        rates = self.conditioned_activities()
        for k in range(self.nb_inputs):
            error = self.V @ (rates[k] - self.ma_0) + self.intercept - self.targets[k]
            losses[k] = 0.5 * np.dot(error, error) + L_private_noise
        return losses / self.nb_inputs

    def correlation_component_loss(self):
        ac = self.activity_covariance()
        ma = self.mean_activity()
        return 0.5 * np.trace(self.V @ (ac + np.outer(ma, ma)) @ self.V.T)

    # =========  Training ==========
    def max_eigval(self, potentials):
        return np.max(np.abs(np.linalg.eigvals(self.W)))

    def compute_gradient(self):
        v = self.conditioned_potentials()
        partial_grad = np.zeros_like(self.V)
        for k in range(self.nb_inputs):
            partial_grad += np.outer(self.V @ (v[k] - self.ma_0) + self.intercept - self.targets[k], v[k])

        partial_grad_private_var = self.V @ self.conditioned_covariance() if self.noise > 1e-8 else np.zeros_like(
            self.V)
        grad = (self.V @ self.inv_I_minus_W()).T @ (partial_grad / self.nb_inputs + partial_grad_private_var)
        # ng = np.linalg.norm(grad)
        # threshold = 1.
        # grad = threshold*grad/ng if ng >= threshold else grad  # gradient clipping
        return grad

    #  ========  Plotting functions  =========
    def plot_output(self, outfile_name=None, n_samples=int(100)):
        plt.figure(figsize=(45 * units_convert['mm'], 45 * units_convert['mm'] / 1.25))
        potentials = self.conditioned_activities()
        for k in range(self.nb_inputs):
            for sample in range(n_samples):
                v = potentials[k] + self.noise**0.5 * self.inv_I_minus_W() @ self.rng.standard_normal(size=self.network_size)
                u = self.V @ (v - self.ma_0) + self.intercept
                plt.scatter(u[0], u[1], s=8,
                            facecolor=target_colors[k], edgecolors='white', lw=0.2)
        for k in range(self.nb_inputs):
            plt.scatter(self.targets[k][0], self.targets[k][1], s=13,
                        facecolor=target_colors[k], edgecolors='black', lw=0.4, zorder=10)
        plt.xticks([-2, 2])
        plt.yticks([-2, 2])
        plt.xlabel('$u_x$')
        plt.ylabel('$u_y$')
        plt.gca().set_axis_off()
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        if outfile_name is None:
            plt.show()
        else:
            plt.savefig(outfile_name, transparent=True)
            plt.close()
