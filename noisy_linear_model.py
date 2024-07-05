from nonlinear_model import NonlinearDeterministicNetwork
import numpy as np
import matplotlib.pyplot as plt
from utils import units_convert, target_colors
plt.style.use('rnn4bci_plot_params.dms')


class NoisyLinearNetwork(NonlinearDeterministicNetwork):
    def __init__(self, network_size=100, nb_readouts=100, nb_inputs=6, exponent_W=0.55,
                 global_mean_input_is_zero=False, rng_seed=1, noise=0.):
        super().__init__(network_size=network_size, nb_readouts=nb_readouts, nb_inputs=nb_inputs,
                         exponent_W=exponent_W,
                         global_mean_input_is_zero=global_mean_input_is_zero, rng_seed=rng_seed,
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
            error = self.decoder(rates[k]) - self.targets[k]
            L += np.dot(error, error)
        L = 0.5 * L / self.nb_inputs
        return (L + 0.5 * np.trace(self.decoder.VR() @ self.conditioned_covariance() @ self.decoder.VR().T)) if self.noise > 1e-8 else L

    def loss_for_each_target(self):
        losses = np.zeros(self.nb_inputs)
        T = self.decoder.VR() @ self.inv_I_minus_W()
        L_private_noise = 0.5 * self.noise * np.trace(T @ T.T) if self.noise > 1e-8 else 0.
        rates = self.conditioned_activities()
        for k in range(self.nb_inputs):
            error = self.decoder(rates[k]) - self.targets[k]
            losses[k] = 0.5 * np.dot(error, error) + L_private_noise
        return losses / self.nb_inputs

    def correlation_component_loss(self):
        ac = self.activity_covariance()
        ma = self.mean_activity()
        return 0.5 * np.trace(self.decoder.VR() @ (ac + np.outer(ma, ma)) @ self.decoder.VR().T)

    # =========  Training ==========
    def max_eigval(self, potentials):
        return np.max(np.abs(np.linalg.eigvals(self.W)))

    def compute_gradient(self):
        v = self.conditioned_potentials()
        partial_grad = np.zeros_like(self.decoder.VR())
        for k in range(self.nb_inputs):
            partial_grad += np.outer(self.decoder(v[k]) - self.targets[k], v[k])

        partial_grad_private_var = self.decoder.VR() @ self.conditioned_covariance() if self.noise > 1e-8 else np.zeros_like(
            self.decoder.VR())
        grad = (self.decoder.VR() @ self.inv_I_minus_W()).T @ (partial_grad / self.nb_inputs + partial_grad_private_var)
        # ng = np.linalg.norm(grad)
        # threshold = 1.
        # grad = threshold*grad/ng if ng >= threshold else grad  # gradient clipping
        return grad

    # ========= Sampling model =========
    def sample(self, nb_epochs=int(100)):
        network_activity = np.zeros((self.nb_inputs * nb_epochs, self.network_size))
        outputs = np.zeros((self.nb_inputs * nb_epochs, self.output_size))
        potentials = self.conditioned_activities()

        i = 0
        for _ in range(nb_epochs):
            for k in range(self.nb_inputs):
                v = potentials[k] + self.noise**0.5 * self.inv_I_minus_W() @ self.rng.standard_normal(self.network_size)
                network_activity[i] = v
                outputs[i] = self.decoder(v)
                i += 1
        return network_activity, outputs

    #  ========  Plotting functions  =========
    def plot_output(self, outfile_name=None, n_samples=int(100)):
        plt.figure(figsize=(45 * units_convert['mm'], 45 * units_convert['mm'] / 1.25))
        potentials = self.conditioned_activities()
        for k in range(self.nb_inputs):
            for _ in range(n_samples):
                v = potentials[k] + self.noise**0.5 * self.inv_I_minus_W() @ self.rng.standard_normal(self.network_size)
                u = self.decoder(v)
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
