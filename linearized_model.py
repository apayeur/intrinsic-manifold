import numpy as np
from toy_model_new import NonlinearDeterministicNetwork
import copy


class LinearizedModel(NonlinearDeterministicNetwork):
    def __init__(self, network_size=100, nb_inputs=6, exponent_W=0.55, exponent_V=1,
                 global_mean_input_is_zero=False, do_z_score=False, rng_seed=1, activation_function='tanh'):
        super().__init__(network_size=network_size, nb_inputs=nb_inputs,
                         exponent_W=exponent_W, exponent_V=exponent_V,
                         global_mean_input_is_zero=global_mean_input_is_zero,
                         do_z_score=do_z_score, rng_seed=rng_seed,
                         activation_function=activation_function)

        self.init_conditional_potentials = self.conditioned_potentials()
        self.W0 = copy.copy(self.W)
        self.jac_init = [self.phi_jac(v) for v in self.init_conditional_potentials]
        self.inv_I_minus_W_init = self.inv_I_minus_W()
        self.prefactors_grad = []
        for k in range(self.nb_inputs):
            self.prefactors_grad.append(
                np.linalg.inv(np.eye(self.network_size) - self.W0 @ self.jac_init[k]).T @ self.jac_init[k]
            )

    def conditioned_activities(self):
        cps = []
        for k in range(self.nb_inputs):
            cps.append(
                self.phi(self.init_conditional_potentials[k]) + self.jac_init[k] @ self.inv_I_minus_W_init @ (self.W - self.W0) @ self.phi(self.init_conditional_potentials[k])
                       )
        return cps

    def compute_gradient(self):
        cas = self.conditioned_activities()
        if self.activation_function != 'linear':
            grad = np.zeros_like(self.W)
            for k in range(self.nb_inputs):
                error = self.V @ (cas[k] - self.ma_0) + self.intercept - self.targets[k]
                grad += self.prefactors_grad[k] @ self.V.T @ np.outer(error, self.phi(self.init_conditional_potentials[k]))
            grad /= self.nb_inputs
        else:
            partial_grad = np.zeros_like(self.V)
            for k in range(self.nb_inputs):
                partial_grad += np.outer(self.V @ (cas[k] - self.ma_0) + self.intercept - self.targets[k], self.init_conditional_potentials[k])
            grad = (self.V @ self.inv_I_minus_W_init).T @ partial_grad / self.nb_inputs
        # ng = np.linalg.norm(grad)
        # threshold = 1.
        # grad = threshold*grad/ng if ng >= threshold else grad  # gradient clipping
        return grad

