import numpy as np
import copy
from nonlinear_model import NonlinearDeterministicNetwork
from decoder import Decoder


class LinearizedModel(NonlinearDeterministicNetwork):
    def __init__(self, network_size=100, nb_readouts=100, nb_inputs=6, exponent_W=0.55,
                 global_mean_input_is_zero=False, rng_seed=1, activation_function='tanh'):
        super().__init__(network_size=network_size, nb_readouts=nb_readouts, nb_inputs=nb_inputs,
                         exponent_W=exponent_W,
                         global_mean_input_is_zero=global_mean_input_is_zero, rng_seed=rng_seed,
                         activation_function=activation_function)

        self.init_conditional_potentials = self.conditioned_potentials()
        self.jac_init = [self.phi_jac(v) for v in self.init_conditional_potentials]
        self.inv_I_minus_W_init = self.inv_I_minus_W()
        self.W0 = copy.copy(self.W)

        self.prefactors_grad = []
        for k in range(self.nb_inputs):
            self.prefactors_grad.append(
                np.linalg.inv(np.eye(self.network_size) - self.W0 @ self.jac_init[k]).T @ self.jac_init[k]
            )

    def init_decoder(self, nb_readouts):
        if nb_readouts > self.network_size:
            raise ValueError("Number of readout units must be smaller than or equal to size of network.")
        ac = np.zeros((self.network_size, self.network_size))

        cma = super().conditioned_activities()
        global_mean = np.mean(cma, axis=0)
        for k in range(self.nb_inputs):
            ac += np.outer(cma[k] - global_mean, cma[k] - global_mean)
        ac /= self.nb_inputs
        readouts_units = np.nonzero(np.diag(ac) > 1e-6)[0][:nb_readouts]
        nb_readouts = len(readouts_units)
        print("Number of readout units", nb_readouts)
        V = 0.5 * self.rng.standard_normal(size=(2, nb_readouts)) / nb_readouts ** 0.5
        #initial_decoder_fac = 0.2
        #V *= (initial_decoder_fac / np.linalg.norm(V)) * (800 / nb_readouts) ** 0.5
        self.decoder = Decoder(readouts_units, self.network_size, V)

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
                error = self.decoder(cas[k]) - self.targets[k]
                grad += self.prefactors_grad[k] @ self.decoder.VR().T @ np.outer(error, self.phi(self.init_conditional_potentials[k]))
            grad /= self.nb_inputs
        else:
            partial_grad = np.zeros_like(self.decoder.VR())
            for k in range(self.nb_inputs):
                partial_grad += np.outer(self.decoder(cas[k]) - self.targets[k], self.init_conditional_potentials[k])
            grad = (self.decoder.VR() @ self.inv_I_minus_W_init).T @ partial_grad / self.nb_inputs
        # ng = np.linalg.norm(grad)
        # threshold = 1.
        # grad = threshold*grad/ng if ng >= threshold else grad  # gradient clipping
        return grad

