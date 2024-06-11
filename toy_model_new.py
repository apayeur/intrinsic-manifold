import numpy as np
import matplotlib.pyplot as plt
from utils import target_colors, gram_schmidt, principal_angles, units_convert
from random_vector import MultivariateNormal, GaussianMixture
plt.style.use('rnn4bci_plot_params.dms')
import sklearn.linear_model as lm
from numba import njit
from numba.typed import List as nbList
import copy
import itertools
from math import factorial
from scipy.linalg import subspace_angles


class Network:
    def __init__(self, network_size=50, nb_inputs=8, noise_intensity=1e-3, exponent_W=0.55,
                 global_mean_input_is_zero=False, do_z_score=False, rng_seed=1):
        self.size = (nb_inputs, network_size, 2)
        self.input_size, self.network_size, self.output_size = self.size
        self.nb_inputs = nb_inputs
        self.global_mean_input_is_zero = global_mean_input_is_zero
        self.rng = np.random.default_rng(rng_seed)
        self.do_z_score = do_z_score

        # Private Gaussian noise
        self.noise_intensity = noise_intensity
        self.private_noise = MultivariateNormal(cov=noise_intensity * np.eye(self.network_size), rng=self.rng)

        # BCI decoder attributes
        self.decoder = None  # sklearn LinearModel object
        self.C = None  # projection matrix, shape = (intrinsic_manifold_dim, self.network_size)
        self.D = None  # decoding matrix, shape = (self.output_size, intrinsic_manifold_dim)
        self.intercept = np.zeros(self.output_size)  # intercept of the decoder
        self.inv_Sv = np.eye(self.network_size)  # matrix for z-scoring activity
        self.inv_Sz = None # matrix for z-scoring PCs

        # Targets
        self.targets = [np.array([np.cos(2 * np.pi * i / self.nb_inputs),
                                  np.sin(2 * np.pi * i / self.nb_inputs)]) for i in range(self.nb_inputs)]

        # Inputs
        if self.global_mean_input_is_zero:
            self.inputs = [-np.ones(self.input_size) / self.nb_inputs for i in range(self.nb_inputs)]
        else:
            self.inputs = [np.zeros(self.input_size) for i in range(self.nb_inputs)]
        for i in range(self.nb_inputs):
            self.inputs[i][i] = 1. + self.inputs[i][i]

        self.U, self.W, self.V = self.init_params(exponent_W=exponent_W)

        # Perturbations
        self.selected_permutation_WM = None
        self.selected_permutation_OM = None

    def init_params(self, exponent_W=0.5):
        U = self.rng.uniform(low=-1, high=1, size=(self.network_size, self.input_size))
        W = self.rng.standard_normal(size=(self.network_size, self.network_size)) / self.network_size ** exponent_W
        V = self.rng.standard_normal(size=(2, self.network_size))
        initial_decoder_fac = 0.2
        V *= (initial_decoder_fac / np.linalg.norm(V)) * (800 / self.network_size) ** 0.5
        return U, W, V

    # ==============  Statistics  ==================à
    def inv_I_minus_W(self):
        return np.linalg.inv(np.eye(self.network_size) - self.W)

    def average_input(self):
        return np.mean(self.inputs, axis=0)

    def conditioned_mean_activity(self):
        return [self.inv_I_minus_W() @ self.U @ self.inputs[k] for k in range(self.nb_inputs)]

    def mean_activity(self):
        return np.mean(self.conditioned_mean_activity(), axis=0)

    def conditioned_activity_covariance(self):
        return self.noise_intensity * self.inv_I_minus_W()  @ self.inv_I_minus_W().T

    def activity_covariance(self):
        ac = self.conditioned_activity_covariance()
        cma = self.conditioned_mean_activity()
        for k in range(self.nb_inputs):
            ac += (1/self.nb_inputs) * np.outer(cma[k] - self.mean_activity(), cma[k] - self.mean_activity())
        return ac

    def compute_dVar(self, dW):
        """
        Compute dVar[v], the differential of the total covariance.

        Parameter:
        ---------
        dW : array of shape=(`self.network_size`, `self.network_size`)
            Weight update (= -eta_W * grad_W)
        """
        Wterm = self.inv_I_minus_W() @ dW @ self.activity_covariance()
        return Wterm + Wterm.T

    # ==============  Loss ================
    def task_loss(self):
        cac = self.conditioned_activity_covariance()
        cma = self.conditioned_mean_activity()
        L = 0.
        for k in range(self.nb_inputs):
            L += (np.trace(self.V @ cac @ self.V.T)
                  + np.dot(self.V @ cma[k] - self.targets[k], self.V @ cma[k] - self.targets[k]))
        return L / 2 / self.nb_inputs

    def correlation_component_loss(self):
        ac = self.activity_covariance()
        ma = self.mean_activity()
        return 0.5 * np.trace(self.V @ (ac + np.outer(ma, ma)) @ self.V.T)

    # =========  Training ==========
    def compute_gradient(self):
        cac = self.conditioned_activity_covariance()
        cma = self.conditioned_mean_activity()
        partial_grad = np.zeros_like(self.V)
        for k in range(self.nb_inputs):
            partial_grad += self.V @ cac + np.outer(self.V @ cma[k]-self.targets[k], cma[k])
        return self.inv_I_minus_W().T @ self.V.T @ partial_grad / self.nb_inputs

    def train(self, lr=1.e-3, nb_iter=int(1e3), stopping_crit=None):
        data = {
            'losses': {'task': [], 'corr': []},
            'norm_gradW': [],
            'max_angles': {'dVar_vs_VT': [], 'UpperVar_vs_VT': [], 'LowerVar_vs_VT': [], 'UpperVar_vs_VarBCI': []},
            'min_anlges': {'dVar_vs_VT': [], 'UpperVar_vs_VT': [], 'LowerVar_vs_VT': [], 'UpperVar_vs_VarBCI': []},
            'normalized_variance_explained': [],
            'A': {'D': [], 'DP_WM': []}, 'R': [], 'f': [], 'rel_proj_var_OM': [], 'pr': [], 'max_eigvals': []
        }

        if self.D is not None:
            d = self.D.shape[1]

        var_init = self.activity_covariance()

        if stopping_crit is not None:
            nb_iter = 0
        else:
            stopping_crit = 1e6

        # Learning
        i = 0
        loss = 1e9
        while i < int(nb_iter) or loss > stopping_crit:
            # Compute loss and loss components
            loss = self.task_loss()
            # data['losses']['task'].append(loss)
            # data['losses']['corr'].append(self.correlation_component_loss())

            # data['pr'].append(self.participation_ratio())
            data['max_eigvals'].append(np.max(np.abs(np.linalg.eigvals(self.W))))
            if data['max_eigvals'][-1] >= 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!\n", "EIGENVALUE GREATER THAN 1\n", "!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break

            if nb_iter == 0:
                if i % 500 == 0:
                    print(f"Loss at iteration {i} = {loss}")
            elif nb_iter > 5:
                if i % (nb_iter // 5) == 0 or i == nb_iter - 1:
                    print(f"Loss at iteration {i} = {loss}")

            # Compute gradient
            g = self.compute_gradient()

            if self.C is not None:
                # Compute norm of the gradient
                # data['norm_gradW']['loss'].append(np.linalg.norm(g))

                # Compute angles
                dVar = self.compute_dVar(-lr * g)
                Var = self.activity_covariance()
                U_Var, _, VT_Var = np.linalg.svd(Var)
                upper_var = U_Var[:, :d]
                lower_var = U_Var[:, d:]

                # data['max_angles']['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[0]))
                # data['max_angles']['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[0]))
                # data['max_angles']['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[0]))
                # data['max_angles']['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.C.T)[0]))

                # data['min_angles']['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[-1]))
                # data['min_angles']['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[-1]))
                # data['min_angles']['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[-1]))
                # data['min_angles']['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.C.T)[-1]))

                # Compute manifold overlap (as per Feulner and Clopath)
                beta1 = np.trace(self.C @ var_init @ self.C.T) / np.trace(var_init)
                beta2 = np.trace(self.C @ Var @ self.C.T) / np.trace(
                    Var)  # note that self.C is never reassigned, so it stays at its initial value
                # data['normalized_variance_explained'].append(beta2 / beta1)
                # data['f'].append(beta2)

                if self.selected_permutation_OM is not None:
                    tmp1 = np.trace(self.C[:, self.selected_permutation_OM] @ Var
                                    @ self.C[:, self.selected_permutation_OM].T)
                    tmp2 = np.trace(self.C @ Var @ self.C.T)
                    # data['R'].append(tmp1 / tmp2)
                    # data['rel_proj_var_OM'].append(tmp1 / np.trace(self.C[:, self.selected_permutation_OM] @ var_init @ self.C[:, self.selected_permutation_OM].T))

                if self.selected_permutation_WM is not None:
                    _, _, VDT = np.linalg.svd(self.D)
                    _, _, VDT_WM = np.linalg.svd(self.D[:, self.selected_permutation_WM])
                    # data['A']['D'].append(np.trace(VDT[:2] @ self.C @ Var @ self.C.T @ VDT[:2].T))
                    # data['A']['DP_WM'].append(np.trace(VDT_WM[:2] @ self.C @ Var @ self.C.T @ VDT_WM[:2].T))

            self.W -= lr * g
            i += 1
        return data

    # ============ Methods related to dimensionality ==============
    @staticmethod
    def dimensionality_(covariance_matrix, threshold=0.99):
        _, singular_values, _ = np.linalg.svd(
            covariance_matrix)  # using svg instead of eigvals because we want them properly ordered
        cum_var = np.cumsum(singular_values)
        return np.nonzero(cum_var > threshold * cum_var[-1])[0][0] + 1  # +1 because array elements start at zero

    def dimensionality(self, threshold=0.99):
        return self.dimensionality_(self.activity_covariance(), threshold=threshold)
    @staticmethod
    def participation_ratio_(covariance_matrix):
        return (np.trace(covariance_matrix)) ** 2 / np.trace(covariance_matrix @ covariance_matrix)

    def participation_ratio(self):
        return self.participation_ratio_(self.activity_covariance())
