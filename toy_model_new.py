import numpy as np
import matplotlib.pyplot as plt
from utils import target_colors, gram_schmidt, principal_angles, units_convert
from random_vector import MultivariateNormal, GaussianMixture
plt.style.use('rnn4bci_plot_params.dms')
import sklearn.linear_model as lm
from numba import njit
from utils import target_colors, gram_schmidt, principal_angles, units_convert
from numba.typed import List as nbList
import activation_functions
import copy
import itertools
from math import factorial
from scipy.linalg import subspace_angles
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression


class NonlinearDeterministicNetwork:
    def __init__(self, network_size=100, nb_inputs=6, exponent_W=0.55, exponent_V=1,
                 global_mean_input_is_zero=False, do_z_score=False, rng_seed=1, activation_function='tanh'):
        self.size = (nb_inputs, network_size, 2)
        self.input_size, self.network_size, self.output_size = self.size
        self.nb_inputs = nb_inputs
        self.global_mean_input_is_zero = global_mean_input_is_zero
        self.rng = np.random.default_rng(rng_seed)
        self.do_z_score = do_z_score
        self.activation_function = activation_function
        if activation_function == 'tanh':
            self.phi = np.tanh
            self.phi_prime = activation_functions.tanh_prime
            self.phi_jac = activation_functions.tanh_jac
        elif activation_function == 'relu':
            self.phi = activation_functions.relu
            self.phi_prime = activation_functions.relu_prime
            self.phi_jac = activation_functions.relu_jac
        elif activation_function == 'linear':
            self.phi = activation_functions.linear
            self.phi_prime = activation_functions.linear_prime
            self.phi_jac = activation_functions.linear_jac
        else:
            raise ValueError("`activation_function` must be 'tanh', 'relu' or 'linear'")

        # BCI decoder attributes
        self.decoder = None  # sklearn LinearModel object
        self.C = None  # projection matrix, shape = (intrinsic_manifold_dim, self.network_size)
        self.D = None  # decoding matrix, shape = (self.output_size, intrinsic_manifold_dim)
        self.intercept = np.zeros(self.output_size)  # intercept of the decoder
        self.inv_Sv = np.eye(self.network_size)  # matrix for z-scoring activity
        self.inv_Sz = None # matrix for z-scoring PCs
        self.ma_0 = np.zeros(self.network_size)

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

        self.U, self.W, self.V, self.b = self.init_params(exponent_W=exponent_W, exponent_V=exponent_V)

        self.prev_potentials = [self.inv_I_minus_W() @ (self.U @ self.inputs[k] + self.b) for k in range(self.nb_inputs)] # as initial condition for solver

        # Perturbations
        self.selected_permutation_WM = None
        self.selected_permutation_OM = None

    def init_params(self, exponent_W, exponent_V):
        U = self.rng.uniform(low=-1, high=1, size=(self.network_size, self.input_size))
        W = self.rng.standard_normal(size=(self.network_size, self.network_size)) / self.network_size ** exponent_W
        V = self.rng.standard_normal(size=(2, self.network_size)) # / self.network_size ** exponent_V
        b = np.zeros(self.network_size)  # self.rng.uniform(low=0, high=1, size=(self.network_size, ))
        initial_decoder_fac = 0.2
        V *= (initial_decoder_fac / np.linalg.norm(V)) * (800 / self.network_size) ** 0.5
        return U, W, V, b

    # ============= For activity solver =============
    @staticmethod
    def F(v, W, c, a_fun):
        if a_fun == 'tanh':
            return v - W @ np.tanh(v) + c
        elif a_fun == 'relu':
            return v - W @ activation_functions.relu(v) + c

    @staticmethod
    def dF(v, W, c, a_fun):
        if a_fun == 'tanh':
            return np.eye(W.shape[0]) - W @ activation_functions.tanh_jac(v)
        elif a_fun == 'relu':
            return np.eye(W.shape[0]) - W @ activation_functions.relu_jac(v)

    # ==============  Statistics  ==================
    def inv_I_minus_W(self):
        return np.linalg.inv(np.eye(self.network_size) - self.W)

    def average_input(self):
        return np.mean(self.inputs, axis=0)

    def conditioned_potentials(self):
        """Solve F(v) = v - Wf(v) + Ux"""
        cps = []
        for k in range(self.nb_inputs):
            initial_value = self.prev_potentials[k]
            if self.activation_function != 'linear':
                sol, _, ier, _ = fsolve(self.F, initial_value,
                                        args=(self.W, self.U@self.inputs[k]+self.b, self.activation_function), fprime=self.dF,
                                        full_output=True)
                if ier:
                    cps.append(sol)
                    self.prev_potentials[k] = sol

                else:
                    raise Exception("Root not found")
            else:
                cps.append(self.inv_I_minus_W()@(self.U@self.inputs[k]+self.b))
        return cps

    def conditioned_activities(self):
        v = self.conditioned_potentials()
        return [self.phi(v[k]) for k in range(self.nb_inputs)]

    def mean_activity(self):
        return np.mean(self.conditioned_activities(), axis=0)

    def mean_potential(self):
        return np.mean(self.conditioned_potentials(), axis=0)

    def activity_covariance(self):
        ac = np.zeros((self.network_size, self.network_size))
        cma = self.conditioned_activities()
        for k in range(self.nb_inputs):
            ac += np.outer(cma[k] - self.mean_activity(), cma[k] - self.mean_activity())
        return ac / self.nb_inputs

    def activity_correlation(self):
        return self.activity_covariance() + np.outer(self.mean_activity(), self.mean_activity())


    # ==============  Loss ================
    def task_loss(self):
        rates = self.conditioned_activities()
        L = 0.
        for k in range(self.nb_inputs):
            error = self.V @ rates[k] - self.targets[k]
            L += np.dot(error, error)
        return 0.5 * L / self.nb_inputs

    def loss_for_each_target(self):
        losses = np.zeros(self.nb_inputs)
        rates = self.conditioned_activities()
        for k in range(self.nb_inputs):
            error = self.V @ rates[k] - self.targets[k]
            losses[k] = 0.5 * np.dot(error, error)
        return losses

    def correlation_component_loss(self):
        """TODO: Check if formula still valid with nonlinearity."""
        ac = self.activity_covariance()
        ma = self.mean_activity()
        return 0.5 * np.trace(self.V @ (ac + np.outer(ma, ma)) @ self.V.T)

    # =========  Training ==========
    def compute_gradient(self):
        potentials = self.conditioned_potentials()

        grad = np.zeros_like(self.W)
        for k in range(self.nb_inputs):
            J = self.phi_jac(potentials[k])
            error = self.V @ self.phi(potentials[k]) - self.targets[k]
            grad += np.linalg.inv(np.eye(self.network_size) - J@self.W.T) @ J @ self.V.T @ np.outer(error, self.phi(potentials[k]))
        grad /= self.nb_inputs
        ng = np.linalg.norm(grad)
        threshold = 1.
        #grad = threshold*grad/ng if ng >= threshold else grad  # gradient clipping
        return grad

    def train(self, lr=1.e-2, nb_iter=int(1e3), stopping_crit=None, do_record_data=True):
        if do_record_data:
            data = {
                'losses': {'task': [], 'corr': []},
                'norm_gradW': [],
                'max_angles': {'dVar_vs_VT': [], 'UpperVar_vs_VT': [], 'LowerVar_vs_VT': [], 'UpperVar_vs_VarBCI': []},
                'min_angles': {'dVar_vs_VT': [], 'UpperVar_vs_VT': [], 'LowerVar_vs_VT': [], 'UpperVar_vs_VarBCI': []},
                'normalized_variance_explained': [],
                'A': {'D': [], 'DP_WM': []}, 'R': [], 'f': [], 'rel_proj_var_OM': [], 'pr': [], 'max_eigvals': [],
                'tot_var': []
            }
        else:
            data = None
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
        grad_norm = 0.
        while i < int(nb_iter) or loss > stopping_crit:
            var_prev = self.activity_covariance()
            if do_record_data:
                data['tot_var'] = np.trace(var_prev)
            # Compute loss and loss components
            loss = self.task_loss()
            if do_record_data:
                data['losses']['task'].append(loss)
                data['losses']['corr'].append(self.correlation_component_loss())

                data['pr'].append(self.participation_ratio())
                potentials = self.conditioned_potentials()
                max_eigvals = [np.max(np.abs(np.linalg.eigvals(self.W@self.phi_jac(potentials[k])))) for k in range(self.nb_inputs)]
                #data['max_eigvals'].append(np.max(max_eigvals))
                if np.max(max_eigvals) >= 1:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!\n", "EIGENVALUE GREATER THAN 1\n", "!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    data['losses']['task'][-1] = -1

            if nb_iter == 0:
                if i % 500 == 0:
                    print(f"Loss at iteration {i} = {loss}")
            elif nb_iter > 5:
                if i % (nb_iter // 5) == 0 or i == nb_iter - 1:
                    print(f"Loss at iteration {i} = {loss}")

            # Compute gradient
            g = self.compute_gradient()
            grad_norm = np.linalg.norm(g)

            if self.C is not None:
                if do_record_data:
                    # Compute norm of the gradient
                    data['norm_gradW'].append(np.linalg.norm(g))

                    # Compute angles
                    Var = self.activity_covariance()
                    # dVar = Var - var_prev
                    # U_Var, _, VT_Var = np.linalg.svd(Var)
                    # upper_var = U_Var[:, :d]
                    # lower_var = U_Var[:, d:]

                    # data['max_angles']['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[0]))
                    # data['max_angles']['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[0]))
                    # data['max_angles']['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[0]))
                    # data['max_angles']['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.C.T)[0]))
                    #
                    # data['min_angles']['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[-1]))
                    # data['min_angles']['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[-1]))
                    # data['min_angles']['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[-1]))
                    # data['min_angles']['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.C.T)[-1]))

                    # Compute manifold overlap (as per Feulner and Clopath)
                    beta1 = np.trace(self.C @ var_init @ self.C.T) / np.trace(var_init)
                    beta2 = np.trace(self.C @ Var @ self.C.T) / np.trace(
                        Var)  # note that self.C is never reassigned, so it stays at its initial value
                    data['normalized_variance_explained'].append(beta2 / beta1)
                    data['f'].append(beta2)

                    if self.selected_permutation_OM is not None:
                        tmp1 = np.trace(self.C[:, self.selected_permutation_OM] @ Var
                                        @ self.C[:, self.selected_permutation_OM].T)
                        tmp2 = np.trace(self.C @ Var @ self.C.T)
                        data['R'].append(tmp1 / tmp2)
                        data['rel_proj_var_OM'].append(tmp1 / np.trace(self.C[:, self.selected_permutation_OM] @ var_init @ self.C[:, self.selected_permutation_OM].T))

                    if self.selected_permutation_WM is not None:
                        _, _, VDT = np.linalg.svd(self.D)
                        _, _, VDT_WM = np.linalg.svd(self.D[:, self.selected_permutation_WM])
                        data['A']['D'].append(np.trace(VDT[:2] @ self.C @ Var @ self.C.T @ VDT[:2].T))
                        data['A']['DP_WM'].append(np.trace(VDT_WM[:2] @ self.C @ Var @ self.C.T @ VDT_WM[:2].T))

            self.W -= lr * g
            i += 1
        return data

    # ============ Methods related to decoder ==============
    def fit_decoder(self, intrinsic_manifold_dim=None, threshold=0.95, fit_intercept=False):
        tot_var = self.activity_correlation() if self.do_z_score else self.activity_covariance()
        _, s, vt = np.linalg.svd(tot_var)
        evs = np.linalg.eigvals(tot_var)

        dim = self.dimensionality(threshold=threshold)
        print('Number of PCs for {} of total variance = {}'.format(threshold, dim))
        if intrinsic_manifold_dim is None:
            intrinsic_manifold_dim = dim

        self.C = vt[:intrinsic_manifold_dim, :]  # projection matrix
        self.inv_Sz = np.diag(np.sqrt(np.diag(self.C @ tot_var @ self.C.T)) ** -1) if self.do_z_score else np.eye(
            intrinsic_manifold_dim)
        C_loc = self.inv_Sz @ self.C @ self.inv_Sv
        self.ma_0 = self.mean_activity() if self.do_z_score else np.zeros(self.network_size)

        if fit_intercept or self.do_z_score:
            lr = LinearRegression()
            ca = np.asarray(self.conditioned_activities())
            lr.fit((ca - self.ma_0) @ C_loc.T, ca @ self.V.T)
            self.intercept = lr.intercept_
            self.D = lr.coef_
            print("Fit R2:", lr.score((ca - self.ma_0) @ C_loc.T, ca @ self.V.T))
            print("D =", self.D)
            target_shift = self.D @ C_loc @ self.ma_0 - self.intercept
            for i in range(self.nb_inputs):
                self.targets[i] += target_shift
        else:
            Var = self.activity_covariance()
            vbarvbarT = np.outer(self.mean_activity(), self.mean_activity())
            self.D = self.V @ (Var + vbarvbarT) @ C_loc.T @ np.linalg.inv(C_loc @ (Var + vbarvbarT) @ C_loc.T)
        self.V = self.D @ C_loc
        return intrinsic_manifold_dim, dim

    def select_perturb(self, intrinsic_manifold_dim, nb_om_permuted_units=30, nb_samples=int(1e3)):
        """Select the WM and OM perturbations"""
        nb_samples_wm = factorial(intrinsic_manifold_dim) if intrinsic_manifold_dim <= 8 else nb_samples
        nb_samples_om = max(nb_samples, nb_samples_wm)

        wm_permutations = np.empty(shape=(nb_samples_wm, intrinsic_manifold_dim))
        om_permutations = np.empty(shape=(nb_samples_om, self.network_size))
        wm_losses = np.empty(shape=(nb_samples_wm, self.nb_inputs))
        om_losses = np.empty(shape=(nb_samples_om, self.nb_inputs))
        wm_total_losses = []
        om_total_losses = []

        # WM
        if intrinsic_manifold_dim > 7:
            for perm_counter in range(nb_samples_wm):
                indices = np.arange(intrinsic_manifold_dim)
                self.rng.shuffle(indices)
                perm = indices
                self.V = self.D[:, perm] @ self.inv_Sz @ self.C @ self.inv_Sv
                wm_losses[perm_counter] = self.loss_for_each_target()
                wm_permutations[perm_counter] = perm
                wm_total_losses.append(self.task_loss())
        else:  # comb over all possible permutations
            for perm_counter, perm in enumerate(itertools.permutations(range(intrinsic_manifold_dim))):
                self.V = self.D[:, perm] @ self.inv_Sz @ self.C @ self.inv_Sv
                wm_losses[perm_counter] = self.loss_for_each_target()
                wm_permutations[perm_counter] = perm
                wm_total_losses.append(self.task_loss())
        print(f"Median total loss for WM perturbation : {np.median(wm_total_losses)}")
        print(f"Median target-wise loss for WM perturbation : {np.median(wm_losses, axis=0)}")

        # OM
        self.V = self.D @ self.inv_Sz @ self.C @ self.inv_Sv
        mds = self.get_modulation_depth()
        sorted_indices = np.argsort(mds)
        indices_to_permute = sorted_indices[-nb_om_permuted_units:]

        for perm_counter in range(nb_samples_om):
            indices = copy.deepcopy(indices_to_permute)
            self.rng.shuffle(indices)
            indices_i = np.arange(self.network_size)
            indices_i[indices_to_permute] = indices
            self.V = self.D @ self.inv_Sz @ self.C[:, indices_i] @ self.inv_Sv
            om_losses[perm_counter] = self.loss_for_each_target()
            om_total_losses.append(self.task_loss())
            om_permutations[perm_counter] = indices_i

        # Return to original mapping
        self.V = self.D @ self.inv_Sz @ self.C @ self.inv_Sv

        # Compute median target-specific losses across all WM and OM permutations
        median_per_target_loss = np.median(np.vstack((wm_losses, om_losses)), axis=0, keepdims=True)
        print(f'Combined median per-target loss = {median_per_target_loss}')

        # Find WM and OM permutations closest to median WM perturbations
        normed_diff = np.linalg.norm(wm_losses - median_per_target_loss, axis=1)
        selected_wm = wm_permutations[np.argmin(normed_diff)]
        self.selected_permutation_WM = np.asarray(selected_wm, dtype=int)

        normed_diff = np.linalg.norm(om_losses - median_per_target_loss, axis=1)
        selected_om = om_permutations[np.argmin(normed_diff)]
        self.selected_permutation_OM = np.asarray(selected_om, dtype=int)
        return self.selected_permutation_WM, self.selected_permutation_OM, wm_total_losses, om_total_losses

    def apply_wm_perturb(self, selected_wm):
        self.V = self.D[:, selected_wm] @ self.inv_Sz @ self.C @ self.inv_Sv

    def apply_om_perturb(self, selected_om):
        self.V = self.D @ self.inv_Sz @ self.C[:, selected_om] @ self.inv_Sv

    def get_modulation_depth(self):
        """
        Compute max - min expected activity across target to get an approximate modulation depth
        (one would need to compute tuning curve for a more appropriate value).

        Return:
        ------
        modulation_depth : 1D array of shape (`self.network_size`, )
        """
        ca = np.asarray(self.conditioned_activities())
        lr = LinearRegression()
        mds = []
        for i in range(self.network_size):
            lr.fit(np.array(self.targets), ca[:, i])
            r = lr.predict(np.array(self.targets))
            #plt.plot(np.arange(self.nb_inputs), r)
            #plt.plot(np.arange(self.nb_inputs), ca[:, i], label='true')
            #plt.show()
            mds.append(np.max(r) - np.min(r))
        return mds

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

    #  ========  Plotting functions  =========
    def plot_output(self, outfile_name=None):
        plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
        rates = self.conditioned_activities()
        original_targets = [np.array([np.cos(2 * np.pi * i / self.nb_inputs),
                                      np.sin(2 * np.pi * i / self.nb_inputs)]) for i in range(self.nb_inputs)]
        for k in range(self.nb_inputs):
            u = self.V @ rates[k] - self.targets[k] + original_targets[k]
            plt.scatter(u[0], u[1], s=8,
                        facecolor=target_colors[k], edgecolors='white', lw=0.2, zorder=10)
            plt.scatter(original_targets[k][0],  original_targets[k][1], s=13,
                        facecolor=target_colors[k], edgecolors='black', lw=0.4)
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
