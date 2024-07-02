import numpy as np
import matplotlib.pyplot as plt
plt.style.use('rnn4bci_plot_params.dms')
from utils import target_colors, units_convert
import activation_functions
import copy
import itertools
from math import factorial
from scipy.linalg import subspace_angles
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from decoder import Decoder


class NonlinearDeterministicNetwork:
    def __init__(self, network_size=100, nb_inputs=6, exponent_W=0.55,
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

        # Targets
        self.targets = [np.array([np.cos(2 * np.pi * i / self.nb_inputs),
                                  np.sin(2 * np.pi * i / self.nb_inputs)]) for i in range(self.nb_inputs)]

        # Inputs
        if self.global_mean_input_is_zero:
            self.inputs = [-np.ones(self.input_size) / self.nb_inputs] * self.nb_inputs
        else:
            self.inputs = [np.zeros(self.input_size)] * self.nb_inputs
        for i in range(self.nb_inputs):
            self.inputs[i][i] = 1. + self.inputs[i][i]

        self.U, self.W, V, self.b = self.init_params(exponent_W=exponent_W)

        # Decoder
        self.decoder = Decoder(np.arange(self.network_size), V)

        # Perturbations
        self.selected_permutation_WM = None
        self.selected_permutation_OM = None

        # Initial conditions for potential solver
        self.prev_potentials = [self.inv_I_minus_W() @ (self.U @ self.inputs[k] + self.b) for k in range(self.nb_inputs)]

    def init_params(self, exponent_W):
        U = self.rng.uniform(low=-1, high=1, size=(self.network_size, self.input_size))
        #U = self.rng.standard_normal(size=(self.network_size, self.input_size)) / self.input_size **
        W = self.rng.standard_normal(size=(self.network_size, self.network_size)) / self.network_size ** exponent_W
        V = self.rng.standard_normal(size=(2, self.network_size))
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
        if self.activation_function != 'linear':
            for k in range(self.nb_inputs):
                sol, _, ier, _ = fsolve(self.F, self.prev_potentials[k],
                                        args=(self.W, self.U@self.inputs[k]+self.b, self.activation_function), fprime=self.dF,
                                        full_output=True)
                if ier:
                    cps.append(sol)
                    self.prev_potentials[k] = sol

                else:
                    raise Exception("Root not found")
        else:
            Q = self.inv_I_minus_W()
            for k in range(self.nb_inputs):
                cps.append(Q@(self.U@self.inputs[k]+self.b))
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
        return self.decoder.R @ ac @ self.decoder.R / self.nb_inputs

    def activity_correlation_matrix(self):
        S_v_inv = np.diag(np.sqrt(np.diag(self.activity_covariance())) ** -1)
        return S_v_inv @ self.activity_covariance() @ S_v_inv

    # ==============  Loss ================
    def task_loss(self):
        rates = self.conditioned_activities()
        L = 0.
        for k in range(self.nb_inputs):
            error = self.decoder(rates[k]) - self.targets[k]
            L += np.dot(error, error)
        return 0.5 * L / self.nb_inputs

    def loss_for_each_target(self):
        losses = np.zeros(self.nb_inputs)
        rates = self.conditioned_activities()
        for k in range(self.nb_inputs):
            error = self.decoder(rates[k]) - self.targets[k]
            losses[k] = 0.5 * np.dot(error, error)
        return losses / self.nb_inputs

    def correlation_component_loss(self):
        """TODO: Check if formula still valid with nonlinearity."""
        ac = self.activity_covariance()
        ma = self.decoder.R @ self.mean_activity()
        return 0.5 * np.trace(self.decoder.V @ (ac + np.outer(ma, ma)) @ self.decoder.V.T)

    # =========  Training ==========
    def max_eigval(self, potentials):
        return np.max([np.max(np.abs(np.linalg.eigvals(self.W @ self.phi_jac(potentials[k])))) for k in range(self.nb_inputs)])

    def compute_gradient(self):
        potentials = self.conditioned_potentials()
        if self.activation_function != 'linear':
            grad = np.zeros_like(self.W)
            for k in range(self.nb_inputs):
                J = self.phi_jac(potentials[k])
                error = self.decoder(self.phi(potentials[k]))- self.targets[k]
                grad += np.linalg.inv(np.eye(self.network_size) - J@self.W.T) @ J @ self.decoder.R.T @ self.decoder.V.T @ np.outer(error, self.phi(potentials[k]))
            grad /= self.nb_inputs
        else:
            partial_grad = np.zeros_like(self.V)
            for k in range(self.nb_inputs):
                partial_grad += np.outer(self.decoder(potentials[k]) - self.targets[k], potentials[k])
            grad = (self.decoder.R @ self.decoder.V @ self.inv_I_minus_W()).T @ partial_grad / self.nb_inputs
        # ng = np.linalg.norm(grad)
        # threshold = 1.
        # grad = threshold*grad/ng if ng >= threshold else grad  # gradient clipping
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

        var_init = self.activity_covariance()

        if stopping_crit is not None:
            nb_iter = 0
        else:
            stopping_crit = 1e6

        # Learning
        i = 0
        loss = 1e9
        initial_loss = self.task_loss()
        while i < int(nb_iter) or loss > stopping_crit:
            var_prev = self.activity_covariance()
            if do_record_data:
                data['tot_var'].append(np.trace(var_prev))
            loss = self.task_loss()

            potentials = self.conditioned_potentials()
            max_ev = self.max_eigval(potentials)
            if max_ev >= 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!\n", "EIGENVALUE GREATER THAN 1\n", "!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if do_record_data:
                data['losses']['task'].append(loss if max_ev < 1 else -1)
                data['losses']['corr'].append(self.correlation_component_loss())
                data['pr'].append(self.participation_ratio())
                data['max_eigvals'].append(max_ev)

            if nb_iter == 0:
                if i % 100 == 0:
                    print(f"Iteration {i:>4} : loss = {loss:.10e}  |  relative loss = {loss/initial_loss:.10e}")
            elif nb_iter > 5:
                if i % (nb_iter // 5) == 0 or i == nb_iter - 1:
                    print(f"Iteration {i:>4} : loss = {loss:.10e}  |  relative loss = {loss/initial_loss:.10e}")

            # Compute gradient
            g = self.compute_gradient()

            if self.decoder.C is not None:
                if do_record_data:
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
                    # data['max_angles']['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.decoder.C.T)[0]))
                    #
                    # data['min_angles']['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[-1]))
                    # data['min_angles']['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[-1]))
                    # data['min_angles']['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[-1]))
                    # data['min_angles']['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.decoder.C.T)[-1]))

                    # Compute manifold overlap (as per Feulner and Clopath)
                    beta1 = np.trace(self.decoder.C @ var_init @ self.decoder.C.T) / np.trace(var_init)
                    beta2 = np.trace(self.decoder.C @ Var @ self.decoder.C.T) / np.trace(
                        Var)  # note that self.decoder.C is never reassigned, so it stays at its initial value
                    data['normalized_variance_explained'].append(beta2 / beta1)
                    data['f'].append(beta2)

                    if self.selected_permutation_OM is not None:
                        tmp1 = np.trace(self.decoder.C[:, self.selected_permutation_OM] @ Var
                                        @ self.decoder.C[:, self.selected_permutation_OM].T)
                        tmp2 = np.trace(self.decoder.C @ Var @ self.decoder.C.T)
                        data['R'].append(tmp1 / tmp2)
                        data['rel_proj_var_OM'].append(tmp1 / np.trace(self.decoder.C[:, self.selected_permutation_OM] @ var_init @ self.decoder.C[:, self.selected_permutation_OM].T))

                    if self.selected_permutation_WM is not None:
                        _, _, VDT = np.linalg.svd(self.D)
                        _, _, VDT_WM = np.linalg.svd(self.D[:, self.selected_permutation_WM])
                        data['A']['D'].append(np.trace(VDT[:2] @ self.decoder.C @ Var @ self.decoder.C.T @ VDT[:2].T))
                        data['A']['DP_WM'].append(np.trace(VDT_WM[:2] @ self.decoder.C @ Var @ self.decoder.C.T @ VDT_WM[:2].T))

            self.W -= lr * g
            i += 1
        return data

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
            mds.append(np.max(r) - np.min(r))
        return mds

    # ============ Methods related to dimensionality ==============
    @staticmethod
    def dimensionality_(covariance_matrix, threshold):
        w = np.linalg.eigvals(covariance_matrix)
        ranked_eigvals = np.sort(w)[::-1]
        cum_var = np.cumsum(ranked_eigvals)
        return np.nonzero(cum_var > threshold * cum_var[-1])[0][0] + 1  # +1 because array elements start at zero

    def dimensionality(self, threshold=0.99):
        return self.dimensionality_(self.activity_correlation_matrix(), threshold) if self.do_z_score \
            else self.dimensionality_(self.activity_covariance(), threshold)

    @staticmethod
    def participation_ratio_(covariance_matrix):
        return (np.trace(covariance_matrix)) ** 2 / np.trace(covariance_matrix @ covariance_matrix)

    def participation_ratio(self):
        return self.participation_ratio_(self.activity_correlation_matrix()) if self.do_z_score \
            else self.participation_ratio_(self.activity_covariance())

    #  ========  Plotting functions  =========
    def plot_output(self, outfile_name=None):
        plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
        rates = self.conditioned_activities()
        for k in range(self.nb_inputs):
            u = self.V @ (rates[k] - self.ma_0) + self.intercept
            plt.scatter(u[0], u[1], s=8,
                        facecolor=target_colors[k], edgecolors='white', lw=0.2, zorder=10)
            plt.scatter(self.targets[k][0], self.targets[k][1], s=13,
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
