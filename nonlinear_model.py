import numpy as np
import matplotlib.pyplot as plt
from utils import target_colors, units_convert, build_data_container
import activation_functions
import copy
import itertools
from math import factorial
from scipy.linalg import subspace_angles
from scipy.optimize import fsolve
from scipy.linalg import solve as scipy_solve
from sklearn.linear_model import LinearRegression
from decoder import Decoder
plt.style.use('rnn4bci_plot_params.dms')


class NonlinearDeterministicNetwork:
    def __init__(self, network_size=100, nb_readouts=100, nb_inputs=6, exponent_W=0.55,
                 global_mean_input_is_zero=False, rng_seed=1, activation_function='tanh'):
        self.nb_inputs, self.network_size, self.output_size = nb_inputs, network_size, 2
        self.rng = np.random.default_rng(rng_seed)
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
        if global_mean_input_is_zero:
            self.inputs = [-np.ones(self.nb_inputs) / self.nb_inputs for _ in range(self.nb_inputs)]
        else:
            self.inputs = [np.zeros(self.nb_inputs) for _ in range(self.nb_inputs)]
        for i in range(self.nb_inputs):
            self.inputs[i][i] = 1. + self.inputs[i][i]

        # Network parameter initialization (see # Decoder below for V)
        self.U, self.W, self.b = self.init_params(exponent_W=exponent_W)

        # Perturbations
        self.selected_permutation = {'WM': None, 'OM': None}

        # Initial conditions for potential solver
        self.prev_potentials = [self.inv_I_minus_W() @ (self.U @ self.inputs[k] + self.b) for k in range(self.nb_inputs)]

        # Only used in LinearizedModel, but needs to be defined here 'cause I suck at coding
        self.init_conditional_potentials = self.conditioned_potentials()
        self.jac_init = [self.phi_jac(v) for v in self.init_conditional_potentials]
        self.inv_I_minus_W_init = self.inv_I_minus_W()
        self.W0 = copy.copy(self.W)

        # Decoder
        if nb_readouts > network_size:
            raise ValueError("Number of readout units must be smaller than or equal to size of network.")
        readouts_units = np.nonzero(np.diag(self.network_covariance()) > 1e-6)[0][:nb_readouts]
        nb_readouts = len(readouts_units)
        print("Number of readout units", nb_readouts)
        #V = 0.5 * self.rng.standard_normal(size=(2, nb_readouts)) / nb_readouts ** 0.5  # DEBUG!!!!
        if self.activation_function == 'linear':
            V = self.rng.standard_normal(size=(2, nb_readouts)) / nb_readouts ** exponent_W
        else:
            V = 2*self.rng.standard_normal(size=(2, nb_readouts))  / nb_readouts ** exponent_W

        #initial_decoder_fac = 0.2
        #V *= (initial_decoder_fac / np.linalg.norm(V)) * (800 / nb_readouts) ** 0.5
        self.decoder = Decoder(readouts_units, self.network_size, V)

    def init_params(self, exponent_W):
        U = self.rng.uniform(low=-1, high=1, size=(self.network_size, self.nb_inputs))
        # U = self.rng.standard_normal(size=(self.network_size, self.nb_inputs)) / self.nb_inputs **
        W = self.rng.standard_normal(size=(self.network_size, self.network_size)) / self.network_size ** 0.5 #exponent_W  DEBUG
        b = self.rng.uniform(low=0, high=1, size=(self.network_size,)) if self.activation_function == 'relu' else np.zeros(self.network_size)  # DEBUG !!!
        return U, W, b

    # ============= For activity solver =============
    @staticmethod
    def F(v, W, ff_input, a_fun):
        if a_fun == 'tanh':
            return v - W @ np.tanh(v) - ff_input
        elif a_fun == 'relu':
            return v - W @ activation_functions.relu(v) - ff_input

    @staticmethod
    def dF(v, W, c, a_fun):
        if a_fun == 'tanh':
            return np.eye(W.shape[0]) - W @ activation_functions.tanh_jac(v)
        elif a_fun == 'relu':
            return np.eye(W.shape[0]) - W @ activation_functions.relu_jac(v)

    # ==============  Statistics  ==================
    def inv_I_minus_W(self):
        return np.linalg.inv(np.eye(self.network_size) - self.W)

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
                #cps.append( np.linalg.solve(np.eye(self.network_size) - self.W, self.U@self.inputs[k]+self.b) )
                #cps.append( scipy_solve(np.eye(self.network_size) - self.W, self.U@self.inputs[k]+self.b) )
                cps.append(Q@(self.U@self.inputs[k]+self.b))
        return cps

    def conditioned_activities(self):
        v = self.conditioned_potentials()
        return [self.phi(v[k]) for k in range(self.nb_inputs)]

    def mean_activity(self):
        return np.mean(self.conditioned_activities(), axis=0)

    def network_covariance(self):
        ac = np.zeros((self.network_size, self.network_size))
        cma = self.conditioned_activities()
        for k in range(self.nb_inputs):
            ac += np.outer(cma[k] - self.mean_activity(), cma[k] - self.mean_activity())
        return ac / self.nb_inputs

    def representation_similarity_matrix(self):
        """Compute RSM based on mean activities"""
        cas = np.array(self.conditioned_activities())
        return cas @ cas.T

    def neural_tangent_kernel(self):
        """Technically, for a 2 output the NTK would be a tensor: K_{ab}(x_i, x_j),
        where a,b = {x, y} and x_i, x_j are inputs. To simplify, we compute K_{xx}(x_i, x_j) + K_{yy}(x_i, x_j)."""
        K = [np.zeros((self.nb_inputs, self.nb_inputs)), np.zeros((self.nb_inputs, self.nb_inputs)),
             np.zeros((self.nb_inputs, self.nb_inputs)), np.zeros((self.nb_inputs, self.nb_inputs))]
        du_xdW, du_ydW = [], []
        cps = self.conditioned_potentials()

        for v in cps:
            M = np.linalg.inv(np.eye(self.network_size) - self.W @ self.phi_jac(v))
            du_xdW.append(np.outer(self.phi(v), self.decoder.VR()[0, :] @ self.phi_jac(v)@M))
            du_ydW.append(np.outer(self.phi(v), self.decoder.VR()[1, :] @ self.phi_jac(v)@M))

        for i in range(self.nb_inputs):
            for j in range(i, self.nb_inputs):
                K[0][i, j] = np.trace(du_xdW[i] @ du_xdW[j].T)
                K[1][i, j] = np.trace(du_xdW[i] @ du_ydW[j].T)
                K[2][i, j] = np.trace(du_ydW[i] @ du_xdW[j].T)
                K[3][i, j] = np.trace(du_ydW[i] @ du_ydW[j].T)

        for i in range(self.nb_inputs):
            for j in range(i):
                K[0][i, j] = K[0][j, i]
                K[1][i, j] = K[1][j, i]
                K[2][i, j] = K[2][j, i]
                K[3][i, j] = K[3][j, i]
        return K

    def neural_tangent_kernel_finite_diff(self):
        """Compute an approximation of the NTK using finite differences."""
        K = [np.zeros((self.nb_inputs, self.nb_inputs)), np.zeros((self.nb_inputs, self.nb_inputs)),
             np.zeros((self.nb_inputs, self.nb_inputs)), np.zeros((self.nb_inputs, self.nb_inputs))]
        du_xdW, du_ydW = ([np.zeros_like(self.W) for _ in range(self.nb_inputs)],
                          [np.zeros_like(self.W) for _ in range(self.nb_inputs)])
        W = copy.copy(self.W)
        incr = 1e-3*np.min(W)

        rates = self.conditioned_activities()
        u_W = []
        for k in range(self.nb_inputs):
            u_W.append(self.decoder(rates[k]))

        for i in range(self.network_size):
            for j in range(self.network_size):
                self.W = copy.copy(W)
                self.W[i, j] += incr
                rates = self.conditioned_activities()
                for k in range(self.nb_inputs):
                    u_W_plus_DeltaW = self.decoder(rates[k])
                    du_xdW[k][i, j] = (u_W_plus_DeltaW[0] - u_W[k][0]) / incr
                    du_ydW[k][i, j] = (u_W_plus_DeltaW[1] - u_W[k][1]) / incr

        for i in range(self.nb_inputs):
            for j in range(i, self.nb_inputs):
                K[0][i, j] = np.sum(du_xdW[i] * du_xdW[j])
                K[1][i, j] = np.sum(du_xdW[i] * du_ydW[j])
                K[2][i, j] = np.sum(du_ydW[i] * du_xdW[j])
                K[3][i, j] = np.sum(du_ydW[i] * du_ydW[j])

        for i in range(self.nb_inputs):
            for j in range(i):
                K[0][i, j] = K[0][j, i]
                K[1][i, j] = K[1][j, i]
                K[2][i, j] = K[2][j, i]
                K[3][i, j] = K[3][j, i]
        return K


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
        ac = self.network_covariance()
        ma = self.mean_activity()
        return 0.5 * np.trace(self.decoder.V @ self.decoder.R @ (ac + np.outer(ma, ma)) @ self.decoder.R.T @ self.decoder.V.T)

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
            partial_grad = np.zeros_like(self.decoder.V @ self.decoder.R)
            for k in range(self.nb_inputs):
                partial_grad += np.outer(self.decoder(potentials[k]) - self.targets[k], potentials[k])
            grad = (self.decoder.V @ self.decoder.R @ self.inv_I_minus_W()).T @ partial_grad / self.nb_inputs
        # ng = np.linalg.norm(grad)
        # threshold = 1.
        # if ng > threshold:
        #     grad = threshold*grad/ng  # gradient clipping
        #     print("Gradient clipped...")
        return grad

    def train(self, lr=1.e-2, nb_iter=int(1e3), stopping_crit=None, do_record_data=True):
        if do_record_data:
            data = build_data_container()
        else:
            data = None

        readout_var_init = self.decoder.R @ self.network_covariance() @ self.decoder.R.T

        if stopping_crit is not None:
            nb_iter = 0
        else:
            stopping_crit = 1e6

        # Learning
        i = 0
        loss = 1e9
        initial_loss = self.task_loss()
        while i < int(nb_iter) or loss > stopping_crit:
            readout_var_prev = self.decoder.R @ self.network_covariance() @ self.decoder.R.T
            if do_record_data:
                data['total_variance'].append(np.trace(readout_var_prev))
            loss = self.task_loss()

            potentials = self.conditioned_potentials()
            max_ev = self.max_eigval(potentials)
            if max_ev >= 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!\n", f"Iter {i} EIGENVALUE GREATER THAN 1\n", "!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if do_record_data:
                data['loss'].append(loss if max_ev < 1 else -1)
                data['loss_corr'].append(self.correlation_component_loss())
                data['p_ratio'].append(self.participation_ratio_(readout_var_prev))
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
                    readout_var = self.decoder.R @ self.network_covariance() @ self.decoder.R.T
                    # dVar = readout_var - readout_var_prev
                    # U_Var, _, VT_Var = np.linalg.svd(readout_var)
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
                    beta1 = np.trace(self.decoder.C @ readout_var_init @ self.decoder.C.T) / np.trace(readout_var_init)
                    beta2 = np.trace(self.decoder.C @ readout_var @ self.decoder.C.T) / np.trace(
                        readout_var)  # note that self.decoder.C is never reassigned, so it stays at its initial value
                    data['normalized_variance_explained'].append(beta2 / beta1)
                    data['f'].append(beta2)

                    if self.selected_permutation['OM'] is not None:
                        tmp1 = np.trace(self.decoder.C[:, self.selected_permutation['OM']] @ readout_var
                                        @ self.decoder.C[:, self.selected_permutation['OM']].T)
                        tmp2 = np.trace(self.decoder.C @ readout_var @ self.decoder.C.T)
                        data['R'].append(tmp1 / tmp2)
                        data['rel_proj_var_OM'].append(tmp1 / np.trace(self.decoder.C[:, self.selected_permutation['OM']] @ readout_var_init @ self.decoder.C[:, self.selected_permutation['OM']].T))

                    if self.selected_permutation['WM'] is not None:
                        _, _, VDT = np.linalg.svd(self.decoder.D)
                        _, _, VDT_WM = np.linalg.svd(self.decoder.D[:, self.selected_permutation['WM']])
                        data['A']['D'].append(np.trace(VDT[:2] @ self.decoder.C @ readout_var @ self.decoder.C.T @ VDT[:2].T))
                        data['A']['DP_WM'].append(np.trace(VDT_WM[:2] @ self.decoder.C @ readout_var @ self.decoder.C.T @ VDT_WM[:2].T))

            self.W -= lr * g
            i += 1
        return data

    def select_perturb(self, intrinsic_manifold_dim, nb_om_permuted_units=30, nb_samples=int(1e3),
                       om_select_method='original'):
        """Select the WM and OM perturbations"""
        if om_select_method != 'original' and om_select_method != 'modified':
            raise ValueError(f"OM selection method was {om_select_method} but must be either `original` or `modified`.")
        if om_select_method == 'original' and (nb_om_permuted_units > self.decoder.nb_readouts):
            raise ValueError(f"Number of requested OM permuted units, {nb_om_permuted_units}, "
                             f"should be smaller than number of readouts {self.decoder.nb_readouts}")
        nb_samples_wm = factorial(intrinsic_manifold_dim) if intrinsic_manifold_dim <= 8 else nb_samples
        nb_samples_om = max(nb_samples, nb_samples_wm) if om_select_method == 'original' else factorial(intrinsic_manifold_dim)

        permutations = {
            'WM': np.empty((nb_samples_wm, intrinsic_manifold_dim), dtype=int),
            'OM': np.empty((nb_samples_om, self.decoder.nb_readouts), dtype=int)
        }
        input_wise_losses = {
            'WM': np.empty((nb_samples_wm, self.nb_inputs)),
            'OM': np.empty((nb_samples_om, self.nb_inputs))
        }
        total_losses = {
            'WM': np.empty(nb_samples_wm),
            'OM': np.empty(nb_samples_om)
        }

        # WM
        if intrinsic_manifold_dim > 7:
            for perm_counter in range(nb_samples_wm):
                indices = np.arange(intrinsic_manifold_dim)
                self.rng.shuffle(indices)
                perm = indices
                self.decoder.apply_perturb(perm, 'WM')
                input_wise_losses['WM'][perm_counter] = self.loss_for_each_target()
                permutations['WM'][perm_counter] = perm
                total_losses['WM'][perm_counter] = self.task_loss()
        else:  # comb over all possible permutations
            for perm_counter, perm in enumerate(itertools.permutations(range(intrinsic_manifold_dim))):
                self.decoder.apply_perturb(perm, 'WM')
                input_wise_losses['WM'][perm_counter] = self.loss_for_each_target()
                permutations['WM'][perm_counter] = perm
                total_losses['WM'][perm_counter] = self.task_loss()
        print(f"Median total loss for WM perturbation : {np.median(total_losses['WM'])}")
        print(f"Median target-wise loss for WM perturbation : {np.median(input_wise_losses['WM'], axis=0)}")

        # OM
        self.decoder.restore_intuitive()
        mds = self.get_modulation_depth()[self.decoder.readout_ids]
        sorted_indices = np.argsort(mds)

        if om_select_method == 'original':
            indices_to_permute = sorted_indices[-nb_om_permuted_units:]

            for perm_counter in range(nb_samples_om):
                indices = copy.deepcopy(indices_to_permute)
                self.rng.shuffle(indices)
                indices_i = np.arange(self.decoder.nb_readouts)
                indices_i[indices_to_permute] = indices
                self.decoder.apply_perturb(indices_i, 'OM')
                input_wise_losses['OM'][perm_counter] = self.loss_for_each_target()
                total_losses['OM'][perm_counter] = self.task_loss()
                permutations['OM'][perm_counter] = indices_i
        else:
            nb_blocks = intrinsic_manifold_dim
            nb_units_per_blocks = self.decoder.nb_readouts // nb_blocks
            nb_remaining_units = self.decoder.nb_readouts % nb_blocks

            if nb_remaining_units == 0:
                nb_units_per_blocks = (self.decoder.nb_readouts - 1) // nb_blocks
                nb_remaining_units = 1 + (self.decoder.nb_readouts - 1) % nb_blocks

            partial_indices = sorted_indices[:-nb_remaining_units]
            blocks = [partial_indices[i*nb_units_per_blocks:(i+1)*nb_units_per_blocks] for i in range(nb_blocks)]
            assert len(blocks) == intrinsic_manifold_dim, "len(block) not equal to intrinsic manifold dimension"
            s = 0
            for b in range(nb_blocks):
                s += len(blocks[b])
            s += nb_remaining_units
            assert s == self.decoder.nb_readouts, f"s = {s} different from nb of readouts {self.decoder.nb_readouts}"
            for perm_counter, perm in enumerate(itertools.permutations(range(intrinsic_manifold_dim))):
                indices = np.hstack((*[blocks[i] for i in perm], sorted_indices[-nb_remaining_units:]))
                self.decoder.apply_perturb(indices, 'OM')
                input_wise_losses['OM'][perm_counter] = self.loss_for_each_target()
                total_losses['OM'][perm_counter] = self.task_loss()
                permutations['OM'][perm_counter] = indices

        print(f"\nMedian total loss for OM perturbation : {np.median(total_losses['OM'])}")
        print(f"Median target-wise loss for OM perturbation : {np.median(input_wise_losses['OM'], axis=0)}\n")

        # Return to original mapping
        self.decoder.restore_intuitive()

        # Compute median target-specific losses across all WM and OM permutations
        # median_per_target_loss = np.median(np.vstack((input_wise_losses['WM'], input_wise_losses['OM'])), axis=0, keepdims=True)
        # print(f'\nCombined median per-target loss = {median_per_target_loss}')
        # print(f'Combined median total loss = {np.sum(median_per_target_loss)}')

        # Find WM and OM permutations closest to median perturbations
        for pert_type in ['WM', 'OM']:
            # normed_diff = np.linalg.norm(input_wise_losses[pert_type] - median_per_target_loss, axis=1)  # DEBUG !!!
            normed_diff = np.linalg.norm(input_wise_losses[pert_type] -
                                         np.median(input_wise_losses[pert_type], axis=0), axis=1)
            self.selected_permutation[pert_type] = permutations[pert_type][np.argmin(normed_diff)]
            print(f"Target-wise loss for selected {pert_type} perturbation : "
                  f"{input_wise_losses[pert_type][np.argmin(normed_diff)]}")

        return self.selected_permutation, total_losses

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
        return np.array(mds)

    # ============ Methods related to dimensionality ==============
    @staticmethod
    def dimensionality_(covariance_matrix, threshold):
        w = np.linalg.eigvals(covariance_matrix)
        ranked_eigvals = np.sort(w)[::-1]
        cum_var = np.cumsum(ranked_eigvals)
        return np.nonzero(cum_var > threshold * cum_var[-1])[0][0] + 1  # +1 because array elements start at zero

    def dimensionality(self, threshold=0.99):
        readout_cov = self.decoder.R @ self.network_covariance() @ self.decoder.R.T
        return self.dimensionality_(readout_cov, threshold)

    @staticmethod
    def participation_ratio_(covariance_matrix):
        return (np.trace(covariance_matrix)) ** 2 / np.trace(covariance_matrix @ covariance_matrix)

    def participation_ratio(self):
        readout_cov = self.decoder.R @ self.network_covariance() @ self.decoder.R.T
        return self.participation_ratio_(readout_cov)

    #  ========  Plotting functions  =========
    def plot_output(self, outfile_name=None):
        plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
        rates = self.conditioned_activities()
        for k in range(self.nb_inputs):
            u = self.decoder(rates[k])
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
