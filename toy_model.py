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


class ToyNetwork:
    """
    A static, linear and Gaussian neural network.

    The network activity, v, is given by
    v = (I - W)^{-1} (b + U@x + xi)
    where
    I : identity matrix
    W : recurrent weights
    b : bias
    U : input matrix
    x : input vector
    xi : isotropic Gaussian noise

    The linear readout, u, is given by
    u = V @ v
    where
    V : output matrix

    Attributes:
    ----------
    size : tuple
        Size of the network, with
        size = (input_size, recurrent_size, output_size)

    nb_inputs : int
        Number of different inputs (equivalently, the number of targets).

    private_noise_intensity : float
        Variance of private noise

    input_noise_intensity : float
        Variance of input noise.

    input_subspace_dim : int
        Rank of the input covariance matrix

    initialization_type : str or int
        Initialization type for matrices U and W.
        Either, one of 'random', 'W-zero', 'zero-all' defined in `self.init_params` member function, or
        an integer (>= 1) specifying the rank of the initialization matrix.

    network_name : str
        Name of the network, for bookkeeping.

    global_mean_input_is_zero : bool
        Whether the global mean input ( = (1/self.nb_inputs) * sum_{k=1}^{self.nb_inputs} mean_for_input_k)
        is forced to be zero.

    use_data_for_decoding : bool
        Whether to use data to fit the BCI decoder.
        The alternative (when this option is false) is to minimize the Frobenius norm of the difference between
        the initial V and DC.

    sparsity_factor : float in [0, 1]
        How sparse is the recurrent matrix.
    """
    def __init__(self, network_name, size=(100,50,2), nb_inputs=8, private_noise_intensity=1e-3,
                 input_noise_intensity=1e-2, input_subspace_dim=100, initialization_type='random', exponent_W=0.5,
                 global_mean_input_is_zero=True, use_data_for_decoding=False, orthogonalize_input_means=True,
                 rng_seed=1, sparsity_factor=1):
        self.size = size
        self.input_size, self.network_size, self.output_size = size
        self.nb_inputs = nb_inputs
        self.private_noise_intensity = private_noise_intensity
        self.input_noise_intensity = input_noise_intensity
        self.input_subspace_dim = input_subspace_dim
        self.initialization_type = initialization_type
        self.network_name = network_name
        self.global_mean_input_is_zero = global_mean_input_is_zero
        self.use_data_for_decoding = use_data_for_decoding
        self.sample_mean_v = np.empty(shape=self.network_size)
        self.rng = np.random.default_rng(rng_seed)
        self.sparsity_factor = sparsity_factor
        self.orthogonalize_input_means = orthogonalize_input_means
        self.P = None  # inverse correlation matrix

        # BCI decoder attributes
        self.decoder = None  # sklearn LinearModel object
        self.C = None  # projection matrix, shape = (intrinsic_manifold_dim, self.network_size)
        self.D = None  # decoding matrix, shape = (self.output_size, intrinsic_manifold_dim)
        self.intercept = np.zeros(self.output_size)  # intercept of the decoder

        # Input generator
        self.input_noise = self.create_input()

        # Private Gaussian noise
        self.private_noise = MultivariateNormal(cov=private_noise_intensity * np.eye(size[1]), rng=self.rng)

        # Desired targets
        if self.output_size == 2:
            self.targets = [np.array([np.cos(2 * np.pi * i / self.nb_inputs),
                                      np.sin(2 * np.pi * i / self.nb_inputs)]) for i in range(self.nb_inputs)]
        elif self.output_size == 1:
            self.targets = [np.array([1]), np.array([-1])]

        # Network parameter initialization
        self.U, self.W, self.V, self.b = self.init_params(type_=initialization_type, exponent_W=exponent_W)

        # Sparsity
        self.mask = np.ones_like(self.W)
        if self.sparsity_factor < 1.:
            self.g = 1.
            mask = self.rng.uniform(size=(self.network_size, self.network_size)) < self.sparsity_factor
            np.fill_diagonal(mask, np.zeros(self.network_size))
            self.mask = mask
            self.W = self.g * self.W * mask / self.sparsity_factor**0.5

        self.selected_permutation_WM = None
        self.selected_permutation_OM = None

    # -------------------------- Properties and initializations -------------------------- #
    @property
    def network_name(self):
        return self._network_name

    @network_name.setter
    def network_name(self, name):
        self._network_name = str(name)
        self.results_folder = f'I{self.input_size}-N{self.network_size}-K{self.nb_inputs}' \
                              f'-PrivateNoise{self.private_noise_intensity}' \
                              f'-InputNoise{self.input_noise_intensity}-' \
                              f'InputSubspaceDim{self.input_subspace_dim}-InitType_{self.initialization_type}/NetworkName-{self._network_name}'

    def create_input(self):
        if self.nb_inputs == self.input_size and self.input_noise_intensity < 1e-6:
            # 1-of-K encoding
            means = [np.zeros(self.input_size) for i in range(self.nb_inputs)]
            for i in range(self.nb_inputs):
                means[i][i] = 1.
            covs = [np.zeros((self.input_size, self.input_size))] * self.nb_inputs
        else:
            # Means
            means = [self.rng.standard_normal(self.input_size) for i in range(self.nb_inputs)]
            if self.global_mean_input_is_zero:
                global_mean = np.mean(means, axis=0)
                means = [means[i] - global_mean for i in range(self.nb_inputs)]
            means = [means[i] / np.linalg.norm(means[i]) for i in range(self.nb_inputs)]

            if self.orthogonalize_input_means:
                means = gram_schmidt(means)
            # Covariances
            if self.input_subspace_dim == self.input_size:
                covs = [self.input_noise_intensity * np.eye(self.input_size)] * self.nb_inputs
            else:
                covs = []
                for component_i in range(self.nb_inputs):
                    Q = self.rng.standard_normal((self.input_size, self.input_subspace_dim))
                    covs.append(self.input_noise_intensity * Q @ Q.T / self.input_subspace_dim)
        return GaussianMixture(means, covs, rng=self.rng)

    def init_params(self, type_='random', exponent_W=0.5):
        if type_ == 'random':
            if self.nb_inputs == self.input_size and self.input_noise_intensity < 1e-6:
                U = self.rng.uniform(low=-1, high=1, size=(self.network_size, self.input_size))
            else:
                U = self.rng.standard_normal(size=(self.network_size, self.input_size)) / self.input_size ** 0.5
            W = self.rng.standard_normal(size=(self.network_size, self.network_size)) / self.network_size ** exponent_W
        elif type_ == 'W-zero':
            if self.nb_inputs == self.input_size and self.input_noise_intensity < 1e-6:
                U = self.rng.uniform(low=-1, high=1, size=(self.network_size, self.input_size))
            else:
                U = self.rng.standard_normal(size=(self.network_size, self.input_size)) / self.input_size ** 0.5
            W = np.zeros(shape=(self.network_size, self.network_size))
        elif type_ == 'zero-all':
            U = np.zeros(shape=(self.network_size, self.input_size))
            W = np.zeros(shape=(self.network_size, self.network_size))
        elif isinstance(type_, int):
            U = np.zeros(shape=(self.network_size, self.input_size))
            W = np.zeros(shape=(self.network_size, self.network_size))
            for i in range(type_):
                W += np.outer(self.rng.standard_normal(size=self.network_size) / self.network_size ** 0.5,
                              self.rng.standard_normal(size=self.network_size) / self.network_size ** 0.5)
                U += np.outer(self.rng.standard_normal(size=self.network_size) / self.network_size ** 0.5,
                              self.rng.standard_normal(size=self.input_size) / self.input_size ** 0.5)
        if self.nb_inputs == self.input_size and self.input_noise_intensity < 1e-6:
            V = self.rng.standard_normal(size=(2, self.network_size))
            initial_decoder_fac = 0.2
            V *= (initial_decoder_fac / np.linalg.norm(V)) * (800 / self.network_size) ** 0.5
        else:
            V = self.rng.standard_normal(size=(self.output_size, self.network_size)) / self.network_size ** 0.5
        b = np.zeros(self.network_size)

        return U, W, V, b


    # -------------------------- Loss-related functions -------------------------- #
    def loss_function(self):
        # Boilerplate because nested mutable elements (list of numpy.arrays) will be fed to numba.njit
        input_noise_covs = nbList()
        [input_noise_covs.append(x) for x in self.input_noise.covs]
        input_noise_means = nbList()
        [input_noise_means.append(x) for x in self.input_noise.means]
        targets = nbList()
        [targets.append(x) for x in self.targets]

        # Call to numba accelerated function
        L = _loss_function_jit(self.network_size, self.nb_inputs,
                               self.U, self.W, self.V, self.b,
                               input_noise_covs, input_noise_means, self.private_noise.cov,
                               targets)
        return L

    def loss_for_each_target(self):
        input_noise_covs = nbList()
        [input_noise_covs.append(x) for x in self.input_noise.covs]
        input_noise_means = nbList()
        [input_noise_means.append(x) for x in self.input_noise.means]
        targets = nbList()
        [targets.append(x) for x in self.targets]

        # Call to numba accelerated function
        return _loss_for_each_target_jit(self.network_size, self.nb_inputs,
                                         self.U, self.W, self.V, self.b,
                                         input_noise_covs, input_noise_means, self.private_noise.cov, targets)

    def loss_function_components(self):
        """ The loss function can be expressed as L = L_1 + L_2,
        where L_1 is a term including conditioned variances and
        L_2 contains the expected error

        :return:
        L_1, L_2
        """
        I = np.eye(self.network_size)
        L_var, L_exp = 0, 0
        inv_I_minus_W = np.linalg.inv(I - self.W)

        for c, p in enumerate(self.input_noise.p):
            Var_v_given_k = inv_I_minus_W @ (self.U @ self.input_noise.covs[c] @ self.U.T +
                                             self.private_noise.cov) @ inv_I_minus_W.T
            L_var += 0.5 * p * np.trace(self.V @ Var_v_given_k @ self.V.T)

            Exp_v_given_k = inv_I_minus_W @ (self.b + self.U @ self.input_noise.means[c])
            L_exp += 0.5 * p * np.linalg.norm(self.V @ Exp_v_given_k - self.targets[c])**2

        return L_var, L_exp

    def floor_loss(self):
        I = np.eye(self.network_size)
        inv_I_minus_W = np.linalg.inv(I - self.W)
        return 0.5*np.trace(self.V @ inv_I_minus_W @ self.private_noise.cov @ inv_I_minus_W.T @ self.V.T)

    def loss_components_correlation_and_projection(self):
        """Compute the loss as L = 0.5 + 0.5 * tr[V(Cov + vbar@vbar.T)V] - mean projection.

        Returns:
        ------
        L_corr : float
            0.5 * tr[V(Var + vbar@vbar.T)V]
        L_proj : float
            - mean projection
        """
        Cov = self.compute_total_covariance()
        vbar = self.get_mean_activity()
        L_corr = 0.5 * np.trace(self.V @ (Cov + np.outer(vbar, vbar)) @ self.V.T)

        L_proj = 0
        inv_I_minus_W = np.linalg.inv(np.eye(self.network_size) - self.W)
        for c, p in enumerate(self.input_noise.p):
            Exp_v_given_k = inv_I_minus_W @ (self.b + self.U @ self.input_noise.means[c])
            L_proj -= p * self.targets[c] @ (self.V @ Exp_v_given_k)
        return L_corr, L_proj

    # -------------------------- Functions related to training of the network -------------------------- #
    def compute_gradient(self):
        """Compute the exact gradient."""
        inv_I_minusW = np.linalg.inv(np.eye(self.network_size) - self.W)
        Vars_given_k = self.compute_conditioned_covariances()
        Exps_given_k = self.get_conditioned_mean_activity()
        recast_op = np.linalg.inv(np.eye(self.network_size) - self.W.T) @ self.V.T


        grad = {'U': np.zeros_like(self.U), 'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        for k, p in enumerate(self.input_noise.p):
            error_k = self.V @ Exps_given_k[k]-self.targets[k]
            grad['U'] += p * recast_op @ (
                    self.V @ inv_I_minusW @ self.U @ self.input_noise.covs[k]
                    + np.outer(error_k, self.input_noise.means[k])
            )
            grad['W'] += p * recast_op @ (self.V @ Vars_given_k[k] + np.outer(error_k, Exps_given_k[k]))
            grad['b'] += p * recast_op @ error_k

        """
        grad = {'U': np.zeros_like(self.U.T), 'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        prefactors = {'(I-W).-TV.TV(I-W)^-1': np.transpose(np.linalg.inv(I - self.W)) @ self.V.T @ self.V @ np.linalg.inv(I - self.W),
                      'V(I-W)^-1': self.V @ np.linalg.inv(I - self.W),
                      '(I-W)^-1': np.linalg.inv(I - self.W)}
        for c, p in enumerate(self.input_noise.p):
            grad['U'] += p * (self.input_noise.covs[c] @ self.U.T + np.outer(self.input_noise.means[c], (self.b + self.U @ self.input_noise.means[c]))) @ prefactors[
                '(I-W).-TV.TV(I-W)^-1'] \
                         + p * np.outer(self.input_noise.means[c], self.intercept - self.targets[c]) @ prefactors['V(I-W)^-1']
            grad['W'] += p * prefactors['(I-W)^-1'] @ (
                    self.U @ self.input_noise.covs[c] @ self.U.T + self.private_noise.cov + np.outer(self.b + self.U @ self.input_noise.means[c], self.b + self.U @ self.input_noise.means[c])) @ prefactors[
                             '(I-W).-TV.TV(I-W)^-1'] + p * (
                                 prefactors['(I-W)^-1'] @ np.outer(self.b + self.U @ self.input_noise.means[c], self.intercept - self.targets[c]) @ prefactors[
                             'V(I-W)^-1'])
            grad['b'] += p * (self.b + self.U @ self.input_noise.means[c]).T @ prefactors['(I-W).-TV.TV(I-W)^-1'] + p * (self.intercept -self.targets[c]) @ prefactors[
                'V(I-W)^-1']
        for key in grad.keys():
            grad[key] = grad[key].T
        """
        return grad

    def compute_gradient_component(self, component_name):
        if component_name == 'loss_tot_var':
            return np.linalg.inv(np.eye(self.network_size) - self.W.T) @ self.V.T @ self.V @ self.compute_total_covariance()
        else:
            print(f"{component_name} not yet implemented")
            return

    def train_with_node_perturbation(self, lr=(1e-3, 1.e-3, 1e-3), nb_iter=int(1e3)):
        I = np.eye(self.network_size)
        losses = {'total': np.empty(shape=nb_iter),  # total loss
                  'var': np.empty(shape=nb_iter),  # loss component related to target-conditioned variances
                  'exp': np.empty(shape=nb_iter),  # loss component related to error
                  'corr': np.empty(shape=nb_iter),  # loss component related to total correlation
                  'proj': np.empty(shape=nb_iter),
                  # loss component related to projection of conditioned output on target
                  'vbar': np.empty(shape=nb_iter)}  # loss subcomponent related to global mean
        norm_gradW = {'loss': [], 'loss_tot_var': []}
        max_angles = {'dVar_vs_VT': np.empty(shape=nb_iter),
                      'UpperVar_vs_VT': np.empty(shape=nb_iter),
                      'LowerVar_vs_VT': np.empty(shape=nb_iter),
                      'UpperVar_vs_VarBCI': np.empty(shape=nb_iter)}
        min_angles = {'dVar_vs_VT': np.empty(shape=nb_iter),
                      'UpperVar_vs_VT': np.empty(shape=nb_iter),
                      'LowerVar_vs_VT': np.empty(shape=nb_iter),
                      'UpperVar_vs_VarBCI': np.empty(shape=nb_iter)}
        normalized_variance_explained = np.empty(shape=nb_iter)
        R = np.empty(shape=nb_iter)
        A = {'D': np.empty(shape=nb_iter),
             'DP_WM': np.empty(shape=nb_iter)}

        if self.D is not None:
            d = self.D.shape[1]

        Var_init = self.compute_total_covariance()

        # Learning
        loss = []
        for i in range(int(nb_iter)):
            # Compute loss and loss components
            losses['var'][i], losses['exp'][i] = self.loss_function_components()
            losses['total'][i] = losses['var'][i] + losses['exp'][i]
            losses['corr'][i], losses['proj'][i] = self.loss_components_correlation_and_projection()
            losses['vbar'][i] = 0.5 * np.linalg.norm(self.V @ self.get_mean_activity()) ** 2

            if i % int(nb_iter / 10) == 0:
                print(f"Loss at iteration {i} = {losses['total'][i]} with learning rate = {lr}")

            inv_I_minus_W = np.linalg.inv(I - self.W)
            inv_I_minus_WT = np.linalg.inv(I - self.W.T)

            X = self.input_noise.draw_each_component_once()
            noise = self.private_noise.draw(size=self.nb_inputs)

            gW = np.zeros_like(self.W)
            gU = np.zeros_like(self.U)
            gb = np.zeros_like(self.b)

            for k in range(self.nb_inputs):
                # Noiseless loss eval
                v0 = inv_I_minus_W @ (self.b + self.U @ X[k])
                u = self.V @ v0 + self.intercept
                L0 = 0.5 * np.linalg.norm(self.targets[k] - u)**2

                # Perturbation
                v = inv_I_minus_W @ (self.b + self.U @ X[k] + noise[k])
                u = self.V @ v + self.intercept
                L = 0.5 * np.linalg.norm(self.targets[k] - u)**2

                # Update gradients
                gU += (L - L0) * np.outer(noise[k], X[k]) / self.private_noise_intensity
                gW += (L - L0) * (-inv_I_minus_WT + np.outer(noise[k], v0) / self.private_noise_intensity
                                  + np.outer(noise[k], noise[k])@inv_I_minus_WT/self.private_noise_intensity)
                gb += (L - L0) * noise[k] / self.private_noise_intensity

            gU = gU / self.nb_inputs
            gW = gW * self.mask / self.nb_inputs
            gb = gb / self.nb_inputs

            if self.C is not None:
                # Compute norm of the gradient
                norm_gradW['loss'].append(np.linalg.norm(gW))
                norm_gradW['loss_tot_var'].append(np.linalg.norm(self.compute_gradient_component('loss_tot_var')))

                # Compute angles
                dVar = self.compute_dVar(-lr[1] * gW * self.mask, -lr[0] * gU)
                Var = self.compute_total_covariance()
                U_Var, _, VT_Var = np.linalg.svd(Var)
                upper_var = U_Var[:, :d]
                lower_var = U_Var[:, d:]

                max_angles['dVar_vs_VT'][i] = np.rad2deg(subspace_angles(dVar, self.V.T)[0])
                max_angles['UpperVar_vs_VT'][i] = np.rad2deg(subspace_angles(upper_var, self.V.T)[0])
                max_angles['LowerVar_vs_VT'][i] = np.rad2deg(subspace_angles(lower_var, self.V.T)[0])
                max_angles['UpperVar_vs_VarBCI'][i] = np.rad2deg(subspace_angles(upper_var, self.C.T)[0])

                min_angles['dVar_vs_VT'][i] = np.rad2deg(subspace_angles(dVar, self.V.T)[-1])
                min_angles['UpperVar_vs_VT'][i] = np.rad2deg(subspace_angles(upper_var, self.V.T)[-1])
                min_angles['LowerVar_vs_VT'][i] = np.rad2deg(subspace_angles(lower_var, self.V.T)[-1])
                min_angles['UpperVar_vs_VarBCI'][i] = np.rad2deg(subspace_angles(upper_var, self.C.T)[-1])

                # Compute manifold overlap (as per Feulner and Clopath)
                beta1 = np.trace(self.C @ Var_init @ self.C.T) / np.trace(Var_init)
                beta2 = np.trace(self.C @ Var @ self.C.T) / np.trace(Var)  # note that self.C is never reassigned, so it stays at its initial value
                normalized_variance_explained[i] = beta2 / beta1

                if self.selected_permutation_OM is not None:
                    tmp1 = np.trace(self.C[:, self.selected_permutation_OM] @ Var
                                    @ self.C[:, self.selected_permutation_OM].T)
                    tmp2 = np.trace(self.C @ Var @ self.C.T)
                    R[i] = tmp1/tmp2

                if self.selected_permutation_WM is not None:
                    _, _, VDT = np.linalg.svd(self.D)
                    _, _, VDT_WM = np.linalg.svd(self.D[:, self.selected_permutation_WM])
                    A['D'][i] = np.trace(VDT[:2] @ self.C @ Var @ self.C.T @ VDT[:2].T)
                    A['DP_WM'][i] = np.trace(VDT_WM[:2] @ self.C @ Var @ self.C.T @ VDT_WM[:2].T)

            self.U -= lr[0] * gU
            self.W -= lr[1] * gW
            self.b -= lr[2] * gb
        return losses, norm_gradW, min_angles, max_angles, normalized_variance_explained, R, A

    def train_with_force(self, nb_iter=1e3, delta=0.1, lambda_=0.9, noise_level=0., fraction_silent=0, wplastic=None):
        if wplastic is None:
            self.W_plastic = [np.where(self.W[i, :] != 0)[0] for i in range(self.network_size)]
        else:
            self.W_plastic = wplastic

        #if self.P is None:
            # covariance = self.compute_covariance()
            # mean_ = self.get_mean_activity()
            # mean_v_square = np.trace(covariance['v'][self.W_plastic[0], self.W_plastic[0]] +
            #                         np.outer(mean_[self.W_plastic[0]], mean_[self.W_plastic[0]]))
            # print(f"E[v^2] = {mean_v_square}")
            # delta = 1e-3 * mean_v_square
        P = [1/delta * np.eye(len(self.W_plastic[i])) for i in range(len(self.W_plastic))]
            # mean_p = 0
            # for p in P:
            #     mean_p += np.trace(p) / p.shape[0]
            # print(f"Mean p = {mean_p / len(P)}")
        #else:
        #    P = copy.deepcopy(self.P)

        # Define feedback matrix
        W_fb = np.linalg.pinv(self.V)
        if noise_level > 0:
            W_fb = W_fb + noise_level*np.std(W_fb)*self.rng.standard_normal((self.network_size, self.output_size))
        elif fraction_silent > 0.:
            mask = self.rng.uniform(size=W_fb.shape[0]) < fraction_silent
            W_fb[mask, :] = 0

        # Learning
        loss = []
        I = np.eye(self.network_size)
        for iter in range(int(nb_iter)):
            exact_loss = self.loss_function()
            loss.append(exact_loss)

            x, component = self.input_noise.draw(size=1)
            noise = self.private_noise.draw(size=1)
            v = np.linalg.inv(I - self.W) @ (self.b + self.U @ x[0] + noise[0])
            u = self.V @ v + self.intercept
            e = W_fb @ (u - self.targets[np.nonzero(component)[0][0]])

            for i in range(self.network_size):
                v_plastic = v[self.W_plastic[i]]
                Pi_v_plastic = P[i]@v_plastic
                norm = lambda_ + v_plastic@Pi_v_plastic
                P[i] = P[i]/lambda_ - (1/lambda_)*np.outer(Pi_v_plastic, Pi_v_plastic) / norm
                self.W[i, self.W_plastic[i]] -= e[i]*P[i]@v_plastic
            if iter % int(nb_iter / 10) == 0:
                print(f"Loss at iteration {iter} = {exact_loss}")

        if self.P is None:
            self.P = copy.deepcopy(P)
        return loss

    def train(self, lr=(1e-3, 1.e-3, 1e-3), nb_iter=int(1e3), stopping_crit=None):
        losses = {'total': [],  # total loss
                  'var': [],    # loss component related to target-conditioned variances
                  'exp': [],    # loss component related to error
                  'corr': [],   # loss component related to total correlation
                  'proj': [],   # loss component related to projection of conditioned output on target
                  'vbar': []}   # loss subcomponent related to global mean
        norm_gradW = {'loss': [], 'loss_tot_var': []}
        max_angles = {'dVar_vs_VT': [],
                      'UpperVar_vs_VT': [],
                      'LowerVar_vs_VT': [],
                      'UpperVar_vs_VarBCI': []}
        min_angles = {'dVar_vs_VT': [],
                      'UpperVar_vs_VT': [],
                      'LowerVar_vs_VT': [],
                      'UpperVar_vs_VarBCI': []}
        normalized_variance_explained = []
        R = []
        A = {'D': [],
             'DP_WM': []}
        f = []
        rel_proj_var_OM = []

        if self.D is not None:
            d = self.D.shape[1]

        Var_init = self.compute_total_covariance()

        if stopping_crit is not None:
            nb_iter = 0
        else:
            stopping_crit = 1e6

        # Learning
        i = 0
        loss = 1e9
        while i < int(nb_iter) or loss > stopping_crit:
            # Compute loss and loss components
            loss_var, loss_exp = self.loss_function_components()
            losses['var'].append(loss_var)
            losses['exp'].append(loss_exp)
            losses['total'].append(loss_var + loss_exp)
            loss_corr, loss_proj = self.loss_components_correlation_and_projection()
            losses['corr'].append(loss_corr)
            losses['proj'].append(loss_proj)
            losses['vbar'].append(0.5*np.linalg.norm(self.V @ self.get_mean_activity())**2)
            loss = loss_var + loss_exp

            if nb_iter == 0:
                if i % 500 == 0:
                    print(f"Loss at iteration {i} = {loss}")
            elif i % int(nb_iter / 5) == 0 or i == nb_iter-1:
                print(f"Loss at iteration {i} = {loss} | floor ratio = {self.floor_loss() / loss}")

            # Compute gradient
            g = self.compute_gradient()

            if self.C is not None:
                # Compute norm of the gradient
                norm_gradW['loss'].append(np.linalg.norm(g['W']))
                norm_gradW['loss_tot_var'].append(np.linalg.norm(self.compute_gradient_component('loss_tot_var')))

                # Compute angles
                dVar = self.compute_dVar(-lr[1] * g['W'] * self.mask, -lr[0] * g['U'])
                Var = self.compute_total_covariance()
                U_Var, _, VT_Var = np.linalg.svd(Var)
                upper_var = U_Var[:, :d]
                lower_var = U_Var[:, d:]

                max_angles['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[0]))
                max_angles['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[0]))
                max_angles['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[0]))
                max_angles['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.C.T)[0]))

                min_angles['dVar_vs_VT'].append(np.rad2deg(subspace_angles(dVar, self.V.T)[-1]))
                min_angles['UpperVar_vs_VT'].append(np.rad2deg(subspace_angles(upper_var, self.V.T)[-1]))
                min_angles['LowerVar_vs_VT'].append(np.rad2deg(subspace_angles(lower_var, self.V.T)[-1]))
                min_angles['UpperVar_vs_VarBCI'].append(np.rad2deg(subspace_angles(upper_var, self.C.T)[-1]))

                # Compute manifold overlap (as per Feulner and Clopath)
                beta1 = np.trace(self.C @ Var_init @ self.C.T) / np.trace(Var_init)
                beta2 = np.trace(self.C @ Var @ self.C.T) / np.trace(Var)  # note that self.C is never reassigned, so it stays at its initial value
                normalized_variance_explained.append(beta2 / beta1)
                f.append(beta2)

                if self.selected_permutation_OM is not None:
                    tmp1 = np.trace(self.C[:, self.selected_permutation_OM] @ Var
                                    @ self.C[:, self.selected_permutation_OM].T)
                    tmp2 = np.trace(self.C @ Var @ self.C.T)
                    R.append(tmp1/tmp2)
                    rel_proj_var_OM.append(tmp1 / np.trace(self.C[:, self.selected_permutation_OM] @ Var_init
                                    @ self.C[:, self.selected_permutation_OM].T))

                if self.selected_permutation_WM is not None:
                    _, _, VDT = np.linalg.svd(self.D)
                    _, _, VDT_WM = np.linalg.svd(self.D[:, self.selected_permutation_WM])
                    A['D'].append(np.trace(VDT[:2] @ self.C @ Var @ self.C.T @ VDT[:2].T))
                    A['DP_WM'].append(np.trace(VDT_WM[:2] @ self.C @ Var @ self.C.T @ VDT_WM[:2].T))

            #if nb_iter == 0:
            #    if i % 500 == 0:
            #        print(f"Max gradient L_V[v] w.r.t. W {i} = {np.max(self.compute_gradient_component('loss_tot_var'))}")
            #elif i % int(nb_iter / 5) == 0 or i == nb_iter-1:
            #    print(f"Max gradient  L_V[v]  w.r.t. W {i} = {np.max(self.compute_gradient_component('loss_tot_var'))}")

            # Parameter update
            self.U -= lr[0] * g['U']
            self.W -= lr[1] * g['W'] * self.mask
            self.b -= lr[2] * g['b']
            max_eig_valW = np.max(np.abs(np.linalg.eigvals(self.W)))
            #if max_eig_valW > 1:
            #    print(f"UNSTABLE: maximum eigval of W = {max_eig_valW}")
            i += 1
        return losses, norm_gradW, min_angles, max_angles, normalized_variance_explained, R, A, f, rel_proj_var_OM

    # -------------------------- Functions related to statistics of the network -------------------------- #
    def compute_covariance(self):
        I = np.eye(self.network_size)
        factors = {'(I-W)^-1U': np.linalg.inv(I - self.W) @ self.U,
                   '(I-W)^-1': np.linalg.inv(I - self.W)}
        cov = {'v': np.zeros_like(self.W),
               'x': np.zeros(shape=(self.U.shape[1], self.U.shape[1])),
               'U_comp_var': np.zeros_like(self.W),
               'U_comp_mean': np.zeros_like(self.W),
               'priv_noise_comp': np.zeros_like(self.W)}
        x_bar = self.input_noise.global_mean()
        for c, p in enumerate(self.input_noise.p):
            cov['U_comp_var'] += p * factors['(I-W)^-1U'] @ self.input_noise.covs[c] @ factors['(I-W)^-1U'].T
            cov['U_comp_mean'] += p * factors['(I-W)^-1U'] @ np.outer(self.input_noise.means[c] - x_bar, self.input_noise.means[c] - x_bar) @ factors[
                '(I-W)^-1U'].T
            cov['x'] += p * (self.input_noise.covs[c] + np.outer(self.input_noise.means[c] - x_bar, self.input_noise.means[c] - x_bar))
        cov['priv_noise_comp'] = factors['(I-W)^-1'] @ self.private_noise.cov @ factors['(I-W)^-1'].T

        cov['v'] = cov['U_comp_var'] + cov['U_comp_mean'] + cov['priv_noise_comp']
        return cov

    def compute_total_covariance(self):
        """
        Compute V[v].

        Return:
        ------
        total_covariance : array with shape = (`self.network_size`, `self.network_size`)
        """
        conditioned_vars = self.compute_conditioned_covariances()
        conditioned_means = self.get_conditioned_mean_activity()
        global_mean = self.get_mean_activity()

        total_covariance = np.zeros(shape=(self.network_size, self.network_size))
        for k, p in enumerate(self.input_noise.p):
            total_covariance += p * (conditioned_vars[k] +
                                     np.outer(conditioned_means[k] - global_mean, conditioned_means[k] - global_mean))
        return total_covariance

    def compute_total_input_covariance(self):
        """
        Compute V[x].

        Return:
        ------
        """
        Vx = np.zeros(shape=(self.input_size, self.input_size))
        x_bar = self.input_noise.global_mean()

        for k, p in enumerate(self.input_noise.p):
            Vx += p * (self.input_noise.covs[k] + np.outer(self.input_noise.means[k] - x_bar, self.input_noise.means[k] - x_bar))
        return Vx

    def compute_conditioned_covariances(self):
        """
        Compute V[v | k] for k = 0, ..., `self.nb_inputs`.

        Return:
        ------
        variances : list of `self.nb_inputs` arrays of shape = (`self.network_size`, `self.network_size`)
        """
        variances = []
        inv_I_minusW = np.linalg.inv(np.eye(self.network_size) - self.W)
        for k in range(self.nb_inputs):
            variances.append(
                inv_I_minusW @ (self.U @ self.input_noise.covs[k] @ self.U.T + self.private_noise.cov) @ inv_I_minusW.T
            )
        return variances

    def compute_dVar(self, dW, dU):
        """
        Compute dVar[v], the differential of the total covariance.

        Parameter:
        ---------
        dW : array of shape=(`self.network_size`, `self.network_size`)
            Weight update (= -eta_W * grad_W)
        """
        tot_var = self.compute_total_covariance()
        Vx = self.compute_total_input_covariance()
        inv_I_minusW = np.linalg.inv(np.eye(self.network_size) - self.W)
        Wterm = inv_I_minusW @ dW @ tot_var
        #(I - W)^{-1}dU \var{\x}U^\T (I - W)^{-\T}
        Uterm = inv_I_minusW @ dU @ Vx @ self.U.T @ inv_I_minusW.T
        return Wterm + Wterm.T + Uterm + Uterm.T

    def get_mean_activity(self):
        x_bar = self.input_noise.global_mean()
        return np.linalg.inv(np.eye(self.network_size) - self.W)@(self.b + self.U@x_bar)

    def get_conditioned_mean_activity(self):
        """
        Compute E[v | k] for k = 0, ..., `self.nb_inputs`.

        Return:
        ------
        means : list of `self.nb_inputs` arrays of shape = (`self.network_size`, )
        """
        return [np.linalg.inv(np.eye(self.network_size) - self.W)@(self.b + self.U@self.input_noise.means[i])
                for i in range(self.nb_inputs)]

    def _exp_u_square_given_k(self, mean, cov):
        I = np.eye(self.network_size)
        M = self.V @ np.linalg.inv(I - self.W) @ \
            (self.U @ cov @ self.U.T + self.private_noise.cov + np.outer(self.b + self.U @ mean, self.b + self.U @ mean)) \
            @ np.transpose(self.V @ np.linalg.inv(I - self.W))
        return np.trace(M)

    def get_modulation_depth(self):
        """
        Compute max - min expected activity across target to get an approximate modulation depth
        (one would need to compute tuning curve for a more appropriate value).

        Return:
        ------
        modulation_depth : 1D array of shape (`self.network_size`, )
        """
        conditioned_mean = np.vstack(self.get_conditioned_mean_activity())
        min_activity = np.min(conditioned_mean, axis=0)
        max_activity = np.max(conditioned_mean, axis=0)
        return max_activity - min_activity

    def sample_model(self, sample_size=1000):
        I = np.eye(self.network_size)
        X, component_freqs = self.input_noise.draw(size=sample_size)
        noise = self.private_noise.draw(size=sample_size)
        v = np.linalg.inv(I - self.W) @ (self.b[:, None] + self.U @ X.T + noise.T)
        u = self.V @ v + self.intercept[:, None]
        targets = np.zeros(shape=(self.size[2], sample_size))
        component_freqs = np.cumsum(component_freqs)
        component_freqs = [0] + list(component_freqs)
        for i in range(len(component_freqs) - 1):
            targets[:, component_freqs[i]:component_freqs[i + 1]] = self.targets[i][:, None]
        return v, u, targets, component_freqs

    def plot_sample(self, sample_size=1000, outfile_name=None):
        v, u, targets, freqs = self.sample_model(sample_size=sample_size)
        plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
        for i in range(len(freqs) - 1):
            plt.scatter(u[0, freqs[i]:freqs[i + 1]], u[1, freqs[i]:freqs[i + 1]], s=8,
                        facecolor=target_colors[i], edgecolors='white', lw=0.2)
            plt.scatter(targets[0, freqs[i]:freqs[i + 1]], targets[1, freqs[i]:freqs[i + 1]], s=13,
                        facecolor=target_colors[i], edgecolors='black', lw=0.4, zorder=10)
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

    # -------------------------- Decoder-related member functions -------------------------- #
    def fit_decoder(self, intrinsic_manifold_dim=None, threshold=0.99, nb_trials=50):
        tot_var = self.compute_total_covariance()
        _, s, vt = np.linalg.svd(tot_var)
        '''
        plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']))
        plt.plot(100 * np.cumsum(s)/np.trace(tot_var))
        plt.xlabel("Ranked eigenvalues")
        plt.ylabel("Cumulative variance (%)")
        plt.tight_layout()
        plt.show()
        '''
        dim = self.dimensionality(threshold=threshold)
        print('Number of PCs for {} of total variance = {}'.format(threshold, dim))
        if intrinsic_manifold_dim is None:
            intrinsic_manifold_dim = dim

        # Projection matrix
        self.C = vt[:intrinsic_manifold_dim, :]

        if self.use_data_for_decoding:
            # Fit decoder using samples
            v, u, _, _ = self.sample_model(sample_size=self.nb_inputs*nb_trials)

            # Centering data
            self.sample_mean_v = np.mean(v, axis=1)
            v -= self.sample_mean_v[:, None]
            u -= np.mean(u, axis=1, keepdims=True)
            Cv = self.C @ v

            self.decoder = lm.LinearRegression(fit_intercept=False)
            self.decoder.fit(Cv.T, u.T)
            y = self.decoder.predict(Cv.T)
            mse = np.mean((y - u.T) ** 2)
            print(f'MSE = {mse}')

            # Define D and redefine V
            self.D = self.decoder.coef_
            print('Frobenius norm of V - DC = '
                  '{}'.format(np.linalg.norm(self.V - self.D @ self.C)**2 / (self.output_size * self.network_size)))
            self.V = self.D @ self.C
            #self.intercept = -self.V @ self.sample_mean_v
        else:
            # Find best fit by minimizing |V - DC|^2 wrt D
            def loss(dec_mat, corr_mat):
                #return np.linalg.norm(self.V - dec_mat @ self.C)**2 / (self.output_size * self.network_size)
                return np.trace((self.V - dec_mat @ self.C)@corr_mat@(self.V - dec_mat @ self.C).T)
            Var = self.compute_total_covariance()
            vbarvbarT = np.outer(self.get_mean_activity(), self.get_mean_activity())
            #D = 0.2*self.rng.normal(size=(self.output_size, intrinsic_manifold_dim))
            #for i in range(int(2e2)):
            #    if i % 10 == 0:
            #        print(loss(D, Var + vbarvbarT))
            #    D += 2e-3 * (self.V - D@self.C) @ (Var + vbarvbarT) @ self.C.T
            #self.D = D
            self.D = self.V @ (Var + vbarvbarT) @self.C.T @ np.linalg.inv(self.C @ (Var + vbarvbarT) @ self.C.T)
            self.V = self.D @ self.C
            #self.intercept = -self.V @ self.get_mean_activity()
        return intrinsic_manifold_dim, dim

    def select_perturb(self, intrinsic_manifold_dim, nb_om_permuted_units=30, nb_samples=int(1e4)):
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
                self.V = self.D @ self.C[perm, :]
                wm_losses[perm_counter] = self.loss_for_each_target()
                wm_permutations[perm_counter] = perm
                wm_total_losses.append(self.loss_function())
        else:  # comb over all possible permutations
            for perm_counter, perm in enumerate(itertools.permutations(range(intrinsic_manifold_dim))):
                self.V = self.D @ self.C[perm, :]
                wm_losses[perm_counter] = self.loss_for_each_target()
                wm_permutations[perm_counter] = perm
                wm_total_losses.append(self.loss_function())
        print(f"Median total loss for WM perturbation : {np.median(wm_total_losses)}")
        print(f"Median target-wise loss for WM perturbation : {np.median(wm_losses, axis=0)}")

        # OM
        self.V = self.D @ self.C
        mds = self.get_modulation_depth()
        sorted_indices = np.argsort(mds)
        indices_to_permute = sorted_indices[-nb_om_permuted_units:]

        for perm_counter in range(nb_samples_om):
            indices = copy.deepcopy(indices_to_permute)
            self.rng.shuffle(indices)
            indices_i = np.arange(self.network_size)
            indices_i[indices_to_permute] = indices
            self.V = self.D @ self.C[:, indices_i]
            om_losses[perm_counter] = self.loss_for_each_target()
            om_total_losses.append(self.loss_function())
            om_permutations[perm_counter] = indices_i

        # Return to original mapping
        self.V = self.D @ self.C

        # Compute median target-specific losses across all WM and OM permutations
        #median_per_target_loss = np.median(wm_losses, axis=0, keepdims=True)
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

    def wm_perturb(self, intrinsic_manifold_dim, nb_samples=200, target_loss=None):
        all_losses = []  # contains list of target-specific losses
        if target_loss is None:
            nb_samples += 1
            indices = []

            permutation_counter = 0
            for perm in itertools.permutations(range(intrinsic_manifold_dim)):
                permutation_counter += 1
                self.V = self.D @ self.C[perm, :]
                all_losses.append(self.loss_for_each_target())
                indices.append(perm)
                if permutation_counter == nb_samples:
                    break

            median_per_target_loss = np.median(all_losses)  # across all tested permutations and targets
            target_objective_vector = np.array([median_per_target_loss for i in range(self.nb_inputs)])
            normed_diff = np.linalg.norm(np.array(all_losses) - target_objective_vector[None, :], axis=1)
            selected_perm = np.argmin(normed_diff)
            self.V = self.D @ self.C[indices[selected_perm],:]  # apply perturbation
        else:
            loc_loss = self.loss_function()
            indices = np.arange(intrinsic_manifold_dim)
            while abs(loc_loss - target_loss) > 1e-2:
                self.rng.shuffle(indices)
                self.V = self.D @ self.C[indices, :]
                loc_loss = self.loss_function()
                print(f'test wm loss = {loc_loss}')

        if self.use_data_for_decoding:
            self.intercept = -self.V @ self.sample_mean_v
        else:
            self.intercept = -self.V @ self.get_mean_activity()
        return all_losses

    def om_perturb(self, nb_permuted_units=30, nb_samples=200, target_loss=None):
        mds = self.get_modulation_depth()
        sorted_indices = np.argsort(mds)
        indices_to_permute = sorted_indices[-nb_permuted_units:]
        all_losses = np.empty(nb_samples)

        if target_loss is None:
            all_indices = []

            for i in range(nb_samples):
                indices = copy.deepcopy(indices_to_permute)
                self.rng.shuffle(indices)
                indices_i = np.arange(self.network_size)
                indices_i[indices_to_permute] = indices
                self.V = self.D @ self.C[:,indices_i]
                loc_loss = self.loss_function()
                all_losses[i] = loc_loss
                all_indices.append(indices_i)

            mean_pert_index = np.argmin(np.abs(all_losses - np.mean(all_losses)))
            self.V = self.D @ self.C[:, all_indices[mean_pert_index]]
        else:
            all_losses = []
            loc_loss = self.loss_function()
            permutation_counter = 0
            while abs(loc_loss - target_loss) > 1e-2:
                indices = copy.deepcopy(indices_to_permute)
                self.rng.shuffle(indices)
                indices_i = np.arange(self.network_size)
                indices_i[indices_to_permute] = indices
                self.V = self.D @ self.C[:, indices_i]
                loc_loss = self.loss_function()
                all_losses.append(loc_loss)
                #print(f'test om loss = {loc_loss}')
                permutation_counter += 1
                if permutation_counter > 1e5:
                    print('Did not find an OM perturbation with the correct loss')
                    break
        plt.hist(all_losses)
        plt.show()
        if self.use_data_for_decoding:
            self.intercept = -self.V @ self.sample_mean_v
        else:
            self.intercept = -self.V @ self.get_mean_activity()
        return all_losses

    # -------------------------- Other functions -------------------------- #
    @staticmethod
    def dimensionality_(covariance_matrix, threshold=0.99):
        _, singular_values, _ = np.linalg.svd(covariance_matrix)  # using svg instead of eigvals because we want them properly ordered
        cum_var = np.cumsum(singular_values)
        return np.nonzero(cum_var > threshold * cum_var[-1])[0][0] + 1  # +1 because array elements start at zero

    def dimensionality(self, threshold=0.99):
        return self.dimensionality_(self.compute_total_covariance(), threshold=threshold)

    def dimensionality_input(self, threshold=0.99):
        return self.dimensionality_(self.compute_total_input_covariance(), threshold=threshold)

    @staticmethod
    def participation_ratio_(covariance_matrix):
        return (np.trace(covariance_matrix))**2/np.trace(covariance_matrix @ covariance_matrix)

    def participation_ratio(self):
        return self.participation_ratio_(self.compute_total_covariance())

    def participation_ratio_input(self):
        return self.participation_ratio_(self.compute_total_input_covariance())

    # -------------------------- Save and load functions -------------------------- #
    def save(self, folder_name):
        np.save(f'{folder_name}/W.npy', self.W)
        np.save(f'{folder_name}/U.npy', self.U)
        np.save(f'{folder_name}/b.npy', self.b)
        np.save(f'{folder_name}/V.npy', self.V)

    def load(self, folder_name):
        print('Loading saved parameters')
        self.W = np.load(f'{folder_name}/W.npy')
        self.U = np.load(f'{folder_name}/U.npy')
        self.b = np.load(f'{folder_name}/b.npy')
        self.V = np.load(f'{folder_name}/V.npy')


# -------------  njit-decorated helper functions for JIT compilation --------------- #
@njit
def _loss_function_jit(network_size, nb_inputs, U, W, V, b, input_noise_covs, input_noise_means, private_noise_cov, targets):
    I = np.eye(network_size)
    L_var, L_exp = 0, 0
    inv_I_minus_W = np.linalg.inv(I - W)

    for c in range(nb_inputs):
        Var_v_given_k = inv_I_minus_W @ (U @ input_noise_covs[c] @ U.T +
                                         private_noise_cov) @ inv_I_minus_W.T
        L_var += 0.5 * (1/nb_inputs) * np.trace(V @ Var_v_given_k @ V.T)

        Exp_v_given_k = inv_I_minus_W @ (b + U @ input_noise_means[c])
        L_exp += 0.5 * (1/nb_inputs) * np.linalg.norm(V @ Exp_v_given_k - targets[c]) ** 2

    return L_var + L_exp

@njit
def _loss_for_each_target_jit(network_size, nb_inputs, U, W, V, b, input_noise_covs, input_noise_means, private_noise_cov, targets):
    losses = np.zeros(nb_inputs)
    I = np.eye(network_size)
    inv_I_minus_W = np.linalg.inv(I - W)

    for c in range(nb_inputs):
        Var_v_given_k = inv_I_minus_W @ (U @ input_noise_covs[c] @ U.T +
                                         private_noise_cov) @ inv_I_minus_W.T
        losses[c] = 0.5 * (1/nb_inputs) * np.trace(V @ Var_v_given_k @ V.T)

        Exp_v_given_k = inv_I_minus_W @ (b + U @ input_noise_means[c])
        losses[c] += 0.5 * (1/nb_inputs) * np.linalg.norm(V @ Exp_v_given_k - targets[c]) ** 2

    return losses