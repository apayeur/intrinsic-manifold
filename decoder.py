import numpy as np
from sklearn.linear_model import LinearRegression
from math import factorial
import itertools
import copy


class Decoder:
    def __init__(self, readouts_units, network_size, V):
        assert V.shape[1] == len(readouts_units), f"Nb of readout units {len(readouts_units)} not match with nb of cols of V {V.shape[1]}"
        self.network_size = network_size
        self.nb_readouts = V.shape[1]
        self.readout_ids = readouts_units
        self.V = V
        self.C = None  # projection matrix, shape = (intrinsic_manifold_dim, self.network_size)
        self.D = None  # decoding matrix, shape = (self.output_size, intrinsic_manifold_dim)
        self.R = None  # readout matrix
        self.update_record_matrix(readouts_units)
        self.intercept = np.zeros(V.shape[0])  # intercept of the decoder
        self.inv_Sv = np.eye(network_size)  # initialization of z-scoring matrix
        self.inv_Sz = None  # matrix for z-scoring PCs
        self.ma_0 = np.zeros(network_size)  # mean activity after initial training (used for z-scoring)

    def __call__(self, activity):
        return self.V @ self.R @ (activity - self.ma_0) + self.intercept

    def T(self):
        return (self.V @ self.R).T

    def update_record_matrix(self, recorded_units_ids):
        self.readout_ids = recorded_units_ids
        self.nb_readouts = len(recorded_units_ids)
        self.R = np.zeros((self.nb_readouts, self.network_size))
        for i in range(self.nb_readouts):
            self.R[i, recorded_units_ids[i]] = 1.

    def fit(self, X, y,
            intrinsic_manifold_dim=None, threshold=0.95, fit_intercept=False, do_z_score=False):
        # Check for any dead units
        alive_units = np.nonzero(np.diag(net.activity_covariance()) > 1e-5)[0]
        self.update_record_matrix(alive_units)
        if do_z_score:
            self.inv_Sv = np.diag(np.sqrt(np.diag(net.activity_covariance())) ** -1)
            self.ma_0 = net.mean_activity()

        # Construct projection
        tot_var = net.activity_correlation_matrix() if do_z_score else net.activity_covariance()
        w, v = np.linalg.eig(tot_var)
        ranked_eig_indices = np.argsort(w)[::-1]  # need to order eigensystem
        vt = v[:, ranked_eig_indices].T

        dim = net.dimensionality(threshold=threshold)
        print(f"Number of PCs for {threshold} of total variance = {dim}")
        if intrinsic_manifold_dim is None:
            intrinsic_manifold_dim = dim

        self.C = vt[:intrinsic_manifold_dim, :]  # projection matrix
        self.inv_Sz = np.diag(np.sqrt(np.diag(self.C @ tot_var @ self.C.T)) ** -1) if do_z_score else np.eye(
            intrinsic_manifold_dim)
        C_loc = self.inv_Sz @ self.C @ self.inv_Sv

        # Fit
        #if not fit_intercept and not do_z_score:
        #    # Exact solution
        #    vbarvbarT = np.outer(net.mean_activity(), net.mean_activity())
        #    self.D = self.V @ (tot_var + vbarvbarT) @ C_loc.T @ np.linalg.inv(C_loc @ (tot_var + vbarvbarT) @ C_loc.T)
        #else:
        lr = LinearRegression(fit_intercept=fit_intercept)
        ca = np.asarray(net.conditioned_activities())
        lr.fit((ca - self.ma_0) @ C_loc.T, ca @ self.V.T)
        self.intercept = lr.intercept_ if fit_intercept else np.zeros(net.output_size)
        self.D = lr.coef_
        print("Fit R2:", lr.score((ca - self.ma_0) @ C_loc.T, ca @ self.V.T))
        print("D =", self.D)
        print("Intercept:", self.intercept)
        self.V = self.D @ C_loc
        return intrinsic_manifold_dim, dim

    def select_perturb(self, net, intrinsic_manifold_dim, nb_om_permuted_units=30, nb_samples=int(1e3),
                       om_select_method='original'):
        if om_select_method != 'original' and om_select_method != 'modified':
            raise ValueError(f"OM selection method was {om_select_method} but must be either `original` or `modified`.")

        """Select the WM and OM perturbations"""
        nb_samples_wm = factorial(intrinsic_manifold_dim) if intrinsic_manifold_dim <= 8 else nb_samples
        nb_samples_om = max(nb_samples, nb_samples_wm) if om_select_method == 'original' else factorial(intrinsic_manifold_dim)

        wm_permutations = np.empty(shape=(nb_samples_wm, intrinsic_manifold_dim))
        om_permutations = np.empty(shape=(nb_samples_om, self.nb_readouts))
        wm_losses = np.empty(shape=(nb_samples_wm, net.nb_inputs))
        om_losses = np.empty(shape=(nb_samples_om, net.nb_inputs))
        wm_total_losses = []
        om_total_losses = []

        # WM
        if intrinsic_manifold_dim > 7:
            for perm_counter in range(nb_samples_wm):
                indices = np.arange(intrinsic_manifold_dim)
                net.rng.shuffle(indices)
                perm = indices
                self.apply_wm_perturb(perm)
                wm_losses[perm_counter] = net.loss_for_each_target()
                wm_permutations[perm_counter] = perm
                wm_total_losses.append(net.task_loss())
        else:  # comb over all possible permutations
            for perm_counter, perm in enumerate(itertools.permutations(range(intrinsic_manifold_dim))):
                self.apply_wm_perturb(perm)
                wm_losses[perm_counter] = net.loss_for_each_target()
                wm_permutations[perm_counter] = perm
                wm_total_losses.append(self.task_loss())
        print(f"Median total loss for WM perturbation : {np.median(wm_total_losses)}")
        print(f"Median target-wise loss for WM perturbation : {np.median(wm_losses, axis=0)}")

        # OM
        self.V = self.D @ self.inv_Sz @ self.decoder.C @ self.inv_Sv
        mds = self.get_modulation_depth()
        sorted_indices = np.argsort(mds)

        if om_select_method == 'original':
            indices_to_permute = sorted_indices[-nb_om_permuted_units:]

            for perm_counter in range(nb_samples_om):
                indices = copy.deepcopy(indices_to_permute)
                self.rng.shuffle(indices)
                indices_i = np.arange(self.network_size)
                indices_i[indices_to_permute] = indices

                self.V = self.D @ self.inv_Sz @ self.decoder.C[:, indices_i] @ self.inv_Sv
                om_losses[perm_counter] = self.loss_for_each_target()
                om_total_losses.append(self.task_loss())
                om_permutations[perm_counter] = indices_i
        else:
            nb_dead_units = np.sum(np.asarray(mds) < 1e-6)
            print('Number of dead units, from modulation depth:', nb_dead_units)
            nb_blocks = intrinsic_manifold_dim
            nb_remaining_units = (self.network_size - nb_dead_units) % nb_blocks

            # TODO: There is a less meathead way to do the following.
            if nb_remaining_units == 0 and nb_dead_units == 0:
                nb_units_per_blocks = self.network_size // (nb_blocks + 1)
                nb_remaining_units = nb_units_per_blocks + self.network_size % (nb_blocks + 1)
            if nb_remaining_units > 0 and nb_dead_units == 0:
                nb_units_per_blocks = self.network_size // nb_blocks
                nb_remaining_units = self.network_size % nb_blocks
            if nb_remaining_units == 0 and nb_dead_units > 0:
                nb_remaining_units = nb_dead_units
                nb_units_per_blocks = (self.network_size - nb_remaining_units) % nb_blocks
            if nb_remaining_units > 0 and nb_dead_units > 0:
                nb_remaining_units += nb_dead_units
                nb_units_per_blocks = (self.network_size - nb_remaining_units) % nb_blocks
            if nb_remaining_units == 0:
                raise ValueError(f"0 remaining units, because {self.network_size}%{nb_blocks} = 0.")
            partial_indices = sorted_indices[:-nb_remaining_units] if nb_remaining_units > 0 else sorted_indices
            blocks = [partial_indices[i*nb_units_per_blocks:(i+1)*nb_units_per_blocks] for i in range(nb_blocks)]
            assert len(blocks) == intrinsic_manifold_dim, "len(block) not equal to intrinsic manifold dimension"
            s = 0
            for b in range(nb_blocks):
                s += len(blocks[b])
            s += nb_remaining_units
            assert s == self.network_size, f"s = {s} different for network size {self.network_size}"
            for perm_counter, perm in enumerate(itertools.permutations(range(intrinsic_manifold_dim))):
                indices = np.hstack((*[blocks[i] for i in perm], sorted_indices[-nb_remaining_units:]))
                self.V = self.D @ self.inv_Sz @ self.decoder.C[:, indices] @ self.inv_Sv
                om_losses[perm_counter] = self.loss_for_each_target()
                om_total_losses.append(self.task_loss())
                om_permutations[perm_counter] = indices

        print(f"\nMedian total loss for OM perturbation : {np.median(om_total_losses)}")
        print(f"Median target-wise loss for OM perturbation : {np.median(om_losses, axis=0)}")

        # Return to original mapping
        self.V = self.D @ self.inv_Sz @ self.decoder.C @ self.inv_Sv

        # Compute median target-specific losses across all WM and OM permutations
        median_per_target_loss = np.median(np.vstack((wm_losses, om_losses)), axis=0, keepdims=True)
        print(f'\nCombined median per-target loss = {median_per_target_loss}')

        # Find WM and OM permutations closest to median WM perturbations
        normed_diff = np.linalg.norm(wm_losses - median_per_target_loss, axis=1)
        selected_wm = wm_permutations[np.argmin(normed_diff)]
        self.selected_permutation_WM = np.asarray(selected_wm, dtype=int)

        normed_diff = np.linalg.norm(om_losses - median_per_target_loss, axis=1)
        selected_om = om_permutations[np.argmin(normed_diff)]
        self.selected_permutation_OM = np.asarray(selected_om, dtype=int)
        return self.selected_permutation_WM, self.selected_permutation_OM, wm_total_losses, om_total_losses

    def apply_wm_perturb(self, selected_wm):
        self.V = self.D[:, selected_wm] @ self.inv_Sz @ self.decoder.C @ self.inv_Sv

    def apply_om_perturb(self, selected_om):
        self.V = self.D @ self.inv_Sz @ self.decoder.C[:, selected_om] @ self.inv_Sv

