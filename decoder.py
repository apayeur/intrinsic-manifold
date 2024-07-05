import numpy as np
from sklearn.linear_model import LinearRegression
from math import factorial
import itertools
import copy


class Decoder:
    def __init__(self, readout_units, network_size, V):
        assert V.shape[1] == len(readout_units), f"Nb of readout units {len(readout_units)} not match with nb of cols of V {V.shape[1]}"
        self.network_size = network_size
        self.nb_readouts = len(readout_units)
        self.readout_ids = readout_units
        self.output_dim = V.shape[0]
        self.V = V
        self.C = None  # projection matrix, shape = (intrinsic_manifold_dim, nb_readouts)
        self.D = None  # decoding matrix, shape = (self.output_size, intrinsic_manifold_dim)
        self.R = None  # record matrix, shape = (nb_readout, network_size)
        self.update_record_matrix(readout_units)
        self.intercept = np.zeros(V.shape[0])  # intercept of the decoder
        self.inv_Sv = None  # z-scoring matrix
        self.inv_Sz = None  # matrix for z-scoring PCs
        self.ma_0 = np.zeros(network_size)  # mean activity after initial training (used for z-scoring)

    def __call__(self, activity):
        return self.V @ (activity[self.readout_ids] - self.ma_0[self.readout_ids]) + self.intercept

    def transpose(self):
        return (self.V @ self.R).T

    def VR(self):
        return self.V @ self.R

    def update_record_matrix(self, readouts_units_ids):
        old_readouts_units_ids = copy.copy(self.readout_ids)
        self.readout_ids = readouts_units_ids
        self.nb_readouts = len(self.readout_ids)
        self.R = np.zeros((self.nb_readouts, self.network_size))

        for i, r_id in enumerate(self.readout_ids):
            self.R[i, r_id] = 1.

        if self.V.shape[1] != self.nb_readouts:
            new_V = np.zeros((self.output_dim, self.nb_readouts))
            total_new_ids = 0
            for i, old_id in enumerate(old_readouts_units_ids):
                if old_id in self.readout_ids:
                    new_V[:, total_new_ids] = self.V[:, i]
                    total_new_ids += 1
            self.V = new_V

    def fit(self, activities, network_covariance, intrinsic_manifold_dim=None, fit_intercept=False, do_z_score=False):
        R_0 = copy.copy(self.R)
        V_0 = copy.copy(self.V)

        indices_alive_units = np.nonzero(np.diag(network_covariance)[self.readout_ids] > 1e-6)[0]
        self.update_record_matrix(self.readout_ids[indices_alive_units])
        print("Number of readouts at fitting time :", self.nb_readouts)

        readout_covariance = self.R @ network_covariance @ self.R.T

        # If z-scoring, use matrix of correlation coefficient:
        if do_z_score:
            self.inv_Sv = np.diag(np.sqrt(np.diag(readout_covariance)) ** -1)
            self.ma_0 = np.mean(activities, axis=0)
            readout_covariance = self.inv_Sv @ readout_covariance @ self.inv_Sv

            if np.linalg.norm(np.diag(readout_covariance) - np.ones(self.nb_readouts)) > 1e-6:
                raise ValueError(f"Correlation coeff matrix not properly constructed. C = {readout_covariance}")
        else:
            self.inv_Sv = np.eye(self.nb_readouts)

        # Construct projection
        w, v = np.linalg.eigh(readout_covariance)
        ranked_eig_indices = np.argsort(w)[::-1]  # need to order eigen system
        vt = v[:, ranked_eig_indices].T

        cum_var = np.cumsum(w[ranked_eig_indices])
        assert w[ranked_eig_indices][0] > w[ranked_eig_indices][1]
        dim = np.nonzero(cum_var > 0.95 * cum_var[-1])[0][0] + 1  # +1 because how arrays are indexed
        print(f"Number of dimension for 95% of total variance = {dim}")
        if intrinsic_manifold_dim is None:
            intrinsic_manifold_dim = dim
            print(f"Dimension of intrinsic manifold was set to {dim}")

        self.C = vt[:intrinsic_manifold_dim, :]  # projection matrix
        self.inv_Sz = np.diag(np.sqrt(w[ranked_eig_indices][:intrinsic_manifold_dim]) ** -1) if do_z_score else np.eye(
            intrinsic_manifold_dim)
        print("Inverse of Sz:", np.diag(self.inv_Sz))
        C_loc = self.inv_Sz @ self.C @ self.inv_Sv

        # Fit
        #if not fit_intercept and not do_z_score:
        #    # Exact solution
        #    vbarvbarT = np.outer(net.mean_activity(), net.mean_activity())
        #    self.D = self.V @ (tot_var + vbarvbarT) @ C_loc.T @ np.linalg.inv(C_loc @ (tot_var + vbarvbarT) @ C_loc.T)
        #else:
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit((activities - self.ma_0) @ self.R.T @ C_loc.T, activities @ R_0.T @ V_0.T)
        self.intercept = lr.intercept_ if fit_intercept else np.zeros(self.output_dim)
        self.D = lr.coef_
        print("Fit R2:", lr.score((activities - self.ma_0) @ self.R.T @ C_loc.T, activities @ R_0.T @ V_0.T))
        print("D =", self.D)
        print("Intercept:", self.intercept)
        self.V = self.D @ C_loc

        # Checks
        #print("\nNorm of difference between intercept and V_0 R_0 r_0",
        #      np.linalg.norm(self.intercept - V_0 @ R_0 @ self.ma_0))
        #print("Norm of difference between V R and V_0 R_0",
        #      np.linalg.norm(self.V @ self.R - V_0 @ R_0))

        return intrinsic_manifold_dim, dim

    def apply_perturb(self, selected_permutation, perturbation_type):
        if perturbation_type != 'WM' and perturbation_type != 'OM':
            raise ValueError(f"Perturbation type should be 'WM' or 'OM', but was {perturbation_type}.")
        if perturbation_type == 'WM':
            self.V = self.D[:, selected_permutation] @ self.inv_Sz @ self.C @ self.inv_Sv
        else:
            self.V = self.D @ self.inv_Sz @ self.C[:, selected_permutation] @ self.inv_Sv

    def restore_intuitive(self):
        self.V = self.D @ self.inv_Sz @ self.C @ self.inv_Sv

