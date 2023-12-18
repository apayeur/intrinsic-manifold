from toy_model import ToyNetwork
import numpy as np
import copy
import os
from scipy.linalg import subspace_angles


def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)              # (input size, recurrent size, output size)
    input_noise_intensity = 0e-4    # set to zero for 1-of-K encoding
    private_noise_intensity = 1e-2
    intrinsic_manifold_dim = 6      # dimension of manifold for control (M)
    lr_init = (0, 1e-2, 0)          # learning rate for initial training
    lr_decoder = (0, 5e-3, 0)       # not used wen `relearn_after_decoder_fitting = False` below
    lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]  # learning rate during adaptation
    nb_iter = int(5e2)              # nb of gradient iteration during initial training
    nb_iter_adapt = int(1e3)        # nb of gradient iteration during adaptation
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False
    exponent_W = 0.55        # W_0 ~ N(0, 1/N^exponent_W)

    for lr in lrs:
        lr_adapt = (0, lr, 0)
        tag = f"fig3-exponent_W{exponent_W}-lr{lr_adapt[1]}-lrstudy"  # identification of this experiment, for bookkeeping
        save_dir = f"data/egd/{tag}"
        save_dir_results = f"results/egd/{tag}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir_results):
            os.makedirs(save_dir_results)

        # DEFINITION OF DATA CONTAINERS FOR SAVED DATA
        # Eigenvalues (max abs eigval at the end of training)
        eigenvals_0 = []
        eigenvals_init = []
        eigenvals_after_WMP = []
        eigenvals_after_OMP = []

        # Losses
        loss = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

        for seed_id, seed in enumerate(seeds):
            print(f'\n|==================================== Seed {seed} =====================================|')
            print('\n|-------------------------------- Initial training --------------------------------|')
            net0 = ToyNetwork('initial', size=size,
                              input_noise_intensity=input_noise_intensity,
                              private_noise_intensity=private_noise_intensity,
                              input_subspace_dim=0, nb_inputs=size[0],
                              use_data_for_decoding=False,
                              global_mean_input_is_zero=False,
                              initialization_type='random', exponent_W=exponent_W,
                              rng_seed=seed)

            eigenvals_0.append(np.max(np.abs(np.linalg.eigvals(net0.W))))

            _, _, _, _, _, _, _, _, _, _, _, _ = net0.train(lr=lr_init, nb_iter=nb_iter)

            eigenvals_init.append(np.max(np.abs(np.linalg.eigvals(net0.W))))

            print('\n|-------------------------------- Fit decoder --------------------------------|')
            net1 = copy.deepcopy(net0)
            intrinsic_manifold_dim, _ = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim, threshold=0.95)
            net1.network_name = 'fitted'
            V_0 = copy.deepcopy(net1.V)

            print('\n|-------------------------------- Re-training with decoder --------------------------------|')
            net2 = copy.deepcopy(net1)
            net2.network_name = 'retrained_after_fitted'
            if relearn_after_decoder_fitting:
                loss_decoder_retraining, _, _, _, _, _, _,_, _, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)

            print('\n|-------------------------------- Select perturbations --------------------------------|')
            selected_wm, selected_om, wm_t_l, om_t_l = \
                net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1])

            print('\n|-------------------------------- WM perturbation --------------------------------|')
            net_wm = copy.deepcopy(net2)
            net_wm.network_name = 'wm'
            net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]

            l, _, _, _, _, _, _, _, _, _, _, _ = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

            if len(l['total']) == nb_iter_adapt:
                loss['WM'][seed_id] = l['total']

            eigval_after_WM = np.max(np.abs(np.linalg.eigvals(net_wm.W)))
            print(f"Max eigenvalue of W: {eigval_after_WM}")
            eigenvals_after_WMP.append(eigval_after_WM)

            print('\n|-------------------------------- OM perturbation --------------------------------|')
            net_om = copy.deepcopy(net2)
            net_om.network_name = 'om'
            net_om.V = net_om.D @ net_om.C[:, selected_om]

            l, _, _, _, _, _, _, _, _, _, _, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

            if len(l['total']) == nb_iter_adapt:
                loss['OM'][seed_id] = l['total']

            eigval_after_OM = np.max(np.abs(np.linalg.eigvals(net_om.W)))
            print(f"Max eigenvalue of W: {eigval_after_OM}")
            eigenvals_after_OMP.append(eigval_after_OM)

        # Save parameters
        param_dict = {'size': size,
                      'nb_seeds': len(seeds),
                      'input_noise_intensity': input_noise_intensity,
                      'private_noise_intensity': private_noise_intensity,
                      'lr_init': lr_init, 'lr_decoder': lr_decoder,'lr_adapt': lr_adapt,
                      'nb_iter': nb_iter,
                      'nb_iter_adapt': nb_iter_adapt,
                      'intrinsic_manifold_dim': intrinsic_manifold_dim,
                      'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
        np.save(f"{save_dir}/params", param_dict)

        # Save performances
        loss_dict = {'loss': loss}
        np.save(f"{save_dir}/loss", loss_dict)

        # Save eigenvalues
        np.save(f"{save_dir}/eigenvals_0", eigenvals_0)
        np.save(f"{save_dir}/eigenvals_init", eigenvals_init)
        np.save(f"{save_dir}/eigenvals_after_WMP", eigenvals_after_WMP)
        np.save(f"{save_dir}/eigenvals_after_OMP", eigenvals_after_OMP)


if __name__ == '__main__':
    main()
