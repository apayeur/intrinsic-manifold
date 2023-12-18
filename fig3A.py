from toy_model import ToyNetwork
import numpy as np
import copy
import os
from scipy.linalg import subspace_angles


def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)  # (input size, recurrent size, output size)
    input_noise_intensity = 0e-4  # set to zero for 1-of-K encoding
    private_noise_intensity = 1e-2
    intrinsic_manifold_dim = 6  # dimension of manifold for control (M)
    lr_init = (0, 1e-2, 0)  # learning rate for initial training
    lr_decoder = (0, 5e-3, 0)  # not used wen `relearn_after_decoder_fitting = False` below
    lr = 0.001  # learning rate during adaptation
    nb_iter = int(5e2)  # nb of gradient iteration during initial training
    nb_iter_adapt = int(5e2)  # nb of gradient iteration during adaptation
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False
    exponent_W = 0.55  # W_0 ~ N(0, 1/N^exponent_W)

    lr_adapt = (0, lr, 0)
    tag = f"fig3-exponent_W{exponent_W}-lr{lr_adapt[1]}-lrstudy"  # identification of this experiment, for bookkeeping
    save_dir = f"data/egd/{tag}"
    save_dir_results = f"results/egd/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    # DEFINITION OF DATA CONTAINERS FOR SAVED DATA
    # Max absolute eigenvalues during training
    max_eigvals = {'initial': np.zeros(shape=(len(seeds), nb_iter)),
                   'WM': np.zeros(shape=(len(seeds), nb_iter_adapt)),
                   'OM': np.zeros(shape=(len(seeds), nb_iter_adapt))}

    # Simulation
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

        _, _, _, _, _, _, _, _, _, _, max_eigvals['initial'][seed_id], _  = net0.train(lr=lr_init, nb_iter=nb_iter)

        print('\n|-------------------------------- Fit decoder --------------------------------|')
        net1 = copy.deepcopy(net0)
        intrinsic_manifold_dim, _ = net1.fit_decoder(
            intrinsic_manifold_dim=intrinsic_manifold_dim,
            threshold=0.95)
        net1.network_name = 'fitted'
        V_0 = copy.deepcopy(net1.V)

        print('\n|-------------------------------- Re-training with decoder --------------------------------|')
        net2 = copy.deepcopy(net1)
        net2.network_name = 'retrained_after_fitted'
        if relearn_after_decoder_fitting:
            _, _, _, _, _, _, _, _, _, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter // 10)

        print('\n|-------------------------------- Select perturbations --------------------------------|')
        selected_wm, selected_om, wm_t_l, om_t_l = \
            net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1])

        print('\n|-------------------------------- WM perturbation --------------------------------|')
        net_wm = copy.deepcopy(net2)
        net_wm.network_name = 'wm'
        net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]

        l, norm, a_min, a_max, nve, _, A_tmp, f_seed, _, _, max_eigvals['WM'][seed_id], _ \
            = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

        print('\n|-------------------------------- OM perturbation --------------------------------|')
        net_om = copy.deepcopy(net2)
        net_om.network_name = 'om'
        net_om.V = net_om.D @ net_om.C[:, selected_om]

        l, norm, a_min, a_max, nve, R_seed, _, f_seed, rel_proj_var_OM_seed,_, max_eigvals['OM'][seed_id], _ \
            = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

    # Save parameters
    param_dict = {'size': size,
                  'nb_seeds': len(seeds),
                  'input_noise_intensity': input_noise_intensity,
                  'private_noise_intensity': private_noise_intensity,
                  'lr_init': lr_init, 'lr_decoder': lr_decoder, 'lr_adapt': lr_adapt,
                  'nb_iter': nb_iter,
                  'nb_iter_adapt': nb_iter_adapt,
                  'intrinsic_manifold_dim': intrinsic_manifold_dim,
                  'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
    np.save(f"{save_dir}/params", param_dict)

    # Save eigvals during adaptation
    np.save(f"{save_dir}/max_eigvals", max_eigvals)


if __name__ == '__main__':
    main()
