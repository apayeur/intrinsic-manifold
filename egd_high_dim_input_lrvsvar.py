from toy_model import ToyNetwork
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
"""
Description: Studying the link between dimensionality and learning during initial training.
"""

def main():
    tag = "dim_vs_lr_higher_private_noise"  # identification of this experiment, for bookkeeping
    save_dir = f"data/egd-high-dim-input/{tag}"
    save_dir_results = f"results/egd-high-dim-input/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    # Parameters
    size = (200, 100, 2)  # (200, 100, 2)
    input_noise_intensity = 5e-4  # 1e-3
    private_noise_intensity = 5e-4  # 1e-4
    nb_inputs = 6
    input_subspace_dim = size[0]//10
    lrs = [2e-3, 5e-3, 10e-3, 25e-3, 50e-3]  #[5e-5, 1e-4, 5.e-4, 1e-3]
    seeds = np.arange(10, dtype=int)
    threshold = 0.95

    # Total losses
    loss_init = {lr: {seed: [] for seed in seeds} for lr in lrs}

    # Initial manifold dimension
    real_dims = {lr: [] for lr in lrs}
    real_dims_input = {lr: [] for lr in lrs}

    participation_ratios = {lr: [] for lr in lrs}
    participation_ratios_input = {lr: [] for lr in lrs}

    eigenvalues_spectra_W = {lr: [] for lr in lrs}
    singular_spectra_U = {lr: [] for lr in lrs}


    for lr in lrs:
        lr_init = (lr, lr, 0)
        print(f'\n|==================================== Learning rate {lr} =====================================|')
        for seed_id, seed in enumerate(seeds):
            print(f'\n|==================================== Seed {seed} =====================================|')
            net0 = ToyNetwork('initial', size=size,
                              input_noise_intensity=input_noise_intensity,
                              private_noise_intensity=private_noise_intensity,
                              input_subspace_dim=input_subspace_dim, nb_inputs=nb_inputs,
                              use_data_for_decoding=False,
                              global_mean_input_is_zero=False,
                              orthogonalize_input_means=False,
                              initialization_type='random',
                              rng_seed=seed)
            #net0.plot_sample(1000)
            l, _, _, _, _, _, _ = net0.train(lr=lr_init, stopping_crit=3e-3)
            print("Max abs. eigval:")
            print(np.max(np.abs(np.linalg.eigvals(net0.W))))
            loss_init[lr][seed] = l['total']

            participation_ratios[lr].append(net0.participation_ratio())
            participation_ratios_input[lr].append(net0.participation_ratio_input())

            real_dims_input[lr].append(net0.dimensionality_input(threshold=threshold))
            real_dims[lr].append(net0.dimensionality(threshold=threshold))

            eigenvalues_spectra_W[lr].append(np.linalg.eigvals(net0.W))
            _, s, _ = np.linalg.svd(net0.U)
            singular_spectra_U[lr].append(s)

    # Save parameters
    param_dict = {'size': size,
                  'input_noise_intensity': input_noise_intensity,
                  'private_noise_intensity': private_noise_intensity,
                  'lrs': lrs,
                  'threshold': threshold}
    np.save(f"{save_dir}/params", param_dict)

    # Save performances
    loss_dict = {'loss_init': loss_init}
    np.save(f"{save_dir}/loss", loss_dict)

    # Save initial manifold dimension
    np.save(f"{save_dir}/real_dims", real_dims)
    np.save(f"{save_dir}/real_dims_input", real_dims_input)

    # Save participation ratios
    np.save(f"{save_dir}/participation_ratios", participation_ratios)
    np.save(f"{save_dir}/participation_ratios_input", participation_ratios_input)

    # Save eigenvalues
    np.save(f"{save_dir}/eigenvalsW", eigenvalues_spectra_W)
    np.save(f"{save_dir}/singvalsU", singular_spectra_U)


if __name__ == '__main__':
    main()
