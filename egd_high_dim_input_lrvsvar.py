from toy_model import ToyNetwork
import numpy as np
import matplotlib.pyplot as plt
from utils import units_convert
import os

"""
Description: Studying the link between dimensionality and learning during initial training.
"""

def main():
    tag = "dim_vs_lr_th90"  # identification of this experiment, for bookkeeping
    save_dir = f"data/egd-high-dim-input/{tag}"
    save_dir_results = f"results/egd-high-dim-input/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    # Parameters
    size = (200, 100, 2)
    input_noise_intensity = 1e-3
    private_noise_intensity = 1e-4
    nb_inputs = 6
    input_subspace_dim = size[0]//10
    lrs = [5e-3, 10e-3, 25e-3, 50e-3]
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False
    threshold = 0.90

    # Total losses
    loss_init = {lr: {seed: [] for seed in seeds} for lr in lrs}

    # Initial manifold dimension
    real_dims = {lr: [] for lr in lrs}

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
                              initialization_type='random',
                              rng_seed=seed)
            l, _, _, _, _, _, _ = net0.train(lr=lr_init, stopping_crit=3e-3)
            loss_init[lr][seed] = l['total']

            real_dims[lr].append(net0.dimensionality(threshold=threshold))

    print(real_dims)

    # Save parameters
    param_dict = {'size': size,
                  'input_noise_intensity': input_noise_intensity,
                  'private_noise_intensity': private_noise_intensity,
                  'lrs': lrs,
                  'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
    np.save(f"{save_dir}/params", param_dict)

    # Save performances
    loss_dict = {'loss_init': loss_init}
    np.save(f"{save_dir}/loss", loss_dict)

    # Save initial manifold dimension
    np.save(f"{save_dir}/real_dims", real_dims)



if __name__ == '__main__':
    main()
