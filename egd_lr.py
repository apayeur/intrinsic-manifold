"""
Description:
-----------
To examine the effect of the learning rate on WM and OM adaptation.
"""
from toy_model import ToyNetwork
import numpy as np
import copy
import os

def main():
    tag = "effect-of-lr"  # identification of this experiment, for bookkeeping
    save_dir = f"data/egd/{tag}"
    save_dir_results = f"results/egd/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    # Parameters
    size = (6, 100, 2)
    input_noise_intensity = 0e-4
    private_noise_intensity = 1e-3
    intrinsic_manifold_dim = 6  # 6
    lr = 1e-3
    lr_init = (0, lr, 0)
    lr_decoder = (0, lr, 0)

    lr_adapts = [lr, lr/10, lr/25]

    nb_iter = int(1e3)
    seeds = np.arange(3, dtype=int)
    relearn_after_decoder_fitting = False

    # Total losses
    loss_init = np.empty(shape=(len(lr_adapts), len(seeds), nb_iter))
    loss_compare_lr = {'WM': [[np.empty(int(1e3/lr_adapts[lr_i]*lr)) for seed_id in range(len(seeds))] for lr_i in range(len(lr_adapts))],
                       'OM': [[np.empty(int(1e3/lr_adapts[lr_i]*lr)) for seed_id in range(len(seeds))] for lr_i in range(len(lr_adapts))]}


    for lr_i, learning_rate in enumerate(lr_adapts):
        lr_adapt = (0, learning_rate, 0)
        nb_iter_adapt = int(5e2/learning_rate*lr)

        for seed_id, seed in enumerate(seeds):
            print(f'\n|==================================== Seed {seed} =====================================|')
            print('\n|-------------------------------- Initial training --------------------------------|')
            net0 = ToyNetwork('initial', size=size,
                              input_noise_intensity=input_noise_intensity,
                              private_noise_intensity=private_noise_intensity,
                              input_subspace_dim=0, nb_inputs=size[0],
                              use_data_for_decoding=False,
                              global_mean_input_is_zero=False,
                              initialization_type='random',
                              rng_seed=seed)
            l, _, _, _, _, _, _ = net0.train(lr=lr_init, nb_iter=nb_iter)
            loss_init[seed_id] = l['total']
            if seed_id == 0:
                net0.plot_sample(sample_size=1000,
                                 outfile_name=f"{save_dir_results}/SampleEndInitialTraining_lr{learning_rate}.png")


            print('\n|-------------------------------- Fit decoder --------------------------------|')
            net1 = copy.deepcopy(net0)
            intrinsic_manifold_dim, _ = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim, threshold=0.99)
            net1.network_name = 'fitted'
            if seed_id == 0:
                net1.plot_sample(sample_size=1000,
                                 outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting_lr{learning_rate}.png")


            print('\n|-------------------------------- Re-training with decoder --------------------------------|')
            net2 = copy.deepcopy(net1)
            net2.network_name = 'retrained_after_fitted'
            if relearn_after_decoder_fitting:
                loss_decoder_retraining, _, _, _, _, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)
                if seed_id == 0:
                    net2.plot_sample(sample_size=1000,
                                     outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder_lr{learning_rate}.png")


            print('\n|-------------------------------- Select perturbations --------------------------------|')
            selected_wm, selected_om, wm_t_l, om_t_l = \
                net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1])
            if seed_id == 0:
                wm_total_losses, om_total_losses = wm_t_l, om_t_l


            print('\n|-------------------------------- WM perturbation --------------------------------|')
            net_wm = copy.deepcopy(net2)
            net_wm.network_name = 'wm'
            net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]
            if seed_id == 0:
                net_wm.plot_sample(sample_size=1000,
                                   outfile_name=f"{save_dir_results}/SampleWMBeforeLearning_lr{learning_rate}.png")

            l, norm, a_min, a_max, nve, _, A_tmp = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

            if seed_id == 0:
                net_wm.plot_sample(sample_size=1000,
                                   outfile_name=f"{save_dir_results}/SampleWMAfterLearning_lr{learning_rate}.png")

            loss_compare_lr['WM'][lr_i][seed_id] = l['total']



            print('\n|-------------------------------- OM perturbation --------------------------------|')
            net_om = copy.deepcopy(net2)
            net_om.network_name = 'om'
            net_om.V = net_om.D @ net_om.C[:, selected_om]

            if seed_id == 0:
                net_om.plot_sample(sample_size=1000,
                                   outfile_name=f"{save_dir_results}/SampleOMBeforeLearning_lr{learning_rate}.png")

            l, norm, a_min, a_max, nve, R_seed, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

            if seed_id == 0:
                net_om.plot_sample(sample_size=1000,
                                   outfile_name=f"{save_dir_results}/SampleOMAfterLearning_lr{learning_rate}.png")

            loss_compare_lr['OM'][lr_i][seed_id] = l['total']


    # Save parameters
    param_dict = {'size': size,
                  'input_noise_intensity': input_noise_intensity,
                  'private_noise_intensity': private_noise_intensity,
                  'lr': lr, 'lr_init': lr_init, 'lr_decoder': lr_decoder, 'lr_adapts': lr_adapts,
                  'nb_iter': nb_iter,
                  'nb_iter_adapts': np.array(5e2/np.array(lr_adapts)*lr, dtype=int),
                  'intrinsic_manifold_dim': intrinsic_manifold_dim,
                  'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
    np.save(f"{save_dir}/params", param_dict)

    # Save performances
    loss_dict = {'loss_init': loss_init,
                 'loss': loss_compare_lr}
    np.save(f"{save_dir}/loss", loss_dict)


if __name__ == '__main__':
    main()
