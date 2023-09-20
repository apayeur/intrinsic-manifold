"""
Description:
-----------
To examine the effect of incremental training on OMP adaptation.
"""
import matplotlib.pyplot as plt
from toy_model import ToyNetwork
import numpy as np
import copy
import os
from utils import units_convert, col_o, col_w
plt.style.use('rnn4bci_plot_params.dms')

def main(do_incremental_training):
    tag = "incremental-training-no-partial-increment"  # identification of this experiment, for bookkeeping
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

    lr_adapts = [lr]

    nb_iter = int(1e3)
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False

    # Total losses
    loss_init = np.empty(shape=(len(seeds), nb_iter))
    loss_compare_lr = []

    lr_adapt = (0, lr, 0)
    nb_iter_adapt = int(5e2)

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
                             outfile_name=f"{save_dir_results}/SampleEndInitialTraining.png")


        print('\n|-------------------------------- Fit decoder --------------------------------|')
        net1 = copy.deepcopy(net0)
        intrinsic_manifold_dim, _ = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim, threshold=0.99)
        net1.network_name = 'fitted'
        V_intrinsic = copy.copy(net1.V)
        if seed_id == 0:
            net1.plot_sample(sample_size=1000,
                             outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting.png")


        print('\n|-------------------------------- Re-training with decoder --------------------------------|')
        net2 = copy.deepcopy(net1)
        net2.network_name = 'retrained_after_fitted'
        if relearn_after_decoder_fitting:
            loss_decoder_retraining, _, _, _, _, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)
            if seed_id == 0:
                net2.plot_sample(sample_size=1000,
                                 outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder.png")


        print('\n|-------------------------------- Select perturbations --------------------------------|')
        selected_wm, selected_om, wm_t_l, om_t_l = \
            net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1])

        print('\n|-------------------------------- Incremental training --------------------------------|')
        net_om = copy.deepcopy(net2)
        net_om.network_name = 'om'
        V_OM = copy.copy(net_om.D @ net_om.C[:, selected_om])

        if do_incremental_training:
            as_ = [1, 2, 3, 4, 5]  # DEBUG [1, 2, 3, 4, 4.5, 5]
            concat_total_loss = []
            for a in as_:
                net_om.V = (1 - a/max(as_)) * V_intrinsic + a/max(as_) * V_OM
                l, norm, a_min, a_max, nve, R_seed, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)
                concat_total_loss.append(l['total'])
            concat_total_loss = np.concatenate(concat_total_loss)
        else:
            net_om.V = V_OM
            l, norm, a_min, a_max, nve, R_seed, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)
            concat_total_loss = l['total']

        loss_compare_lr.append(concat_total_loss)

        sample_filename = f"{save_dir_results}/SampleOMAfterLearning_Seed{seed_id}.png" \
            if do_incremental_training \
            else f"{save_dir_results}/SampleOMAfterNoIncrementalLearning_Seed{seed_id}.png"
        net_om.plot_sample(sample_size=1000,
                           outfile_name=sample_filename)


        fig, ax = plt.subplots(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
        ax.semilogy(concat_total_loss, color=col_o)
        ax.set_xlabel('Weight update post-perturbation')
        ax.set_ylabel('Loss')
        ax.set_ylim(ymin=1e-3)
        plt.tight_layout()
        filename = f'{save_dir_results}/LossSeed{seed_id}.png' if do_incremental_training \
            else f'{save_dir_results}/LossNoIncrementalTrainingSeed{seed_id}.png'
        plt.savefig(filename)
        plt.close()

    # Save parameters
    param_dict = {'size': size,
                  'input_noise_intensity': input_noise_intensity,
                  'private_noise_intensity': private_noise_intensity,
                  'lr': lr, 'lr_init': lr_init, 'lr_decoder': lr_decoder, 'lr_adapts': lr_adapts,
                  'nb_iter': nb_iter,
                  'nb_iter_adapts': np.array(5e2/np.array(lr_adapts)*lr, dtype=int),
                  'intrinsic_manifold_dim': intrinsic_manifold_dim,
                  'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
    if do_incremental_training:
        np.save(f"{save_dir}/params", param_dict)
    else:
        np.save(f"{save_dir}/params_no_incremental_training", param_dict)

    # Save performances
    loss_dict = {'loss_init': loss_init,
                 'loss': loss_compare_lr}
    if do_incremental_training:
        np.save(f"{save_dir}/loss", loss_dict)
    else:
        np.save(f"{save_dir}/loss_no_incremental_training", loss_dict)


if __name__ == '__main__':
    main(True)
    main(False)
