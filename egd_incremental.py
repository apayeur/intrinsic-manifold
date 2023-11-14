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

    # Parameters
    size = (6, 100, 2)
    input_noise_intensity = 0e-4
    private_noise_intensity = 1e-2  # 1e-3
    intrinsic_manifold_dim = 6  # 6
    lr_init = (0, 1e-2, 0)
    lr_decoder = (0, 5e-3, 0)
    lr_adapt = (0, 0.001, 0)
    exponent_W = 0.55

    nb_iter = int(5e2)
    nb_iter_adapt = int(1e2)
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False
    fraction_decrease_loss = 0.75

    tag = f"incremental-training-exponent{exponent_W}"  # identification of this experiment, for bookkeeping
    save_dir = f"data/egd/{tag}"
    save_dir_results = f"results/egd/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    # Total losses
    loss_init = np.empty(shape=(len(seeds), nb_iter))
    loss = []

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
        l, _, _, _, _, _, _, _, _, _, _ = net0.train(lr=lr_init, nb_iter=nb_iter)
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


        net2 = copy.deepcopy(net1)
        net2.network_name = 'retrained_after_fitted'
        if relearn_after_decoder_fitting:
            print('\n|-------------------------------- Re-training with decoder --------------------------------|')
            loss_decoder_retraining, _, _, _, _, _, _, _, _,  _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)
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

        nb_iter_adapt_per_increment = -1
        if do_incremental_training:
            as_ = [2, 3, 4, 4.5, 5] #np.linspace(1, 5, num=4, endpoint=True)  #[1, 2, 3, 4, 4.5, 5]  # DEBUG [1, 2, 3, 4, 4.5, 5]
            concat_total_loss = []
            cum_nb_iter_incr = 0
            for a in as_:
                print(f"a = {a}: ")
                if cum_nb_iter_incr < nb_iter_adapt:
                    net_om.V = (1 - a/max(as_)) * V_intrinsic + a/max(as_) * V_OM
                    l_0 = net_om.loss_function()
                    l, norm, a_min, a_max, nve, R_seed, _, _, _, _, _ = net_om.train(lr=lr_adapt,
                                                                                     nb_iter=1e6,
                                                                                     stopping_crit=fraction_decrease_loss * l_0)
                    if (len(l['total']) + cum_nb_iter_incr) > nb_iter_adapt:
                        concat_total_loss.append(l['total'][: nb_iter_adapt - cum_nb_iter_incr])
                        print(f"nb_iter = {nb_iter_adapt - cum_nb_iter_incr}")
                        cum_nb_iter_incr = nb_iter_adapt
                        break
                    else:
                        cum_nb_iter_incr += len(l['total'])
                        concat_total_loss.append(l['total'])
                        print(f"nb_iter = {len(l['total'])}")
            if cum_nb_iter_incr < nb_iter_adapt:
                l, norm, a_min, a_max, nve, R_seed, _, _, _, _, _ = net_om.train(lr=lr_adapt,
                                                                                 nb_iter=nb_iter_adapt - cum_nb_iter_incr)
                concat_total_loss.append(l['total'])
            concat_total_loss = np.concatenate(concat_total_loss)
            if a != as_[-1]:
                print("Did not reach full OM")
            print(f"total iterations: {len(concat_total_loss)}")
        else:
            net_om.V = V_OM
            l, norm, a_min, a_max, nve, R_seed, _, _, _, _, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)
            concat_total_loss = l['total']

        loss.append(concat_total_loss)

        eigval_after_OM = np.max(np.abs(np.linalg.eigvals(net_om.W)))
        print(f"Max eigenvalue of W: {eigval_after_OM}")

        sample_filename = f"{save_dir_results}/SampleOMAfterLearning_Seed{seed_id}.png" \
            if do_incremental_training \
            else f"{save_dir_results}/SampleOMAfterNoIncrementalLearning_Seed{seed_id}.png"
        net_om.plot_sample(sample_size=1000,
                           outfile_name=sample_filename)


        fig, ax = plt.subplots(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
        ax.plot(concat_total_loss, color=col_o)
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
                  'lr_init': lr_init, 'lr_decoder': lr_decoder, 'lr_adapt': lr_adapt,
                  'nb_iter': nb_iter,
                  'nb_iter_adapt': nb_iter_adapt,
                  'nb_iter_adapt_per_increment': nb_iter_adapt_per_increment,
                  'exponent_W': exponent_W,
                  'intrinsic_manifold_dim': intrinsic_manifold_dim,
                  'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
    if do_incremental_training:
        np.save(f"{save_dir}/params", param_dict)
    else:
        np.save(f"{save_dir}/params_no_incremental_training", param_dict)

    # Save performances
    loss_dict = {'loss_init': loss_init,
                 'loss': loss}
    if do_incremental_training:
        np.save(f"{save_dir}/loss", loss_dict)
    else:
        np.save(f"{save_dir}/loss_no_incremental_training", loss_dict)


if __name__ == '__main__':
    main(True)
    #main(False)
