from noisy_linear_model import NoisyLinearNetwork
import numpy as np
import copy
import os
from utils import build_data_container, update_data_container


def main():
    outfig_format = 'png'

    # Parameters
    size = (6, 500, 2)              # (input size, recurrent size, output size)
    nb_readouts = 100
    noise = 1e-2
    lr_init = 0.5 / size[1] #5 / size[1]                    # learning rate for initial training
    lr_decod = lr_init
    lr = lr_init / 10  #0.5 / size[1]                        # learning rate during adaptation
    seeds = np.arange(5, dtype=int)
    stopping_crit_pretraining = 5e-3
    stopping_crit = 5e-3
    exponents_W = [0.55, 1.]        # W_0 ~ N(0, 1/N^exponent_W)

    do_record_data = True
    do_z_score = True
    fit_intercept = True
    global_mean_input_is_zero = False
    relearn_after_decoder_fitting = True

    for exponent_W in exponents_W:
        # Manage save and load folders
        tag = (f"fig2-N{size[1]}-slowscaledLR-mdim95-zscore{do_z_score}-zeroedavgx{global_mean_input_is_zero}"
               f"-fitinter{fit_intercept}-expW{exponent_W}")  # identification of this experiment
        save_dir = f"data/egd/{tag}"
        save_dir_results = f"results/egd/{tag}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir_results):
            os.makedirs(save_dir_results)

        # Data containers
        manifold_dimension = np.empty(len(seeds), dtype=int)
        if do_record_data:
            data = {'pretraining': build_data_container(),
                    'WM': build_data_container(),
                    'OM': build_data_container()}


        for seed_id, seed in enumerate(seeds):
            print(f'\n|================================= Seed {seed} ===================================|')
            print('\n|----------------------------- Initial training -----------------------------|')
            net0 = NoisyLinearNetwork(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0],
                                      exponent_W=exponent_W, global_mean_input_is_zero=global_mean_input_is_zero,
                                      rng_seed=seed_id, noise=noise)

            net0.plot_output(outfile_name=f"{save_dir_results}/SampleBeforeInitialTraining_seed{seed}.{outfig_format}")

            data_pretraining = net0.train(lr=lr_init, stopping_crit=stopping_crit_pretraining, do_record_data=do_record_data)

            net0.plot_output(outfile_name=f"{save_dir_results}/SampleEndInitialTraining_seed{seed}.{outfig_format}")

            if do_record_data:
                update_data_container(data_pretraining, data['pretraining'])

            print('\n|-------------------------------- Fit decoder --------------------------------|')
            net1 = copy.deepcopy(net0)
            activities, _ = net1.sample(nb_epochs=100)
            manifold_dimension[seed_id] = net1.decoder.fit(activities,
                                                           net1.network_covariance(),
                                                           fit_intercept=fit_intercept,
                                                           do_z_score=do_z_score)

            net1.plot_output(outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting_seed{seed}.{outfig_format}")

            '----------------------------------------- Retraining w/ decoder -----------------------------------------'
            net2 = copy.deepcopy(net1)
            if relearn_after_decoder_fitting:
                if net2.task_loss() > stopping_crit_pretraining:
                    print(
                        '\n|-------------------------------- Re-training with decoder --------------------------------|')
                    net2.train(lr=lr_decod, stopping_crit=stopping_crit_pretraining)
                    net2.plot_output(outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder_seed{seed}.{outfig_format}")

            print('\n|-------------------------------- Select perturbations --------------------------------|')
            selected_perm, t_l = \
                net2.select_perturb(manifold_dimension[seed_id], nb_om_permuted_units=nb_readouts,
                                    om_select_method='modified')
            np.save(f"{save_dir}/candidate_wm_perturbations_seed{seed}", t_l['WM'])
            np.save(f"{save_dir}/candidate_om_perturbations_seed{seed}", t_l['OM'])

            print('\n|-------------------------------- WM perturbation --------------------------------|')
            net_wm = copy.deepcopy(net2)
            net_wm.decoder.apply_perturb(selected_perm['WM'], 'WM')  # apply WM perturbation

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMBeforeLearning_seed{seed}.{outfig_format}")

            data_wm = net_wm.train(lr=lr, stopping_crit=stopping_crit)

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMAfterLearning_seed{seed}.{outfig_format}")

            if do_record_data:
                update_data_container(data_wm, data['WM'])

            print('\n|-------------------------------- OM perturbation --------------------------------|')
            net_om = copy.deepcopy(net2)
            net_om.decoder.apply_perturb(selected_perm['OM'], 'OM')  # apply OM perturbation

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMBeforeLearning_seed{seed}.{outfig_format}")

            data_om = net_om.train(lr=lr, stopping_crit=stopping_crit)

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMAfterLearning_seed{seed}.{outfig_format}")

            if do_record_data:
                update_data_container(data_om, data['OM'])

        if do_record_data:
            param_dict = {'size': size,
                          'nb_seeds': len(seeds),
                          'lr_init': lr_init, 'lr_decoder': lr_init,'lr_adapt': lr,
                          'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
            np.save(f"{save_dir}/params", param_dict)
            np.save(f"{save_dir}/data", data)

if __name__ == '__main__':
    main()
