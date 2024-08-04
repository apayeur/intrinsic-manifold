from nonlinear_model import NonlinearDeterministicNetwork
from noisy_linear_model import NoisyLinearNetwork
import numpy as np
import copy
import os
from utils import build_data_container, update_data_container, matrix_angle
from scipy.linalg import subspace_angles


def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)  # (input size, recurrent size, output size)
    nb_readouts = 100  # size[1]

    seeds = np.arange(10, dtype=int)
    exponents_W = [1., 0.55]  # W_0 ~ N(0, 1/N^exponent_W)
    activation_function = 'tanh'

    base_lr = 0.5 / size[1]
    lr_init = {0.55: base_lr, 1.: base_lr * nb_readouts}  # learning rate for initial training
    lr = {0.55: 5*base_lr, 1.: 5*base_lr}  # 0.1e-2                                  # learning rate during adaptation

    stopping_crit = 1e-5

    relearn_after_decoder_fitting = True
    do_record_data = True
    do_z_score = True
    fit_intercept = True
    global_mean_input_is_zero = False

    for exponent_W in exponents_W:
        # Manage save and load folders
        tag = (
            f"fig2-{activation_function}-dimthresh{95}-zscore{do_z_score}-zeroedavgx{global_mean_input_is_zero}"
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

            # Alignment
            representation_alignment = {'initial': np.empty(len(seeds)),
                                        'WM': np.empty(len(seeds)),
                                        'OM': np.empty(len(seeds))}
            tangent_kernel_alignment = {'initial': np.zeros((len(seeds), 4)),
                                        'WM': np.zeros((len(seeds), 4)),
                                        'OM': np.zeros((len(seeds), 4))}
            delta_W_norm = {'initial': np.empty(len(seeds)),
                            'WM': np.empty(len(seeds)),
                            'OM': np.empty(len(seeds))}


        for seed_id, seed in enumerate(seeds):
            print(f'\n|==================================== Seed {seed} ======================================|')
            print('\n|-------------------------------- Initial training --------------------------------|')
            if activation_function in ['tanh', 'relu']:
                net0 = NonlinearDeterministicNetwork(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0],
                                                     exponent_W=exponent_W,
                                                     global_mean_input_is_zero=global_mean_input_is_zero,
                                                     rng_seed=seed, activation_function=activation_function)
            elif activation_function == 'linear':
                net0 = NoisyLinearNetwork(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0],
                                          exponent_W=exponent_W, global_mean_input_is_zero=global_mean_input_is_zero,
                                          rng_seed=seed_id, noise=0.)
            else:
                raise ValueError("'activation_function' should be `linear`, `relu` or `tanh`")

            print("norm of output matrix", np.linalg.norm(net0.decoder.VR()))

            if do_record_data:
                NTK_0 = net0.neural_tangent_kernel()
                RSM_0 = net0.representation_similarity_matrix()
                W = copy.copy(net0.W)

            net0.plot_output(outfile_name=f"{save_dir_results}/SampleBeforeInitialTraining_seed{seed}.{output_fig_format}")
            data_loc = net0.train(lr=lr_init[exponent_W], stopping_crit=stopping_crit, do_record_data=do_record_data)
            net0.plot_output(outfile_name=f"{save_dir_results}/SampleEndInitialTraining_seed{seed}.{output_fig_format}")

            if do_record_data:
                NTK = net0.neural_tangent_kernel()
                RSM = net0.representation_similarity_matrix()

                representation_alignment['initial'][seed_id] = matrix_angle(RSM, RSM_0)
                tangent_kernel_alignment['initial'][seed_id] = matrix_angle(NTK, NTK_0)
                delta_W_norm['initial'][seed_id] = np.linalg.norm(net0.W - W)
                update_data_container(data_loc, data['pretraining'])

                NTK_0 = copy.copy(NTK)
                RSM_0 = copy.copy(RSM)
                W = copy.copy(net0.W)


            print('\n|-------------------------------- Fit decoder --------------------------------|')
            net1 = copy.deepcopy(net0)  # not necessary...
            activities = np.array(net1.conditioned_activities())
            manifold_dimension[seed_id] = net1.decoder.fit(activities,
                                                           net1.network_covariance(),
                                                           fit_intercept=fit_intercept,
                                                           do_z_score=do_z_score)

            net1.plot_output(outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting_seed{seed}.{output_fig_format}")
            print("norm of output matrix", np.linalg.norm(net1.decoder.VR()))

            '----------------------------------------- Retraining w/ decoder -----------------------------------------'
            net2 = copy.deepcopy(net1)
            if relearn_after_decoder_fitting:
                if net2.task_loss() > stopping_crit:
                    print(
                        '\n|-------------------------------- Re-training with decoder --------------------------------|')
                    net2.train(lr=lr_init[exponent_W], stopping_crit=stopping_crit)
                    net2.plot_output(
                        outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder_seed{seed}.{output_fig_format}")

            print('\n|-------------------------------- Select perturbations --------------------------------|')
            selected_perm, t_l = \
                net2.select_perturb(manifold_dimension[seed_id], nb_om_permuted_units=nb_readouts,
                                    om_select_method='modified')
            np.save(f"{save_dir}/candidate_wm_perturbations_seed{seed}", t_l['WM'])
            np.save(f"{save_dir}/candidate_om_perturbations_seed{seed}", t_l['OM'])

            print('\n|-------------------------------- WM perturbation --------------------------------|')
            net_wm = copy.deepcopy(net2)
            net_wm.decoder.apply_perturb(selected_perm['WM'], 'WM')  # apply WM perturbation

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMBeforeLearning_seed{seed}.{output_fig_format}")
            data_loc = net_wm.train(lr=lr[exponent_W], stopping_crit=stopping_crit)
            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMAfterLearning_seed{seed}.{output_fig_format}")

            if do_record_data:
                NTK = net_wm.neural_tangent_kernel()
                RSM = net_wm.representation_similarity_matrix()

                representation_alignment['WM'][seed_id] = matrix_angle(RSM, RSM_0)
                tangent_kernel_alignment['WM'][seed_id] = matrix_angle(NTK, NTK_0)
                delta_W_norm['WM'][seed_id] = np.linalg.norm(net_wm.W - W)
                update_data_container(data_loc, data['WM'])


            print('\n|-------------------------------- OM perturbation --------------------------------|')
            net_om = copy.deepcopy(net2)
            net_om.decoder.apply_perturb(selected_perm['OM'], 'OM')  # apply OM perturbation

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMBeforeLearning_seed{seed}.{output_fig_format}")
            data_loc = net_om.train(lr=lr[exponent_W], stopping_crit=stopping_crit)
            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMAfterLearning_seed{seed}.{output_fig_format}")

            if do_record_data:
                NTK = net_om.neural_tangent_kernel()
                RSM = net_om.representation_similarity_matrix()

                representation_alignment['OM'][seed_id] = matrix_angle(RSM, RSM_0)
                tangent_kernel_alignment['OM'][seed_id] = matrix_angle(NTK, NTK_0)
                delta_W_norm['OM'][seed_id] = np.linalg.norm(net_om.W - W)
                update_data_container(data_loc, data['OM'])

        if do_record_data:
            # Save parameters
            param_dict = {'size': size,
                          'nb_seeds': len(seeds),
                          'lr_init': lr_init, 'lr_adapt': lr,
                          'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
            np.save(f"{save_dir}/params", param_dict)

            np.save(f"{save_dir}/data", data)
            np.save(f"{save_dir}/representation_alignment", representation_alignment)
            np.save(f"{save_dir}/tangent_kernel_alignment", tangent_kernel_alignment)
            np.save(f"{save_dir}/delta_W_norm", delta_W_norm)

if __name__ == '__main__':
    main()
