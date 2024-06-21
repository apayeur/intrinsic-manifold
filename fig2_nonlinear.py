from toy_model_new import NonlinearDeterministicNetwork
import numpy as np
import copy
import os
from scipy.linalg import subspace_angles

"""
TANH params
lr_init = 1e-2                  # learning rate for initial training
lr = 0.5e-2                       # learning rate during adaptation
nb_iter = int(5e2)              # nb of gradient iteration during initial training
nb_iter_adapt = int(1e3)        # nb of gradient iteration during adaptation
seeds = np.arange(1, dtype=int)
relearn_after_decoder_fitting = False
exponents_W = [0.55, 1.]        # W_0 ~ N(0, 1/N^exponent_W)
do_record_data = True
activation_function = 'tanh'
"""


def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)              # (input size, recurrent size, output size)
    intrinsic_manifold_dim = 5      # dimension of manifold for control (M)
    lr_init = 5e-2 #3e-2                  # learning rate for initial training
    lr_decod = lr_init / 2
    lr = 2e-3 #0.1e-2                       # learning rate during adaptation
    nb_iter = int(2e2)              # nb of gradient iteration during initial training
    nb_iter_adapt = int(1e3)        # nb of gradient iteration during adaptation
    seeds = np.arange(5, dtype=int)
    exponents_W = [0.5, 1.]        # W_0 ~ N(0, 1/N^exponent_W)
    activation_function = 'relu'

    relearn_after_decoder_fitting = True
    do_record_data = True
    do_z_score = False
    global_mean_input_is_zero = False
    fit_intercept = True

    for exponent_W in exponents_W:
        # Manage save and load folders
        tag = (f"fig2-m{intrinsic_manifold_dim}-zscore{do_z_score}-zeroedavgx{global_mean_input_is_zero}"
               f"-fitinter{fit_intercept}-expW{exponent_W}")  # identification of this experiment
        save_dir = f"data/egd/{tag}"
        save_dir_results = f"results/egd/{tag}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir_results):
            os.makedirs(save_dir_results)

        # DEFINITION OF DATA CONTAINERS FOR SAVED DATA
        if do_record_data:
            # Total losses
            loss_init = np.empty(shape=(len(seeds), nb_iter))
            loss = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                    'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}
            # Loss components
            loss_corr = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                         'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

            # Initial manifold dimension
            real_dims = np.empty(shape=(len(seeds,)))

            # Principal angles
            min_angles = {'WM': {'dVar_vs_VT': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                 'UpperVar_vs_VT': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                 'LowerVar_vs_VT': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                 'UpperVar_vs_VarBCI': np.empty(shape=(len(seeds), nb_iter_adapt))},
                          'OM': {'dVar_vs_VT': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                 'UpperVar_vs_VT': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                 'LowerVar_vs_VT': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                 'UpperVar_vs_VarBCI': np.empty(shape=(len(seeds), nb_iter_adapt))}
                          }
            max_angles = copy.deepcopy(min_angles)

            # Angles between V_OM and V_0 and between V_WM and V_0
            output_matrix_angles = {'WM': np.empty(shape=(len(seeds), size[2])),
                                    'OM': np.empty(shape=(len(seeds), size[2]))}

            # Norm of grad W
            norm_gradW = {'loss':{'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                  'OM': np.empty(shape=(len(seeds), nb_iter_adapt))},
                          'loss_tot_var': {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                       'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}}

            # Normalized variance explained
            normalized_variance_explained = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                             'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

            # Ratio of projected variance (OM)
            R = np.empty(shape=(len(seeds), nb_iter_adapt))

            # tr(C_OM @ Var @ C_OM.T) / tr(C_OM @ Var_init @ C_OM.T)
            rel_proj_var_OM = np.empty(shape=(len(seeds), nb_iter_adapt))

            # tr(C @ Var @ C.T) / tr(Var)
            f = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                 'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

            # Amount of covariability projected along the row space of D
            A = {'D': np.empty(shape=(len(seeds), nb_iter_adapt)),
                 'DP_WM': np.empty(shape=(len(seeds), nb_iter_adapt))}

            # Total variance
            tot_var = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                       'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

            # Candidate perturbation losses
            wm_total_losses, om_total_losses = None, None

            # Participation ratio
            p_ratio = {'initial': np.empty(shape=len(seeds)),
                       'WM': np.empty(shape=len(seeds)),
                       'OM': np.empty(shape=len(seeds))}

            # Frobenius norm of total weight change
            total_change_W_Fnorm = {'WM': np.empty(shape=len(seeds)),
                                    'OM': np.empty(shape=len(seeds))}

        for seed_id, seed in enumerate(seeds):
            print(f'\n|==================================== Seed {seed} =====================================|')
            print('\n|-------------------------------- Initial training --------------------------------|')
            net0 = NonlinearDeterministicNetwork(network_size=size[1], nb_inputs=size[0], exponent_W=exponent_W,
                                                 global_mean_input_is_zero=global_mean_input_is_zero,
                                                 do_z_score=do_z_score, rng_seed=seed_id,
                                                 activation_function=activation_function)
            data = net0.train(lr=lr_init, nb_iter=nb_iter, do_record_data=do_record_data)
            ca = np.asarray(net0.conditioned_activities()).mean(axis=0)

            # compute participation ratio
            if do_record_data:
                p_ratio['initial'][seed_id] = net0.participation_ratio()

            # save loss
            if do_record_data:
                loss_init[seed_id] = data['losses']['task']

            if seed_id == 0:
                net0.plot_output(outfile_name=f"{save_dir_results}/SampleEndInitialTraining_seed{seed}.{output_fig_format}")

            print('\n|-------------------------------- Fit decoder --------------------------------|')
            net1 = copy.deepcopy(net0)
            if do_record_data:
                intrinsic_manifold_dim, real_dims[seed_id] = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim,
                                                                              threshold=0.95, fit_intercept=fit_intercept)
            else:
                intrinsic_manifold_dim, _ = net1.fit_decoder(
                    intrinsic_manifold_dim=intrinsic_manifold_dim,
                    threshold=0.95, fit_intercept=fit_intercept)

            net1.plot_output(outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting_seed{seed}.{output_fig_format}")
            V_0 = copy.deepcopy(net1.V)
            print("Decoder intercept :", net1.intercept)

            '------------------------------------------- Retraining decoder -------------------------------------------'
            net2 = copy.deepcopy(net1)
            if relearn_after_decoder_fitting:
                if net2.task_loss() > 1e-4:
                    print(
                        '\n|-------------------------------- Re-training with decoder --------------------------------|')
                    print("task loss after initial training:", net2.task_loss())
                    net2.train(lr=lr_decod, nb_iter=nb_iter // 2)
                    net2.plot_output(outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder_seed{seed}.{output_fig_format}")

            print('\n|-------------------------------- Select perturbations --------------------------------|')
            selected_wm, selected_om, wm_t_l, om_t_l = \
                net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1] // 2, nb_samples=int(1e3))
            if seed_id == 0:
                wm_total_losses, om_total_losses = wm_t_l, om_t_l

            print('\n|-------------------------------- WM perturbation --------------------------------|')
            net_wm = copy.deepcopy(net2)
            net_wm.apply_wm_perturb(selected_wm)  # apply WM perturbation
            if do_record_data:
                output_matrix_angles['WM'][seed_id] = np.rad2deg(subspace_angles(V_0.T, net_wm.V.T))

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMBeforeLearning_seed{seed}.{output_fig_format}")

            data = net_wm.train(lr=lr, nb_iter=nb_iter_adapt)

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMAfterLearning_seed{seed}.{output_fig_format}")

            if do_record_data:
                loss['WM'][seed_id] = data['losses']['task']
                loss_corr['WM'][seed_id] = data['losses']['corr']
                norm_gradW['loss']['WM'][seed_id] = data['norm_gradW']

                normalized_variance_explained['WM'][seed_id] = data['normalized_variance_explained']
                A['D'][seed_id] = data['A']['D']
                A['DP_WM'][seed_id] = data['A']['DP_WM']
                f['WM'][seed_id] = data['f']
                tot_var['WM'][seed_id] = data['tot_var']

                #for key in min_angles['WM'].keys():
                #    min_angles['WM'][key][seed_id] = data['min_angles'][key]
                #    max_angles['WM'][key][seed_id] = data['max_angles'][key]

                p_ratio['WM'][seed_id] = net_wm.participation_ratio()

                total_change_W_Fnorm['WM'][seed_id] = np.linalg.norm(net_wm.W - net0.W)

            print('\n|-------------------------------- OM perturbation --------------------------------|')
            net_om = copy.deepcopy(net2)
            net_om.apply_om_perturb(selected_om)  # apply OM perturbation
            if do_record_data:
                output_matrix_angles['OM'][seed_id] = np.rad2deg(subspace_angles(V_0.T, net_om.V.T))

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMBeforeLearning_seed{seed}.{output_fig_format}")

            data = net_om.train(lr=lr, nb_iter=nb_iter_adapt)

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMAfterLearning_seed{seed}.{output_fig_format}")

            if do_record_data:
                loss['OM'][seed_id] = data['losses']['task']
                loss_corr['OM'][seed_id] = data['losses']['corr']
                norm_gradW['loss']['OM'][seed_id] = data['norm_gradW']

                normalized_variance_explained['OM'][seed_id] = data['normalized_variance_explained']

                R[seed_id] = data['R']
                f['OM'][seed_id] = data['f']
                tot_var['OM'][seed_id] = data['tot_var']
                rel_proj_var_OM[seed_id] = data['rel_proj_var_OM']
                #for key in min_angles['WM'].keys():
                #    min_angles['OM'][key][seed_id] = data['min_angles'][key]
                #    max_angles['OM'][key][seed_id] = data['max_angles'][key]

                p_ratio['OM'][seed_id] = net_om.participation_ratio()

                total_change_W_Fnorm['OM'][seed_id] = np.linalg.norm(net_om.W - net0.W)

        if do_record_data:
            # Save parameters
            param_dict = {'size': size,
                          'nb_seeds': len(seeds),
                          'lr_init': lr_init, 'lr_decoder': lr_init,'lr_adapt': lr,
                          'nb_iter': nb_iter,
                          'nb_iter_adapt': nb_iter_adapt,
                          'intrinsic_manifold_dim': intrinsic_manifold_dim,
                          'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
            np.save(f"{save_dir}/params", param_dict)

            # Save performances
            loss_dict = {'loss_init': loss_init,
                         'loss': loss,
                         'loss_corr': loss_corr}
            np.save(f"{save_dir}/loss", loss_dict)

            # Save norm_gradW
            np.save(f"{save_dir}/norm_gradW", norm_gradW)

            # Save initial manifold dimension
            np.save(f"{save_dir}/real_dims", real_dims)

            # Save principal angles
            np.save(f"{save_dir}/principal_angles_min", min_angles)
            np.save(f"{save_dir}/principal_angles_max", max_angles)
            np.save(f"{save_dir}/output_matrix_angles", output_matrix_angles)
            print("Min angle WM vs V_0: {} +/- {}".format(np.mean(output_matrix_angles['WM'][:,1]),
                                                          np.std(output_matrix_angles['WM'][:,1], ddof=1)/len(seeds)**0.5))
            print("Max angle WM vs V_0: {} +/- {}".format(np.mean(output_matrix_angles['WM'][:, 0]),
                                                          np.std(output_matrix_angles['WM'][:, 0], ddof=1) / len(seeds) ** 0.5))
            print("Min angle OM vs V_0: {} +/- {}".format(np.mean(output_matrix_angles['OM'][:, 1]),
                                                          np.std(output_matrix_angles['OM'][:, 1], ddof=1) / len(seeds) ** 0.5))
            print("Max angle OM vs V_0: {} +/- {}".format(np.mean(output_matrix_angles['OM'][:, 0]),
                                                          np.std(output_matrix_angles['OM'][:, 0], ddof=1) / len(seeds) ** 0.5))
            # Save normalized variance explained
            np.save(f"{save_dir}/normalized_variance_explained", normalized_variance_explained)

            # Save ratio of projected covariability
            np.save(f"{save_dir}/R", R)

            # Save f
            np.save(f"{save_dir}/f", f)

            # Save total variance
            np.save(f"{save_dir}/tot_var", tot_var)

            # Save rel_proj_var_OM
            np.save(f"{save_dir}/rel_proj_var_OM", rel_proj_var_OM)

            # Save amount of covariability projected along the row space of D
            np.save(f"{save_dir}/A", A)

            # Save candidate perturbations losses
            np.save(f"{save_dir}/candidate_wm_perturbations", wm_total_losses)
            np.save(f"{save_dir}/candidate_om_perturbations", om_total_losses)

            # Save participation ratios
            np.save(f"{save_dir}/participation_ratio", p_ratio)

            # Save Frobenius norm of total weight change
            np.save(f"{save_dir}/total_change_W_Fnorm", total_change_W_Fnorm)


if __name__ == '__main__':
    main()
