from linearized_model import LinearizedModel
import numpy as np
import copy
import os
from scipy.linalg import subspace_angles

def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)              # (input size, recurrent size, output size)
    nb_readouts = 100
    intrinsic_manifold_dim = 5      # dimension of manifold for control (M)
    lr_init = 5e-2 #3e-2                  # learning rate for initial training
    lr_decod = lr_init / 2
    lr = 1./size[1]  #0.1e-2                       # learning rate during adaptation
    nb_iter = int(5e2)              # nb of gradient iteration during initial training
    nb_iter_adapt = int(5e2)        # nb of gradient iteration during adaptation
    seeds = np.arange(1, dtype=int)
    exponents_W = [0.55]        # W_0 ~ N(0, 1/N^exponent_W)
    activation_function = 'linear'
    stopping_crit = 1e-3

    relearn_after_decoder_fitting = True
    do_record_data = True
    do_z_score = True
    global_mean_input_is_zero = False
    fit_intercept = True

    for exponent_W in exponents_W:
        # Manage save and load folders
        tag = (f"fig2-linearized-N{size[1]}-{activation_function}-m{intrinsic_manifold_dim}-zscore{do_z_score}-zeroedavgx{global_mean_input_is_zero}"
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
            loss_init = []
            loss = {'WM': [],
                    'OM': []}
            # Loss components
            loss_corr = {'WM': [],
                         'OM': []}

            # Initial manifold dimension
            real_dims = np.empty(shape=(len(seeds, )))

            # Principal angles
            min_angles = {'WM': {'dVar_vs_VT': [],
                                 'UpperVar_vs_VT': [],
                                 'LowerVar_vs_VT': [],
                                 'UpperVar_vs_VarBCI': []},
                          'OM': {'dVar_vs_VT': [],
                                 'UpperVar_vs_VT': [],
                                 'LowerVar_vs_VT': [],
                                 'UpperVar_vs_VarBCI': []}
                          }
            max_angles = copy.deepcopy(min_angles)

            # Norm of grad W
            norm_gradW = {'loss': {'WM': [],
                                   'OM': []},
                          'loss_tot_var': {'WM': [],
                                           'OM': []}}

            # Normalized variance explained
            normalized_variance_explained = {'WM': [],
                                             'OM': []}

            # Ratio of projected variance (OM)
            R = []

            # tr(C_OM @ Var @ C_OM.T) / tr(C_OM @ Var_init @ C_OM.T)
            rel_proj_var_OM = []

            # tr(C @ Var @ C.T) / tr(Var)
            f = {'WM': [],
                 'OM': []}

            # Amount of covariability projected along the row space of D
            A = {'D': [],
                 'DP_WM': []}

            # Total variance
            tot_var = {'WM': [],
                       'OM': []}

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
            print(f'\n|==================================== Seed {seed} ======================================|')
            print('\n|-------------------------------- Initial training --------------------------------|')
            net0 = LinearizedModel(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0],
                                   exponent_W=exponent_W, global_mean_input_is_zero=global_mean_input_is_zero,
                                   rng_seed=seed_id)
            data = net0.train(lr=lr_init, stopping_crit=0.1*stopping_crit, do_record_data=do_record_data)

            if do_record_data:
                # p_ratio['initial'][seed_id] = net0.participation_ratio()
                loss_init.append(data['losses']['task'])

            if seed_id == 0:
                net0.plot_output(
                    outfile_name=f"{save_dir_results}/SampleEndInitialTraining_seed{seed}.{output_fig_format}")

            print('\n|-------------------------------- Fit decoder --------------------------------|')
            net1 = copy.deepcopy(net0)
            if do_record_data:
                intrinsic_manifold_dim, real_dims[seed_id] = net1.decoder.fit(np.asarray(net1.conditioned_activities()),
                                                                              net1.network_covariance(),
                                                                              intrinsic_manifold_dim=intrinsic_manifold_dim,
                                                                              fit_intercept=fit_intercept,
                                                                              do_z_score=do_z_score)
            else:
                intrinsic_manifold_dim, _ = net1.decoder.fit(np.asarray(net1.conditioned_activities()),
                                                             net1.network_covariance(),
                                                             intrinsic_manifold_dim=intrinsic_manifold_dim,
                                                             fit_intercept=fit_intercept, do_z_score=do_z_score)

            net1.plot_output(
                outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting_seed{seed}.{output_fig_format}")

            '------------------------------------------- Retraining decoder -------------------------------------------'
            net2 = copy.deepcopy(net1)
            if relearn_after_decoder_fitting:
                if net2.task_loss() > stopping_crit:
                    print(
                        '\n|-------------------------------- Re-training with decoder --------------------------------|')
                    print("task loss after initial training:", net2.task_loss())
                    net2.train(lr=lr_decod, stopping_crit=stopping_crit)
                    net2.plot_output(
                        outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder_seed{seed}.{output_fig_format}")

            print('\n|-------------------------------- Select perturbations --------------------------------|')
            selected_perm, t_l = \
                net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=nb_readouts, nb_samples=int(1e3),
                                    om_select_method='modified')
            np.save(f"{save_dir}/candidate_wm_perturbations_seed{seed}", t_l['WM'])
            np.save(f"{save_dir}/candidate_om_perturbations_seed{seed}", t_l['OM'])

            print('\n|-------------------------------- WM perturbation --------------------------------|')
            net_wm = copy.deepcopy(net2)
            net_wm.decoder.apply_perturb(selected_perm['WM'], 'WM')  # apply WM perturbation

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMBeforeLearning_seed{seed}.{output_fig_format}")

            data = net_wm.train(lr=lr, stopping_crit=stopping_crit)

            net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMAfterLearning_seed{seed}.{output_fig_format}")

            if do_record_data:
                loss['WM'].append(data['losses']['task'])
                loss_corr['WM'].append(data['losses']['corr'])
                norm_gradW['loss']['WM'].append(data['norm_gradW'])

                normalized_variance_explained['WM'].append(data['normalized_variance_explained'])
                A['D'].append(data['A']['D'])
                A['DP_WM'].append(data['A']['DP_WM'])
                f['WM'].append(data['f'])
                tot_var['WM'].append(data['tot_var'])

                # for key in min_angles['WM'].keys():
                #    min_angles['WM'][key][seed_id] = data['min_angles'][key]
                #    max_angles['WM'][key][seed_id] = data['max_angles'][key]

                p_ratio['WM'][seed_id] = net_wm.participation_ratio()
                total_change_W_Fnorm['WM'][seed_id] = np.linalg.norm(net_wm.W - net0.W)

            print('\n|-------------------------------- OM perturbation --------------------------------|')
            net_om = copy.deepcopy(net2)
            net_om.decoder.apply_perturb(selected_perm['OM'], 'OM')  # apply OM perturbation

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMBeforeLearning_seed{seed}.{output_fig_format}")

            data = net_om.train(lr=lr, stopping_crit=stopping_crit)

            net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMAfterLearning_seed{seed}.{output_fig_format}")

            if do_record_data:
                loss['OM'].append(data['losses']['task'])
                loss_corr['OM'].append(data['losses']['corr'])
                norm_gradW['loss']['OM'].append(data['norm_gradW'])

                normalized_variance_explained['OM'].append(data['normalized_variance_explained'])
                R.append(data['R'])
                f['OM'].append(data['f'])
                tot_var['OM'].append(data['tot_var'])
                rel_proj_var_OM.append(data['rel_proj_var_OM'])
                # for key in min_angles['WM'].keys():
                #    min_angles['OM'][key][seed_id] = data['min_angles'][key]
                #    max_angles['OM'][key][seed_id] = data['max_angles'][key]

                p_ratio['OM'][seed_id] = net_om.participation_ratio()
                total_change_W_Fnorm['OM'][seed_id] = np.linalg.norm(net_om.W - net0.W)

        if do_record_data:
            # Save parameters
            param_dict = {'size': size,
                          'nb_seeds': len(seeds),
                          'lr_init': lr_init, 'lr_decoder': lr_init, 'lr_adapt': lr,
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

            # Save participation ratios
            np.save(f"{save_dir}/participation_ratio", p_ratio)

            # Save Frobenius norm of total weight change
            np.save(f"{save_dir}/total_change_W_Fnorm", total_change_W_Fnorm)


if __name__ == '__main__':
    main()
