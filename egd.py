from toy_model import ToyNetwork
import numpy as np
import copy
import os
from scipy.linalg import subspace_angles


def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)
    input_noise_intensity = 0e-4
    private_noise_intensity = 1e-2  # 1e-3
    intrinsic_manifold_dim = 6  # 6
    lr_init = (0, 1e-2, 0)
    lr_decoder = (0, 5e-3, 0)
    lrs = [0.001] #, 0.002, 0.005, 0.01, 0.02, 0.05]
    nb_iter = int(5e2)  # int(1e3)
    nb_iter_adapt = int(2e3)  # was 5e3
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False
    #exponent_W = 0.55  # W_0 ~ N(0, 1/N^exponent_W)
    exponents_W = [1] #[0.55, 0.6, 0.7, 0.8, 0.9]
    do_scale_V_OM = False

    for exponent_W in exponents_W:
        for lr in lrs:
            lr_adapt = (0, lr, 0)  # was lr/15
            # Manage save and load folders
            tag = f"exponent_W{exponent_W}-lr{lr_adapt[1]}-M{intrinsic_manifold_dim}-iterAdapt{nb_iter_adapt}"  # identification of this experiment, for bookkeeping
            save_dir = f"data/egd/{tag}"
            save_dir_results = f"results/egd/{tag}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(save_dir_results):
                os.makedirs(save_dir_results)

            # DEFINITION OF DATA CONTAINERS FOR SAVED DATA
            # Total losses
            loss_init = np.empty(shape=(len(seeds), nb_iter))
            loss = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                    'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}
            # Loss components
            loss_var = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                        'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}
            loss_exp = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                        'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}
            loss_corr = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                         'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}
            loss_proj = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                         'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}
            loss_vbar = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
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

            # Candidate perturbation losses
            wm_total_losses, om_total_losses = None, None

            # Eigenvalues
            eigenvals_0 = []
            eigenvals_init = []
            eigenvals_after_WMP = []
            eigenvals_after_OMP = []

            # Participation ratio
            p_ratio = {'initial': np.empty(shape=(len(seeds), nb_iter)),
                       'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                       'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

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
                # compute max eigenvalue
                eigenvals_0.append(np.max(np.abs(np.linalg.eigvals(net0.W))))
                l, _, _, _, _, _, _, _, _, p_ratio['initial'][seed_id] = net0.train(lr=lr_init, nb_iter=nb_iter)
                print(p_ratio['initial'][seed_id][0], p_ratio['initial'][seed_id][-1])
                # compute max eigenvalue
                eigval_init = np.max(np.abs(np.linalg.eigvals(net0.W)))
                print(f"Max eigenvalue of W: {eigval_init}")
                eigenvals_init.append(eigval_init)

                # compute participation ratio
                p_ratio['initial'][seed_id] = net0.participation_ratio()

                # save loss
                loss_init[seed_id] = l['total']

                if seed_id == 0:
                    net0.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleEndInitialTraining.{output_fig_format}")

                print('\n|-------------------------------- Fit decoder --------------------------------|')
                net1 = copy.deepcopy(net0)
                intrinsic_manifold_dim, real_dims[seed_id] = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim,
                                                                              threshold=0.9)
                net1.network_name = 'fitted'
                if seed_id == 0:
                    net1.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting.{output_fig_format}")
                V_0 = copy.deepcopy(net1.V)

                print('\n|-------------------------------- Re-training with decoder --------------------------------|')
                net2 = copy.deepcopy(net1)
                net2.network_name = 'retrained_after_fitted'
                if relearn_after_decoder_fitting:
                    loss_decoder_retraining, _, _, _, _, _, _,_, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)
                    if seed_id == 0:
                        net2.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder.{output_fig_format}")

                print('\n|-------------------------------- Select perturbations --------------------------------|')
                selected_wm, selected_om, wm_t_l, om_t_l = \
                    net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1])
                if seed_id == 0:
                    wm_total_losses, om_total_losses = wm_t_l, om_t_l

                print('\n|-------------------------------- WM perturbation --------------------------------|')
                net_wm = copy.deepcopy(net2)
                net_wm.network_name = 'wm'
                net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]
                output_matrix_angles['WM'][seed_id] = np.rad2deg(subspace_angles(V_0.T, net_wm.V.T))

                squared_mean_output_radius_WM = net_wm.mean_output_square_radius()

                if seed_id == 0:
                    net_wm.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleWMBeforeLearning.{output_fig_format}")

                l, norm, a_min, a_max, nve, _, A_tmp, f_seed, _, p_ratio['WM'][seed_id] = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

                if seed_id == 0:
                    net_wm.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleWMAfterLearning.{output_fig_format}")
                print(p_ratio['WM'][seed_id][0], p_ratio['WM'][seed_id][-1])

                loss['WM'][seed_id] = l['total']
                loss_var['WM'][seed_id] = l['var']
                loss_exp['WM'][seed_id] = l['exp']
                loss_corr['WM'][seed_id] = l['corr']
                loss_vbar['WM'][seed_id] = l['vbar']
                loss_proj['WM'][seed_id] = l['proj']
                norm_gradW['loss']['WM'][seed_id] = norm['loss']
                norm_gradW['loss_tot_var']['WM'][seed_id] = norm['loss_tot_var']

                normalized_variance_explained['WM'][seed_id] = nve
                A['D'][seed_id] = A_tmp['D']
                A['DP_WM'][seed_id] = A_tmp['DP_WM']
                f['WM'][seed_id] = f_seed

                for key in min_angles['WM'].keys():
                    min_angles['WM'][key][seed_id] = a_min[key]
                    max_angles['WM'][key][seed_id] = a_max[key]

                eigval_after_WM = np.max(np.abs(np.linalg.eigvals(net_wm.W)))
                print(f"Max eigenvalue of W: {eigval_after_WM}")
                eigenvals_after_WMP.append(eigval_after_WM)

                #p_ratio['WM'][seed_id] = net_wm.participation_ratio()

                print('\n|-------------------------------- OM perturbation --------------------------------|')
                net_om = copy.deepcopy(net2)
                net_om.network_name = 'om'
                net_om.V = net_om.D @ net_om.C[:, selected_om]
                squared_mean_output_radius_OM = net_om.mean_output_square_radius()
                scale_OM = np.sqrt(squared_mean_output_radius_WM / squared_mean_output_radius_OM) if do_scale_V_OM else 1.
                net_om.V = scale_OM * net_om.V

                output_matrix_angles['OM'][seed_id] = np.rad2deg(subspace_angles(V_0.T, net_om.V.T))

                if seed_id == 0:
                    net_om.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleOMBeforeLearning.{output_fig_format}")

                l, norm, a_min, a_max, nve, R_seed, _, f_seed, rel_proj_var_OM_seed, p_ratio['OM'][seed_id] \
                    = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

                if seed_id == 0:
                    net_om.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleOMAfterLearning.{output_fig_format}")
                print(p_ratio['OM'][seed_id][0], p_ratio['OM'][seed_id][-1])

                loss['OM'][seed_id] = l['total']
                loss_var['OM'][seed_id] = l['var']
                loss_exp['OM'][seed_id] = l['exp']
                loss_corr['OM'][seed_id] = l['corr']
                loss_vbar['OM'][seed_id] = l['vbar']
                loss_proj['OM'][seed_id] = l['proj']
                norm_gradW['loss']['OM'][seed_id] = norm['loss']
                norm_gradW['loss_tot_var']['OM'][seed_id] = norm['loss_tot_var']
                normalized_variance_explained['OM'][seed_id] = nve
                R[seed_id] = R_seed
                f['OM'][seed_id] = f_seed
                rel_proj_var_OM[seed_id] = rel_proj_var_OM_seed
                for key in min_angles['OM'].keys():
                    min_angles['OM'][key][seed_id] = a_min[key]
                    max_angles['OM'][key][seed_id] = a_max[key]

                eigval_after_OM = np.max(np.abs(np.linalg.eigvals(net_om.W)))
                print(f"Max eigenvalue of W: {eigval_after_OM}")
                eigenvals_after_OMP.append(eigval_after_OM)

                #p_ratio['OM'][seed_id] = net_om.participation_ratio()

            # Save parameters
            param_dict = {'size': size,
                          'nb_seeds': len(seeds),
                          'input_noise_intensity': input_noise_intensity,
                          'private_noise_intensity': private_noise_intensity,
                          'lr_init': lr_init, 'lr_decoder': lr_decoder,'lr_adapt': lr_adapt,
                          'nb_iter': nb_iter,
                          'nb_iter_adapt': nb_iter_adapt,
                          'intrinsic_manifold_dim': intrinsic_manifold_dim,
                          'relearn_after_decoder_fitting': relearn_after_decoder_fitting}
            np.save(f"{save_dir}/params", param_dict)

            # Save performances
            loss_dict = {'loss_init': loss_init,
                         'loss': loss,
                         'loss_var': loss_var,
                         'loss_exp': loss_exp,
                         'loss_corr': loss_corr,
                         'loss_proj': loss_proj,
                         'loss_vbar': loss_vbar}
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

            # Save rel_proj_var_OM
            np.save(f"{save_dir}/rel_proj_var_OM", rel_proj_var_OM)

            # Save amount of covariability projected along the row space of D
            np.save(f"{save_dir}/A", A)

            # Save candidate perturbations losses
            np.save(f"{save_dir}/candidate_wm_perturbations", wm_total_losses)
            np.save(f"{save_dir}/candidate_om_perturbations", om_total_losses)

            # Save eigenvalues
            np.save(f"{save_dir}/eigenvals_0", eigenvals_0)
            np.save(f"{save_dir}/eigenvals_init", eigenvals_init)
            np.save(f"{save_dir}/eigenvals_after_WMP", eigenvals_after_WMP)
            np.save(f"{save_dir}/eigenvals_after_OMP", eigenvals_after_OMP)

            # Save participation ratios
            np.save(f"{save_dir}/participation_ratio_during_training", p_ratio)


if __name__ == '__main__':
    main()
