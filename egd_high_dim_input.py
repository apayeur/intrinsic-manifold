from toy_model import ToyNetwork
import numpy as np
import copy
import os

def main():
    # Parameters
    size = (200, 100, 2)
    input_noise_intensity = 5e-4
    private_noise_intensity = 5e-4
    nb_inputs = 6
    intrinsic_manifold_dim = 6 #15  # 6
    input_subspace_dim = size[0]//10
    lr = 5e-3  # was 10e-3
    lr_init = (lr, lr, 0)
    lr_decoder = (lr, lr, 0)
    lr_adapt = (0, 1e-3, 0) #(0, 2e-3, 0)  # (50e-3, 0, 0)
    nb_iter = int(10e3)
    nb_iter_adapt = int(2e3)  # was 5e3
    seeds = np.arange(20, dtype=int)
    relearn_after_decoder_fitting = False

    if lr_adapt[0] > 0 and abs(lr_adapt[1]) < 1e-7:
        prefix = "plasticity-in-U-only"
    elif lr_adapt[1] > 0 and abs(lr_adapt[0]) < 1e-7:
        prefix = "plasticity-in-W-only"
    elif lr_adapt[0] > 0 and lr_adapt[1]:
        prefix = "plasticity-in-U-and-W"

    tag = f"{prefix}-M{intrinsic_manifold_dim}-lrU{lr_adapt[0]}-lrW{lr_adapt[1]}"  # identification of this experiment, for bookkeeping
    save_dir = f"data/egd-high-dim-input/{tag}"
    save_dir_results = f"results/egd-high-dim-input/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

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

    # Norm of grad W
    norm_gradW = {'loss':{'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                          'OM': np.empty(shape=(len(seeds), nb_iter_adapt))},
                  'loss_tot_var': {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                               'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}}

    # Participation ratio of initial network
    participation_ratios = {'input': np.empty(len(seeds)), 'recurrent': np.empty(len(seeds))}

    # Normalized variance explained
    normalized_variance_explained = {'WM': np.empty(shape=(len(seeds), nb_iter_adapt)),
                                     'OM': np.empty(shape=(len(seeds), nb_iter_adapt))}

    # Ratio of projected variance (OM)
    R = np.empty(shape=(len(seeds), nb_iter_adapt))

    # Amount of covariability projected along the row space of D
    A = {'D': np.empty(shape=(len(seeds), nb_iter_adapt)),
         'DP_WM': np.empty(shape=(len(seeds), nb_iter_adapt))}

    # Candidate perturbation losses
    wm_total_losses, om_total_losses = None, None

    # Eigenvalues
    eigenvals_init = []
    eigenvals_after_WMP = []
    eigenvals_after_OMP = []

    for seed_id, seed in enumerate(seeds):
        print(f'\n|==================================== Seed {seed} =====================================|')
        print('\n|-------------------------------- Initial training --------------------------------|')
        net0 = ToyNetwork('initial', size=size,
                          input_noise_intensity=input_noise_intensity,
                          private_noise_intensity=private_noise_intensity,
                          input_subspace_dim=input_subspace_dim, nb_inputs=nb_inputs,
                          use_data_for_decoding=False,
                          global_mean_input_is_zero=False,
                          orthogonalize_input_means=False,
                          initialization_type='random',
                          rng_seed=seed)
        print("Total variance components")
        cov = net0.compute_covariance()
        print(f"Var component: {np.trace(cov['U_comp_var'])}")
        print(f"Mean component: {np.trace(cov['U_comp_mean'])}")
        print(f"Private component: {np.trace(cov['priv_noise_comp'])}")

        if seed_id == 0:
            net0.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleBeforeInitialTraining.png")

        l, _, _, _, _, _, _, _, _ = net0.train(lr=lr_init, nb_iter=nb_iter)
        eigval_init = np.max(np.abs(np.linalg.eigvals(net0.W)))
        print(f"Max eigenvalue of W: {eigval_init}")
        eigenvals_init.append(eigval_init)

        loss_init[seed_id] = l['total']

        participation_ratios['input'][seed_id] = net0.participation_ratio_input()
        participation_ratios['recurrent'][seed_id] = net0.participation_ratio()

        print(f"\nParticipation ratio for recurrent activity = {participation_ratios['recurrent'][seed_id]}")
        print(f"Dimensionality for recurrent activity = {net0.dimensionality(threshold=0.95)}")
        print(f"Participation ratio for input = {participation_ratios['input'][seed_id]}")
        print(f"Dimensionality for input = {net0.dimensionality_input(threshold=0.95)}")

        if seed_id == 0:
            net0.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleEndInitialTraining.png")


        print('\n|-------------------------------- Fit decoder --------------------------------|')
        net1 = copy.deepcopy(net0)
        intrinsic_manifold_dim, real_dims[seed_id] = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim,
                                                                      threshold=0.99)
        net1.network_name = 'fitted'
        if seed_id == 0:
            net1.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting.png")


        print('\n|-------------------------------- Re-training with decoder --------------------------------|')
        net2 = copy.deepcopy(net1)
        net2.network_name = 'retrained_after_fitted'
        if relearn_after_decoder_fitting:
            loss_decoder_retraining, _, _, _, _, _, _, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)
            if seed_id == 0:
                net2.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder.png")


        print('\n|-------------------------------- Select perturbations --------------------------------|')
        selected_wm, selected_om, wm_t_l, om_t_l = \
            net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1], nb_samples=int(1e4))
        if seed_id == 0:
            wm_total_losses, om_total_losses = wm_t_l, om_t_l


        print('\n|-------------------------------- WM perturbation --------------------------------|')
        net_wm = copy.deepcopy(net2)
        net_wm.network_name = 'wm'
        net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]
        if seed_id == 0:
            net_wm.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleWMBeforeLearning.png")

        l, norm, a_min, a_max, nve, _, A_tmp, _, _ = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

        if seed_id == 0:
            net_wm.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleWMAfterLearning.png")

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

        for key in min_angles['WM'].keys():
            min_angles['WM'][key][seed_id] = a_min[key]
            max_angles['WM'][key][seed_id] = a_max[key]

        eigval_after_WM = np.max(np.abs(np.linalg.eigvals(net_wm.W)))
        print(f"Max eigenvalue of W: {eigval_after_WM}")
        eigenvals_after_WMP.append(eigval_after_WM)

        print('\n|-------------------------------- OM perturbation --------------------------------|')
        net_om = copy.deepcopy(net2)
        net_om.network_name = 'om'
        net_om.V = net_om.D @ net_om.C[:, selected_om]

        if seed_id == 0:
            net_om.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleOMBeforeLearning.png")

        l, norm, a_min, a_max, nve, R_seed, _, _, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

        if seed_id == 0:
            net_om.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleOMAfterLearning.png")

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
        for key in min_angles['OM'].keys():
            min_angles['OM'][key][seed_id] = a_min[key]
            max_angles['OM'][key][seed_id] = a_max[key]

        eigval_after_OM = np.max(np.abs(np.linalg.eigvals(net_om.W)))
        print(f"Max eigenvalue of W: {eigval_after_OM}")
        eigenvals_after_OMP.append(eigval_after_OM)

    # Save parameters
    param_dict = {'size': size,
                  'nb_seeds': len(seeds),
                  'input_noise_intensity': input_noise_intensity,
                  'private_noise_intensity': private_noise_intensity,
                  'lr': lr, 'lr_init': lr_init, 'lr_decoder': lr_decoder,'lr_adapt': lr_adapt,
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

    # Save normalized variance explained
    np.save(f"{save_dir}/normalized_variance_explained", normalized_variance_explained)

    # Save ratio of projected covariability
    np.save(f"{save_dir}/R", R)

    # Save amount of covariability projected along the row space of D
    np.save(f"{save_dir}/A", A)

    # Save candidate perturbations losses
    np.save(f"{save_dir}/candidate_wm_perturbations", wm_total_losses)
    np.save(f"{save_dir}/candidate_om_perturbations", om_total_losses)

    # Save participation ratios
    np.save(f"{save_dir}/participation_ratios", participation_ratios)

    # Save eigenvalues
    np.save(f"{save_dir}/eigenvals_init", eigenvals_init)
    np.save(f"{save_dir}/eigenvals_after_WMP", eigenvals_after_WMP)
    np.save(f"{save_dir}/eigenvals_after_OMP", eigenvals_after_OMP)


if __name__ == '__main__':
    main()
