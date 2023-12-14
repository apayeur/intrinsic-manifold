"""Figure 1 of paper"""
from toy_model import ToyNetwork
import numpy as np
import copy
import os


def main():
    output_fig_format = 'png'

    # Parameters
    size = (6, 100, 2)              # (input size, recurrent size, output size)
    input_noise_intensity = 0e-4    # set to zero for 1-of-K encoding
    private_noise_intensity = 1e-2
    intrinsic_manifold_dim = 6      # dimension of manifold for control (M)
    lr_init = (0, 1e-2, 0)          # learning rate for initial training
    lr_decoder = (0, 5e-3, 0)       # not used wen `relearn_after_decoder_fitting = False` below
    lr = 0.001                      # learning rate during adaptation
    lr_adapt = (0, lr, 0)
    nb_iter = int(5e2)              # nb of gradient iteration during initial training
    nb_iter_adapt = int(5e2)        # nb of gradient iteration during adaptation
    seed = 0
    relearn_after_decoder_fitting = False
    exponent_W = 0.55               # W_0 ~ N(0, 1/N^exponent_W) -- in the lazy regime for Fig. 1

    # Manage save and load folders
    tag = f"fig1"
    save_dir = f"data/egd/{tag}"
    save_dir_results = f"results/egd/{tag}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    print('\n|-------------------------------- Initial training --------------------------------|')
    net0 = ToyNetwork('initial', size=size,
                      input_noise_intensity=input_noise_intensity,
                      private_noise_intensity=private_noise_intensity,
                      input_subspace_dim=0, nb_inputs=size[0],
                      use_data_for_decoding=False,
                      global_mean_input_is_zero=False,
                      initialization_type='random', exponent_W=exponent_W,
                      rng_seed=seed)

    l, _, _, _, _, _, _, _, _, _ = net0.train(lr=lr_init, nb_iter=nb_iter)

    net0.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleEndInitialTraining.{output_fig_format}")

    print('\n|-------------------------------- Fit decoder --------------------------------|')
    net1 = copy.deepcopy(net0)
    intrinsic_manifold_dim, _ = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim,
                                                                  threshold=0.95)
    net1.network_name = 'fitted'
    net1.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting.{output_fig_format}")

    net2 = copy.deepcopy(net1)
    net2.network_name = 'retrained_after_fitted'
    if relearn_after_decoder_fitting:
        print('\n|-------------------------------- Re-training with decoder --------------------------------|')
        loss_decoder_retraining, _, _, _, _, _, _,_, _, _ = net2.train(lr=lr_decoder, nb_iter=nb_iter//10)
        net2.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleRetrainingWithDecoder.{output_fig_format}")

    print('\n|-------------------------------- Select perturbations --------------------------------|')
    selected_wm, selected_om, wm_t_l, om_t_l = \
        net2.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=size[1])
    wm_total_losses, om_total_losses = wm_t_l, om_t_l

    print('\n|-------------------------------- WM perturbation --------------------------------|')
    net_wm = copy.deepcopy(net2)
    net_wm.network_name = 'wm'
    net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]  # apply WM perturbation

    net_wm.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleWMBeforeLearning.{output_fig_format}")

    l, norm, a_min, a_max, nve, _, A_tmp, f_seed, _, _ = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

    net_wm.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleWMAfterLearning.{output_fig_format}")

    print('\n|-------------------------------- OM perturbation --------------------------------|')
    net_om = copy.deepcopy(net2)
    net_om.network_name = 'om'
    net_om.V = net_om.D @ net_om.C[:, selected_om]  # apply OM perturbation

    net_om.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleOMBeforeLearning.{output_fig_format}")

    l, norm, a_min, a_max, nve, R_seed, _, f_seed, rel_proj_var_OM_seed, _ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

    net_om.plot_sample(sample_size=1000, outfile_name=f"{save_dir_results}/SampleOMAfterLearning.{output_fig_format}")

    # Save candidate perturbations losses
    np.save(f"{save_dir}/candidate_wm_perturbations", wm_total_losses)
    np.save(f"{save_dir}/candidate_om_perturbations", om_total_losses)


if __name__ == '__main__':
    main()
