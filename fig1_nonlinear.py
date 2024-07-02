from nonlinear_model import NonlinearDeterministicNetwork
import numpy as np
import copy
import os

tag = f"fig1-linear-nonlinear-new-decoder"
save_dir = f"data/egd/{tag}"
save_dir_results = f"results/egd/{tag}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir_results):
    os.makedirs(save_dir_results)


output_fig_format = 'png'

# Parameters
size = (6, 100, 2)              # (input size, recurrent size, output size)
intrinsic_manifold_dim = 6      # dimension of manifold for control (M)
lr_init = 5e-2 #3e-2                  # learning rate for initial training
lr_decod = lr_init / 2
lr = 10e-3 #0.1e-2                       # learning rate during adaptation
nb_iter = int(500)              # nb of gradient iteration during initial training
nb_iter_adapt = int(5e2)        # nb of gradient iteration during adaptation
seed = 0
exponent_W = 0.55        # W_0 ~ N(0, 1/N^exponent_W)
activation_function = 'relu'
nb_readouts = size[1]

relearn_after_decoder_fitting = True
do_record_data = False
do_z_score = True
global_mean_input_is_zero = False
fit_intercept = True


# Create network
net0 = NonlinearDeterministicNetwork(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0], exponent_W=exponent_W,
                                     global_mean_input_is_zero=global_mean_input_is_zero,
                                     do_z_score=do_z_score, rng_seed=seed,
                                     activation_function=activation_function)

data = net0.train(lr=lr_init, nb_iter=nb_iter, do_record_data=do_record_data)
net0.plot_output(outfile_name=f"{save_dir_results}/SampleEndInitialTraining.{output_fig_format}")
print("Max abs eigvals W = ", np.max(np.abs(np.linalg.eigvals(net0.W))))

"""
print('\n|-------------------------------- Fit decoder --------------------------------|')
net1 = copy.deepcopy(net0)
intrinsic_manifold_dim, _ = net1.fit_decoder(intrinsic_manifold_dim=intrinsic_manifold_dim,
                                             threshold=0.95, fit_intercept=fit_intercept)
net1.plot_output(outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting_seed{seed}.{output_fig_format}")

'---------------------------------------- Retraining with decoder ----------------------------------------'
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
wm_total_losses, om_total_losses = wm_t_l, om_t_l

print('\n|-------------------------------- WM perturbation --------------------------------|')
net_wm = copy.deepcopy(net2)
net_wm.apply_wm_perturb(selected_wm)  # apply WM perturbation

net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMBeforeLearning_seed{seed}.{output_fig_format}")

_ = net_wm.train(lr=lr, nb_iter=nb_iter_adapt)

net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMAfterLearning_seed{seed}.{output_fig_format}")

print('\n|-------------------------------- OM perturbation --------------------------------|')
net_om = copy.deepcopy(net2)
net_om.apply_om_perturb(selected_om)  # apply OM perturbation

net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMBeforeLearning_seed{seed}.{output_fig_format}")

_ = net_om.train(lr=lr, nb_iter=nb_iter_adapt)

net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMAfterLearning_seed{seed}.{output_fig_format}")

# Save candidate perturbations losses
np.save(f"{save_dir}/candidate_wm_perturbations", wm_total_losses)
np.save(f"{save_dir}/candidate_om_perturbations", om_total_losses)

"""