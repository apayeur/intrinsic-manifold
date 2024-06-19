from toy_model_new import NonlinearDeterministicNetwork
import numpy as np
import copy
import os

tag = f"fig1-test-nonlinear"
save_dir = f"data/egd/{tag}"
save_dir_results = f"results/egd/{tag}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir_results):
    os.makedirs(save_dir_results)


output_fig_format = 'png'

lr_adapt = 2e-3
nb_iter_adapt = 1e3

net0 = NonlinearDeterministicNetwork(exponent_W=0.5)
#net.plot_output()
net0.train(lr=1.e-2, nb_iter=int(1e3))
net0.plot_output(outfile_name=f"{save_dir_results}/SampleEndInitialTraining.{output_fig_format}")
print("Max abs eigvals W = ", np.max(np.abs(np.linalg.eigvals(net0.W))))

print('\n|-------------------------------- Fit decoder --------------------------------|')
net1 = copy.deepcopy(net0)
intrinsic_manifold_dim, _ = net1.fit_decoder(intrinsic_manifold_dim=6, threshold=0.95)
net1.network_name = 'fitted'
net1.plot_output(outfile_name=f"{save_dir_results}/SampleAfterDecoderFitting.{output_fig_format}")

print('\n|-------------------------------- Select perturbations --------------------------------|')
selected_wm, selected_om, wm_t_l, om_t_l = (
    net1.select_perturb(intrinsic_manifold_dim, nb_om_permuted_units=net1.network_size))
wm_total_losses, om_total_losses = wm_t_l, om_t_l

print('\n|-------------------------------- WM perturbation --------------------------------|')
net_wm = copy.deepcopy(net1)
net_wm.network_name = 'wm'
net_wm.apply_wm_perturb(selected_wm)  # apply WM perturbation

net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMBeforeLearning.{output_fig_format}")

_ = net_wm.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

net_wm.plot_output(outfile_name=f"{save_dir_results}/SampleWMAfterLearning.{output_fig_format}")

print('\n|-------------------------------- OM perturbation --------------------------------|')
net_om = copy.deepcopy(net1)
net_om.network_name = 'om'
net_om.apply_om_perturb(selected_om)  # apply OM perturbation

net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMBeforeLearning.{output_fig_format}")

_ = net_om.train(lr=lr_adapt, nb_iter=nb_iter_adapt)

net_om.plot_output(outfile_name=f"{save_dir_results}/SampleOMAfterLearning.{output_fig_format}")

# Save candidate perturbations losses
np.save(f"{save_dir}/candidate_wm_perturbations", wm_total_losses)
np.save(f"{save_dir}/candidate_om_perturbations", om_total_losses)