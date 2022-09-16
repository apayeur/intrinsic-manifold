import numpy as np
import matplotlib.pyplot as plt
from toy_model import ToyNetwork
plt.style.use('rnn4bci_plot_params.dms')
import copy


def main_np():
    # Parameters
    size = (200, 100, 2)
    input_noise_intensity = 0.
    private_noise_intensity = 1e-3
    lr = 1e-3

    # Pre-training the network
    print('\n|-------------------------------- Initial training --------------------------------|')
    net0 = ToyNetwork('np_initial', size=size,
                      input_noise_intensity=input_noise_intensity,
                      private_noise_intensity=private_noise_intensity,
                      input_subspace_dim=20)
    #net0.load()
    _ = net0.train_with_node_perturbation(lr=(lr, lr, lr), nb_iter=5e3)
    net0.save()
    #cov = net0.compute_covariance()
    #print('\nRatio U_comp_var to total var = {}'.format(np.trace(cov['U_comp_var'])/np.trace(cov['v'])))
    #rint('Ratio U_comp_mean to total var = {}'.format(np.trace(cov['U_comp_mean'])/np.trace(cov['v'])))
    #print('Ratio private noise var to total var = {}\n'.format(np.trace(cov['priv_noise_comp'])/np.trace(cov['v'])))

    # Fit decoder
    print('\n|-------------------------------- Fit decoder --------------------------------|')
    net1 = copy.deepcopy(net0)
    intrinsic_manifold_dim = net1.fit_decoder(threshold=0.99)
    net1.network_name = 'np_fitted'
    net1.plot_sample()

    # Briefly re-trained with decoder
    print('\n|-------------------------------- Re-training with decoder --------------------------------|')
    net2 = copy.deepcopy(net1)
    net2.network_name = 'np_retrained_after_fitted'
    _ = net2.train_with_node_perturbation(lr=(0., lr, 0), nb_iter=1e2)

    # WM perturbation
    print('\n|-------------------------------- WM perturbation --------------------------------|')
    net_wm = copy.deepcopy(net2)
    net_wm.network_name = 'np_wm'
    net_wm.wm_perturb(intrinsic_manifold_dim, nb_samples=int(1e3))
    loss_wm = net_wm.train_with_node_perturbation(lr=(0., 0.1*lr, 0), nb_iter=1e3)

    # OM perturbation
    print('\n|-------------------------------- OM perturbation --------------------------------|')
    net_om = copy.deepcopy(net2)
    net_om.network_name = 'np_om'
    net_om.om_perturb(nb_samples=int(1e3))
    loss_om = net_om.train_with_node_perturbation(lr=(0, 0.1*lr, 0), nb_iter=1e3)

    # Plot loss
    plt.figure(figsize=(2,2/1.25))
    plt.semilogy(loss_wm, label='WM', lw=0.5)
    plt.semilogy(loss_om, label='OM', lw=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Expected loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{net_om.results_folder}/np_Loss.png')
    plt.close()


if __name__ == '__main__':
    main_np()
