import numpy as np
import matplotlib.pyplot as plt
from toy_model import ToyNetwork
plt.style.use('rnn4bci_plot_params.dms')
import copy


def main_force():
    # Parameters
    size = (6, 400, 2)
    input_noise_intensity = 0.
    private_noise_intensity = 1e-2
    seeds = np.linspace(27, 50, 10, dtype=int)
    noise_level = 0  # feedback noise
    fraction_silent = 0.5
    delta = 20.
    delta_adapt = delta
    nb_iter = int(1e2)
    nb_adapt_trials = nb_iter
    lambda_ = 1.

    loss_init = []
    loss_wm = []
    loss_om = []

    for seed in seeds:
        print(f'\n|==================================== Seed {seed} =====================================|')
        print('\n|-------------------------------- Initial training --------------------------------|')
        net0 = ToyNetwork('force_initial', size=size,
                          input_noise_intensity=input_noise_intensity,
                          private_noise_intensity=private_noise_intensity,
                          nb_inputs=size[0], use_data_for_decoding=True,
                          input_subspace_dim=6, rng_seed=seed, sparsity_factor=0.2)
        #net0.plot_sample(sample_size=1000, outfile_name=None)

        l = net0.train_with_force(nb_iter=nb_iter, delta=delta, lambda_=lambda_, noise_level=0., fraction_silent=0)
        loss_init.append(l)
        #net0.plot_sample(sample_size=1000, outfile_name=None)
        mean_p = 0.
        for p in net0.P:
            mean_p += np.trace(p) / p.shape[0]
        print(f"Mean p = {mean_p/len(net0.P)}")

        # Fit decoder
        print('\n|-------------------------------- Fit decoder --------------------------------|')
        net1 = copy.deepcopy(net0)
        intrinsic_manifold_dim = net1.fit_decoder(intrinsic_manifold_dim=10, threshold=0.99)
        net1.network_name = 'force_fitted'
        #net1.plot_sample()

        print('\n|-------------------------------- Re-training with decoder --------------------------------|')
        net2 = copy.deepcopy(net1)
        net2.network_name = 'force_retrained_after_fitted'
        #_ = net2.train_with_force(nb_iter=nb_iter // 10, delta=delta, lambda_=lambda_, noise_level=0., fraction_silent=0)

        selected_wm, selected_om = net2.select_perturb(intrinsic_manifold_dim,
                                                       nb_om_permuted_units=size[1],
                                                       nb_samples=int(1e4))

        print('\n|-------------------------------- WM perturbation --------------------------------|')
        net_wm = copy.deepcopy(net2)
        net_wm.network_name = 'wm'
        net_wm.V = net_wm.D @ net_wm.C[selected_wm, :]
        #net_wm.plot_sample(sample_size=1000, outfile_name=None)
        l_wm = net_wm.train_with_force(nb_iter=nb_adapt_trials, delta=delta_adapt, lambda_=lambda_,
                                       noise_level=noise_level, fraction_silent=fraction_silent)
        loss_wm.append(l_wm)

        print('\n|-------------------------------- OM perturbation --------------------------------|')
        net_om = copy.deepcopy(net2)
        net_om.network_name = 'force_om'
        net_om.V = net_om.D @ net_om.C[:, selected_om]
        l_om = net_om.train_with_force(nb_iter=nb_adapt_trials, delta=delta_adapt,  lambda_=lambda_,
                                       noise_level=noise_level, fraction_silent=fraction_silent)
        loss_om.append(l_om)

    # Plot initial loss
    plt.figure(figsize=(2, 2 / 1.25))
    for i, seed in enumerate(seeds):
        plt.semilogy(loss_init[i], color='black', lw=0.5)
    plt.xlabel('Weight updates')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'LossInitEGD.png')
    plt.close()

    # Plot adaptation loss
    plt.figure(figsize=(2, 2 / 1.25))
    for i, seed in enumerate(seeds):
        plt.plot(loss_wm[i], loss_om[i], '-', color='black', lw=0.5)
    min_ = min(np.min(loss_wm), np.min(loss_om))
    max_ = max(np.max(loss_wm), np.max(loss_om))
    plt.plot([min_, max_], [min_, max_], ':', color='grey', lw=0.25)
    plt.xlim([0, 0.55])
    plt.ylim([0, 0.55])
    plt.xticks([0, 0.5])
    plt.yticks([0, 0.5])
    plt.gca().set_aspect('equal')
    plt.xlabel('Loss WM')
    plt.ylabel('Loss OM')
    plt.tight_layout()
    plt.savefig(f'LossOMvsWM.png')
    plt.close()

    plt.figure(figsize=(2, 2 / 1.25))
    for i, seed in enumerate(seeds):
        line_, = plt.semilogy(loss_wm[i], label='WM' if seed == seeds[0] else None, lw=0.5)
        plt.semilogy(loss_om[i], '--', label='OM' if seed == seeds[0] else None, color=line_.get_color(), lw=0.5)
    plt.xlabel('Weight update')
    plt.ylabel('Expected loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'LossAdapt.png')
    plt.close()

    # Plot relative loss
    plt.figure(figsize=(2, 2 / 1.25))
    subsampling = nb_adapt_trials // 10
    relative_loss = []
    for i, seed in enumerate(seeds):
        relative_loss.append(np.array(loss_om[i]) - np.array(loss_wm[i]))
    relative_loss = np.array(relative_loss)
    relative_loss = relative_loss[:, ::subsampling]
    #m = np.mean(relative_loss, axis=0)
    #std = np.std(relative_loss, axis=0, ddof=1)
    plt.boxplot(relative_loss, labels=np.arange(0, len(loss_om[0]), subsampling), notch=True)
    plt.xlabel('Trial post-perturbation')
    plt.ylabel(r'Loss$_\mathrm{OM}$ - Loss$_\mathrm{WM}$')
    plt.tight_layout()
    plt.savefig(f'RelativeLossEvery{subsampling}_FractionSilent{fraction_silent}.png')
    plt.close()

    # Plot relative performance every 500 iterations
    subsampling = nb_adapt_trials // 10
    shift = subsampling // 5  # for clearer plot
    plt.figure(figsize=(2, 2 / 1.25))
    for i, seed in enumerate(seeds):
        loss_wm[i] = 1 - np.array(loss_wm[i]) / loss_wm[i][0]
        loss_om[i] = 1 - np.array(loss_om[i]) / loss_om[i][0]
        line_, = plt.plot(np.arange(0 - shift, len(loss_wm[i]) - shift, subsampling), loss_wm[i][::subsampling],
                          'o-', markersize=3, markeredgewidth=0,
                          label='WM' if seed == seeds[0] else None, lw=0., alpha=0.5)
        plt.plot(np.arange(0 + shift, len(loss_om[i]) + shift, subsampling), loss_om[i][::subsampling],
                 'x-', markersize=3, markeredgewidth=0.5, label='OM' if seed == seeds[0] else None,
                 color=line_.get_color(), lw=0., alpha=0.5)
    plt.yticks([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Weight update post-perturbation')
    plt.ylabel('Retraining performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Loss_Every{subsampling}_FractionSilent{fraction_silent}.png')
    plt.close()

    # Plot stat every 500 iterations
    plt.figure(figsize=(2, 2 / 1.25))
    m_wm, m_om = np.mean(loss_wm, axis=0), np.mean(loss_om, axis=0)
    std_wm, std_om = np.std(loss_wm, axis=0, ddof=1), np.std(loss_om, axis=0, ddof=1)
    plt.errorbar(np.arange(0, len(loss_wm[0]), subsampling), m_wm[::subsampling],
                 yerr=std_wm[::subsampling], fmt='o-', markersize=3, markeredgewidth=0, label='WM', lw=0.5)
    plt.errorbar(np.arange(0, len(loss_om[0]), subsampling), m_om[::subsampling],
                 yerr=std_om[::subsampling], fmt='s-', markersize=3, markeredgewidth=0, label='OM', lw=0.5)
    plt.yticks([0, 1])
    plt.ylim([0, 1])

    plt.xlabel('Weight update post-perturbation')
    plt.ylabel('Retraining performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Loss_Every{subsampling}_FractionSilent{fraction_silent}_Stat.png')
    plt.close()



if __name__ == '__main__':
    main_force()
