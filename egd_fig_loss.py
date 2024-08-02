import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w, convert_uneven_lists_to_array
import os
plt.style.use('rnn4bci_plot_params.dms')

exponents_W = [1.] #, 0.6, 0.7, 0.8, 0.9, 1]
diff_relative_loss = {exponent_W: [] for exponent_W in exponents_W}
output_fig_format = 'png'
load_dir_suffix = ""  # "-lr0.001-M6-iterAdapt500"

for exponent_W in exponents_W:
    tag = f"pretraining-N100-Nreadouts100-expW{exponent_W}"
    model_type = "egd"
    load_dir = f"data/{model_type}/{tag}"
    save_fig_dir = f"results/{model_type}/{tag}"
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)

    params = np.load(f"{load_dir}/params.npy", allow_pickle=True).item()

    data = np.load(f"{load_dir}/data.npy", allow_pickle=True).item()

    loss = {'WM': data['WM']['loss'], 'OM': data['OM']['loss']}
    loss_init = data['pretraining']['loss']
    #loss_var = loss_dict['loss_var']
    #loss_exp = loss_dict['loss_exp']
    loss_corr = {'WM': data['WM']['loss_corr'], 'OM': data['OM']['loss_corr']}
    #loss_proj = loss_dict['loss_proj']
    #loss_vbar = loss_dict['loss_vbar']

    seeds_to_exclude = [i for i in range(len(loss['OM'])) if np.sum(loss['OM'][i]) < 0]
    seeds_to_include = [i for i in range(params['nb_seeds']) if i not in seeds_to_exclude]

    if isinstance(loss_init, list):
        tmp = [loss_init[s] for s in seeds_to_include]
        loss_init = tmp
        loss_init = convert_uneven_lists_to_array(loss_init)
    else:
        loss_init = loss_init[seeds_to_include]

    for perturbation_type in ['WM', 'OM']:
        if isinstance(loss[perturbation_type], list):
            for s in seeds_to_exclude:
                loss[perturbation_type].pop(s)
            loss[perturbation_type] = convert_uneven_lists_to_array(loss[perturbation_type])
        else:
            loss[perturbation_type] = loss[perturbation_type][seeds_to_include]
        if isinstance(loss_corr[perturbation_type], list):
            for s in seeds_to_exclude:
                loss_corr[perturbation_type].pop(s)
            loss_corr[perturbation_type] = convert_uneven_lists_to_array(loss_corr[perturbation_type])
        else:
            loss_corr[perturbation_type] = loss[perturbation_type][seeds_to_include]

    x_label = 'Weight update post-perturb.' if 'egd' in load_dir else 'Epoch'

    # ------------------------ Loss-related figures ------------------------ #
    support_max = min(loss['OM'].shape[1], loss['WM'].shape[1])
    diff_relative_loss[exponent_W] = (loss['OM'][:, :support_max] / loss['OM'][:, 0:1]
                                      - loss['WM'][:, :support_max] / loss['WM'][:, 0:1])
    # Plot initial loss
    plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    for l in loss_init:
        plt.semilogy(l, color='black', lw=0.5)
    plt.xlabel('Weight update')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/InitialLoss.{output_fig_format}')
    plt.close()
    """
    # Plot adaptation loss for each seed
    nb_seed_with_nonmonotone_learning = {'WM': 0, 'OM': 0}
    plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
    for perturbation_type in ['WM', 'OM']:
        for i in range(loss[perturbation_type].shape[0]):
            perf = loss[perturbation_type][i] / loss[perturbation_type][i, 0]
            if np.any(np.gradient(perf) > 0):
                nb_seed_with_nonmonotone_learning[perturbation_type] += 1
            plt.plot(perf,
                     label=perturbation_type if i == 0 else None,
                     lw=0.4, color=col_w if perturbation_type=='WM' else col_o, alpha=0.3)
        print(f"Proportion of sensitive seeds, {perturbation_type}: "
              f"{nb_seed_with_nonmonotone_learning[perturbation_type]}/{loss[perturbation_type].shape[0]}")
    plt.xlabel(x_label)
    plt.ylabel('$L/L_0$')
    #plt.title(f"Learning rate = {params['lr_adapt'][1]}", pad=0)
    plt.xlim([0, support_max])
    plt.xticks(plt.gca().get_xlim())
    plt.ylim([0,1])
    plt.yticks([0,0.5,1])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/LossAdaptForEachSeed.pdf')
    plt.close()

    # Plot mean +/- 2SEM adaptation loss
    plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
    plot_relative_loss = True
    for perturbation_type in ['WM', 'OM']:
        if plot_relative_loss:
            perf = loss[perturbation_type] / loss[perturbation_type][:,0:1]
            # print(f"Loss {perturbation_type} at update 150", np.mean(perf[:, 150]),
            #       "+/-", 2*np.std(perf[:, 150], ddof=1)/perf.shape[0]**0.5)
            # print(f"Loss {perturbation_type} at update 67", np.mean(perf[:, 67]),
            #       "+/-", 2*np.std(perf[:, 67], ddof=1)/perf.shape[0]**0.5)
        else:
            perf = loss[perturbation_type]
        m = np.mean(perf, axis=0)
        std = np.std(perf, axis=0, ddof=1)
        plt.plot(np.arange(m.shape[0]), m, '-' if perturbation_type=='WM' else '--', label=f"{perturbation_type}, n = {loss[perturbation_type].shape[0]}",
                 color=col_w if perturbation_type=='WM' else col_o, lw=0.5)
        plt.fill_between(np.arange(m.shape[0]),
                         m- 2*std/loss['WM'].shape[0]**0.5,
                         m + 2*std/loss['WM'].shape[0]**0.5,
                         color=col_w if perturbation_type=='WM' else col_o, lw=0, alpha=0.5)
    if plot_relative_loss:
        plt.ylim([0,1])
        plt.yticks([0,0.5,1])
        plt.ylabel('$L/L_0$')
    else:
        plt.ylim([0, 0.55])
        plt.yticks([0, 0.5])
        plt.ylabel('Loss')
    if exponent_W == 0.55 or exponent_W == 0.5:
        plt.gca().text(0.5, 0.9, 'Lazy', ha='center', va='center', transform=plt.gca().transAxes)
    elif exponent_W == 1:
        plt.gca().text(0.5, 0.9, 'Rich', ha='center', va='center', transform=plt.gca().transAxes)
    if exponent_W == 0.55:
         plt.xlim([0, support_max])
         plt.xticks(plt.gca().get_xlim())
    else:
        plt.xlim([0, len(m)])
        plt.xticks([0, len(m)])
    plt.xlabel(x_label)
    plt.legend(loc='center right')
    plt.tight_layout()
    outfile_name = f'{save_fig_dir}/LossAdapt.{output_fig_format}' if not plot_relative_loss else f'{save_fig_dir}/LossAdaptRelative.{output_fig_format}'
    plt.savefig(outfile_name)
    plt.close()
    """
    """
    # Plot subsampled relative performance
    subsampling = nb_iter_adapt // nb_iter_adapt
    shift = nb_iter_adapt // 50  # for clearer plot
    plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    for i in range(loss['WM'].shape[0]):
        perf_wm = 1 - loss['WM'][i] / loss['WM'][i][0]
        perf_om = 1 - loss['OM'][i] / loss['OM'][i][0]
        line_, = plt.plot(np.arange(0 - shift, len(perf_wm) - shift, subsampling), perf_wm[::subsampling],
                          'o-', markersize=3, markeredgewidth=0,
                          label='WM' if i == 0 else None, lw=0., alpha=0.5)
        plt.plot(np.arange(0 + shift, len(perf_om) + shift, subsampling), perf_om[::subsampling],
                 'x-', markersize=3, markeredgewidth=0.5, label='OM' if i == 0 else None,
                 color=line_.get_color(), lw=0., alpha=0.5)
    plt.yticks([0, 1])
    plt.xlabel(x_label)
    plt.ylabel('Retraining performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/SubsampledPerformance.{output_fig_format}')
    plt.close()

    # Plot subsampled relative performance mean +/- SEM
    plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    for perturbation_type in ['WM', 'OM']:
        perf = 1 - loss[perturbation_type] / loss[perturbation_type][:,0:1]
        m, s = np.mean(perf, axis=0), np.std(perf, axis=0, ddof=1)
        # plt.plot(np.arange(1, 1 + loss['WM'].shape[1]), m_wm, color=col_w, zorder=0, label='WM', lw=0.5)
        # plt.plot(np.arange(1, 1 + loss['OM'].shape[1]), m_om, color=col_o, zorder=0, label='OM', lw=0.5)
        # plt.boxplot(loss['WM'][:, ::subsampling], positions=np.arange(1, 1 + loss['WM'].shape[1], subsampling))
        # plt.boxplot(loss['OM'][:, ::subsampling], positions=np.arange(1, 1 + loss['OM'].shape[1], subsampling))
        plt.errorbar(np.arange(0, loss[perturbation_type].shape[1], subsampling), m[::subsampling],
                     yerr=2*s[::subsampling] / s.shape[0]**0.5, fmt='o-',
                     color=col_w if perturbation_type == 'WM' else col_o,
                     markersize=1.5, markeredgewidth=0, label=perturbation_type, lw=0.5)
    plt.yticks([0, 1])
    plt.xlabel(x_label)
    plt.ylabel('Retraining performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/SubsampledPerformance_Stat.{output_fig_format}')
    plt.close()

    # Plot loss components
    fig, (ax_var, ax_exp) = plt.subplots(nrows=2, figsize=(45*units_convert['mm'], 2*45*units_convert['mm']/1.5), sharex=True)
    for perturbation_type in ['WM', 'OM']:
        mean_ = np.mean(loss_var[perturbation_type], axis=0)
        std_ = np.std(loss_var[perturbation_type], axis=0, ddof=1)
        ax_var.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                    label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
        ax_var.fill_between(np.arange(len(mean_)),
                            mean_ - 2*std_/loss_var[perturbation_type].shape[0]**0.5,
                            mean_ + 2 * std_ / loss_var[perturbation_type].shape[0] ** 0.5,
                            color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
    for perturbation_type in ['WM', 'OM']:
        mean_ = np.mean(loss_exp[perturbation_type], axis=0)
        std_ = np.std(loss_exp[perturbation_type], axis=0, ddof=1)
        ax_exp.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                    label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
        ax_exp.fill_between(np.arange(len(mean_)),
                            mean_ - 2*std_/loss_exp[perturbation_type].shape[0]**0.5,
                            mean_ + 2 * std_ / loss_exp[perturbation_type].shape[0] ** 0.5,
                            color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
    ax_var.set_ylabel('Variance component\nof the loss')
    ax_exp.set_ylabel('Expectation component\nof the loss')
    #ax_var.set_xlabel(x_label)
    ax_exp.set_xlabel(x_label)
    ax_var.legend()
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/LossComponents.{output_fig_format}')
    plt.close()

    # Plot loss components correlation and projection
    fig_var, ax_var = plt.subplots(nrows=1, figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    fig_exp, ax_exp = plt.subplots(nrows=1, figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))

    for perturbation_type in ['WM', 'OM']:
        mean_ = np.mean(loss_corr[perturbation_type], axis=0)
        std_ = np.std(loss_corr[perturbation_type], axis=0, ddof=1)
        ax_var.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                    label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
        ax_var.fill_between(np.arange(len(mean_)),
                            mean_ - 2*std_/loss_var[perturbation_type].shape[0]**0.5,
                            mean_ + 2 * std_ / loss_var[perturbation_type].shape[0] ** 0.5,
                            color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
    ax_var.plot(np.arange(len(mean_)), 0.5 * np.ones(len(mean_)), ':', lw=0.5, color='grey')
    for perturbation_type in ['WM', 'OM']:
        mean_ = np.mean(loss_proj[perturbation_type], axis=0)
        std_ = np.std(loss_proj[perturbation_type], axis=0, ddof=1)
        ax_exp.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                    label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
        ax_exp.fill_between(np.arange(len(mean_)),
                            mean_ - 2*std_/loss_exp[perturbation_type].shape[0]**0.5,
                            mean_ + 2 * std_ / loss_exp[perturbation_type].shape[0] ** 0.5,
                            color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
    ax_var.set_ylabel('Correlation component\nof the loss')
    ax_exp.set_ylabel('Projection component\nof the loss')
    ax_var.set_xlabel(x_label)
    ax_exp.set_xlabel(x_label)
    ax_var.legend()
    fig_var.tight_layout()
    fig_exp.tight_layout()
    fig_var.savefig(f'{save_fig_dir}/LossComponentsCorr.{output_fig_format}')
    fig_exp.savefig(f'{save_fig_dir}/LossComponentsProj.{output_fig_format}')
    plt.close()

    # Plot total-covariance loss component vs vbar-related loss component
    plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    ratio = {'WM': loss_vbar['WM'],
             'OM': loss_vbar['OM']}
    m_wm, m_om = np.mean(ratio['WM'], axis=0), np.mean(ratio['OM'], axis=0)
    std_wm, std_om = np.std(ratio['WM'], axis=0, ddof=1), np.std(ratio['OM'], axis=0, ddof=1)
    plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
    plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
    plt.fill_between(np.arange(m_wm.shape[0]),
                     m_wm - 2*std_wm/ratio['WM'].shape[0]**0.5,
                     m_wm + 2*std_wm/ratio['WM'].shape[0]**0.5,
                     color=col_w, lw=0, alpha=0.5)
    plt.fill_between(np.arange(m_om.shape[0]),
                     m_om - 2*std_om/ratio['OM'].shape[0]**0.5,
                     m_om + 2*std_om/ratio['OM'].shape[0]**0.5,
                     color=col_o, lw=0, alpha=0.5)
    plt.xlim([0, 500])
    plt.xticks([0, 500])
    plt.xlabel(x_label)
    plt.ylabel(r'$\frac{1}{2}\|V \bar{\mathbf{v}} \|^2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/Loss_vbar.{output_fig_format}')
    plt.close()


    plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    ratio = {'WM': loss_corr['WM'] - loss_vbar['WM'],
             'OM': loss_corr['OM'] - loss_vbar['OM']}

    m_wm, m_om = np.mean(ratio['WM'], axis=0), np.mean(ratio['OM'], axis=0)
    std_wm, std_om = np.std(ratio['WM'], axis=0, ddof=1), np.std(ratio['OM'], axis=0, ddof=1)
    plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
    plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
    plt.fill_between(np.arange(m_wm.shape[0]),
                     m_wm - 2*std_wm/loss['WM'].shape[0]**0.5,
                     m_wm + 2*std_wm/loss['WM'].shape[0]**0.5,
                     color=col_w, lw=0, alpha=0.5)
    plt.fill_between(np.arange(m_om.shape[0]),
                     m_om - 2*std_om/loss['OM'].shape[0]**0.5,
                     m_om + 2*std_om/loss['OM'].shape[0]**0.5,
                     color=col_o, lw=0, alpha=0.5)

    plt.xlabel(x_label)
    plt.xlim([0, 500])
    plt.xticks([0, 500])
    plt.ylabel(r'$\frac{1}{2}$tr$\{V \mathbb{V}[\bar{\mathbf{v}}] V^\mathsf{T}\}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/Loss_TotalVariance.{output_fig_format}')
    plt.close()

# plot L^{OM}/L_0^{OM}/L^{WM}/L_0^{WM}
regime = []
for exponent_W in exponents_W:
    if exponent_W == 0.55:
        regime.append(" (lazy)")
    elif exponent_W == 1:
        regime.append(" (rich)")
    else:
        regime.append("")
save_fig_dir = f"results/egd/eigvals-study{load_dir_suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)
plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.25))
colors = ['black', 'orange', 'green', 'pink']
i = 0
for expo_i, exponent_W in enumerate(exponents_W):
    if exponent_W not in [0.6, 0.8, 0.9]:
        m = np.mean(diff_relative_loss[exponent_W], axis=0)
        sem = np.std(diff_relative_loss[exponent_W], axis=0, ddof=1) / diff_relative_loss[exponent_W].shape[0]**0.5
        plt.plot(np.arange(m.shape[0]), m, color=colors[i], label=rf"$\alpha$ = {exponent_W}{regime[expo_i]}")
        plt.fill_between(np.arange(m.shape[0]), m - 2*sem,  m + 2*sem, color=colors[i], lw=0, alpha=0.5)
        i += 1
plt.plot(np.arange(m.shape[0]), 0*np.arange(m.shape[0]), ":", color='grey')
#plt.ylabel(r'$L^{(\mathsf{OM})}/L^{(\mathsf{OM})}_0 - L^{(\mathsf{WM})}/L^{(\mathsf{WM})}_0$') #\n(normalized)')
plt.ylabel("Difference of normalized\nlosses (OM $-$ WM)") #\n(normalized)')
plt.xticks([0, 1000, 2000])
plt.xlabel(x_label)
#plt.xlim([0, 2000])
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/DiffLoss.{output_fig_format}')
plt.close()
"""