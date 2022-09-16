import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/final_low_lr"
save_fig_dir = "results/egd/final_low_lr"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss_dict['loss_corr']

types_of_principal_angle = ["min", "max"]

for type_of_principal_angle in types_of_principal_angle:
    angles = np.load(f"{load_dir}/principal_angles_{type_of_principal_angle}.npy", allow_pickle=True).item()

    name_mapping = {'dVar_vs_VT': r'$d\mathbb{V}~[\mathbf{v}]$ vs $V^\mathsf{T}$',
                    'UpperVar_vs_VT': r'First 5 PCs vs $V^\mathsf{T}$',
                    'LowerVar_vs_VT': r'Last PCs vs $V^\mathsf{T}$',
                    'UpperVar_vs_VarBCI': r'$\mathbb{V}~[\mathbf{v}]$ vs $\mathbb{V}~[\mathbf{v}]_\mathsf{BCI}$'}
    plot_label = {'dVar_vs_VT': 'A',
                    'UpperVar_vs_VT': 'B',
                    'LowerVar_vs_VT': 'C',
                    'UpperVar_vs_VarBCI': 'D'}
    nb_subplots = len(angles['WM'].items())
    ncols = nb_subplots // 2 if nb_subplots % 2==0 else nb_subplots //2+1
    fig, axes = plt.subplots(nrows=2, ncols=nb_subplots // 2 if nb_subplots % 2==0 else nb_subplots //2+1,
                           figsize=(45*units_convert['mm']*ncols, 45*units_convert['mm']*2), sharex=True)
    ax_dict = dict()
    axes = axes.ravel()
    for i, key in enumerate(angles['WM'].keys()):
        ax_dict.update({key: axes[i]})

    for perturbation_type in ['WM', 'OM']:
        for angle_name in ax_dict.keys():
            for i in range(angles[perturbation_type][angle_name].shape[0]):
                ax_dict[angle_name].plot(angles[perturbation_type][angle_name][i],
                                             color=col_w if perturbation_type == 'WM' else col_o, lw=0.5, alpha=0.25)
            m = np.mean(angles[perturbation_type][angle_name], axis=0)
            sem = np.std(angles[perturbation_type][angle_name], axis=0, ddof=1) / angles[perturbation_type][angle_name].shape[0]**0.5
            ax_dict[angle_name].plot(np.arange(len(m)), m, label=perturbation_type,
                                            color=col_w if perturbation_type == 'WM' else col_o)
            ax_dict[angle_name].fill_between(np.arange(len(m)),
                               m - 2 * sem,
                               m + 2 * sem,
                               color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
        # mean_dL = np.mean(np.gradient(loss[perturbation_type], axis=1), axis=0)
        # mean_L = np.mean(1 - loss[perturbation_type] / loss[perturbation_type][:,[0]], axis=0)
        # sem_L = np.std(1 - loss[perturbation_type] / loss[perturbation_type][:,[0]], axis=0) / loss[perturbation_type].shape[0]**0.5
        # axes[-1].plot(np.arange(len(mean_L)), mean_L, color=col_w if perturbation_type == 'WM' else col_o)
        # axes[-1].fill_between(np.arange(len(mean_L)),
        #                                  mean_L - 2 * sem_L,
        #                                  mean_L + 2 * sem_L,
        #                                  color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
    for angle_name in ax_dict.keys():
        ax_dict[angle_name].set_title(f'{plot_label[angle_name]}            {name_mapping[angle_name]}', loc='left')
        #ax_dict[angle_name].set_ylim(ymin=0)
        ax_dict[angle_name].legend()
    if type_of_principal_angle == "min":
        axes[0].set_xlabel(f'Smallest principal angle $[deg]$')
        axes[ncols].set_xlabel(f'Smallest principal angle $[deg]$')
    else:
        axes[0].set_xlabel(f'Largest principal angle $[deg]$')
        axes[ncols].set_xlabel(f'Largest principal angle $[deg]$')
    for i in range(ncols):
        axes[ncols+i].set_ylabel('Correlation-related loss')
    #axes[-1].set_ylabel(f'Change in largest principal angle $[deg]$')

    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/PrincipalAngles'
                f'{type_of_principal_angle[0].capitalize()}{type_of_principal_angle[1:]}.png')
    plt.close()