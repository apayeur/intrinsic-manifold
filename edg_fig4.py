"""
Description:
-----------
New fig. 4
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')


load_dir = "data/egd/exponent_W0.55-lr0.001-M6-iterAdapt500"
save_fig_dir = "results/egd/exponent_W0.55-lr0.001-M6-iterAdapt500"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

# Load data
loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss_dict['loss']
loss_corr = loss_dict['loss_corr']
overlap = np.load(f"{load_dir}/normalized_variance_explained.npy", allow_pickle=True).item()
rel_proj_var_OM = np.load(f"{load_dir}/rel_proj_var_OM.npy")
R = np.load(f"{load_dir}/R.npy")
angles = np.load(f"{load_dir}/principal_angles_min.npy", allow_pickle=True).item()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(85*units_convert['mm'], 85*3/2*units_convert['mm']/1.25), sharex=True)
# Panel A
for perturbation_type in ['WM', 'OM']:
    mean_ = np.mean(loss_corr[perturbation_type], axis=0)
    std_ = np.std(loss_corr[perturbation_type], axis=0, ddof=1)
    axes[0,0].plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
    axes[0,0].fill_between(np.arange(len(mean_)),
                        mean_ - 2*std_/loss_corr[perturbation_type].shape[0]**0.5,
                        mean_ + 2 * std_ / loss_corr[perturbation_type].shape[0] ** 0.5,
                        color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
axes[0,0].plot(np.arange(len(mean_)), 0.5 * np.ones(len(mean_)), ':', lw=0.5, color='grey')
axes[0,0].set_yticks([0, 0.5])
axes[0,0].set_ylabel('Correlation component\nof the loss')
axes[0,0].set_title('A', loc='left', pad=0, weight='bold', fontsize=8)
# Panel B
for perturbation_type in ['WM', 'OM']:
    m = np.mean(overlap[perturbation_type], axis=0)
    sem = np.std(overlap[perturbation_type], axis=0, ddof=1) / overlap[perturbation_type].shape[0]**0.5
    axes[0,1].plot(np.arange(len(m)), m, label=perturbation_type, color=col_w if perturbation_type == 'WM' else col_o)
    axes[0,1].fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem,
                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
axes[0,1].set_ylabel('Manifold overlap')
#axes[1,0].set_xticks([0, 2000, 4000])
axes[0,1].set_yticks([0.9, 0.95, 1])
axes[0,1].set_title('B', loc='left', pad=0, weight='bold', fontsize=8)

# Panel C
m = np.mean(R, axis=0)
sem = np.std(R, axis=0, ddof=1) / R.shape[0]**0.5
axes[1,0].plot(np.arange(len(m)), m, color=col_o)
#axes[1,0].set_xlim(0,500)
axes[1,0].fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem, color=col_o, alpha=0.5, lw=0)
axes[1,0].set_ylabel('Ratio of variance projected\nonto $CP_\mathsf{OM}$ and onto $C$')
axes[1,0].set_yticks([0.05, 0.1])
axes[1,0].set_title('C', loc='left', pad=0, weight='bold', fontsize=8)

# Panel D
m = np.mean(rel_proj_var_OM, axis=0)
sem = np.std(rel_proj_var_OM, axis=0, ddof=1) / rel_proj_var_OM.shape[0]**0.5
axes[1,1].plot(np.arange(len(m)), m, color=col_o)
axes[1,1].fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem, color=col_o, alpha=0.5, lw=0)
axes[1,1].set_ylabel('Variance projected onto $CP_\mathsf{OM}$\nrelative to pre-perturbation')
axes[1,1].set_yticks([1, 20])
#axes[0,1].set_ylim([0.5, 55])
axes[1,1].set_title('D', loc='left', pad=0, weight='bold', fontsize=8)

# Panel E
for perturbation_type in ['WM', 'OM']:
    m = np.mean(angles[perturbation_type]['dVar_vs_VT'], axis=0)
    sem = np.std(angles[perturbation_type]['dVar_vs_VT'], axis=0, ddof=1) / angles[perturbation_type]['dVar_vs_VT'].shape[
        0] ** 0.5
    axes[2,0].plot(np.arange(len(m)), m, label=perturbation_type,
                             color=col_w if perturbation_type == 'WM' else col_o)
    axes[2,0].fill_between(np.arange(len(m)),
                                     m - 2 * sem,
                                     m + 2 * sem,
                                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
    axes[2,0].set_ylabel(f'Smallest principal angle $[deg]$\n' + r'between $d\mathbb{V}[\mathbf{v}]$ and $V^\mathsf{T}$')
    axes[2,0].set_xlabel('Weight update post-perturbation')
    axes[2,0].set_ylim([0, 90])
    axes[2,0].set_yticks([0, 45, 90])
    axes[2,0].set_xticks([0, 500])
    axes[2,0].set_title('E', loc='left', pad=0, weight='bold', fontsize=8)

# Panel F
for perturbation_type in ['WM', 'OM']:
    m = np.mean(angles[perturbation_type]['UpperVar_vs_VT'], axis=0)
    sem = np.std(angles[perturbation_type]['UpperVar_vs_VT'], axis=0, ddof=1) / angles[perturbation_type]['UpperVar_vs_VT'].shape[
        0] ** 0.5
    axes[2,1].plot(np.arange(len(m)), m, label=perturbation_type,
                            color=col_w if perturbation_type == 'WM' else col_o)
    axes[2,1].fill_between(np.arange(len(m)),
                                    m - 2 * sem,
                                    m + 2 * sem,
                                    color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
    axes[2,1].set_ylabel(f'Smallest principal angle $[deg]$\n' + r'between dominant modes and $V^\mathsf{T}$')
    axes[2,1].set_xlabel('Weight update post-perturbation')
    axes[2,1].set_ylim([0, 90])
    axes[2,1].set_yticks([0, 45, 90])
    #axes21,1].set_xticks([0, 2000, 4000])
    axes[2,1].set_title('F', loc='left', pad=0, weight='bold', fontsize=8)
    axes[2,1].legend()

#for ax in axes.ravel():
#    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/fig4.png')
plt.close()
plt.show()

