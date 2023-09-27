"""
Description:
-----------
Plot tr(C_OM @ Var @ C_OM.T) / tr(C_OM @ Var_init @ C_OM.T), the relative projection of covariance on C_OM = C@P_OM
`rel_proj_var_OM`and ratio of projected variance tr(C_OM @ Var @ C_OM.T) / tr(C@ Var @ C.T)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
import seaborn as sns
plt.style.use('rnn4bci_plot_params.dms')

mpl.rcParams['font.size'] = 7

load_dir = "data/egd/test-rich3"
save_fig_dir = "results/egd/test-rich3"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

rel_proj_var_OM = np.load(f"{load_dir}/rel_proj_var_OM.npy")
R = np.load(f"{load_dir}/R.npy")
angles = np.load(f"{load_dir}/principal_angles_min.npy", allow_pickle=True).item()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(114*units_convert['mm'], 114*units_convert['mm']/1.25), sharex=True)
m = np.mean(R, axis=0)
sem = np.std(R, axis=0, ddof=1) / R.shape[0]**0.5
axes[0,0].plot(np.arange(len(m)), m, color=col_o)
axes[0,0].fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem, color=col_o, alpha=0.5, lw=0)
#axes[0,0].set_yticks([0.04, 0.05, 0.06])
axes[0,0].set_ylabel('Ratio of variance projected\nonto $CP_\mathsf{OM}$ and onto $C$')
axes[0,0].set_title('A', loc='left', pad=0, weight='bold', fontsize=8)

m = np.mean(rel_proj_var_OM, axis=0)
sem = np.std(rel_proj_var_OM, axis=0, ddof=1) / rel_proj_var_OM.shape[0]**0.5
axes[0,1].plot(np.arange(len(m)), m, color=col_o)
axes[0,1].fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem, color=col_o, alpha=0.5, lw=0)
axes[0,1].set_ylabel('Variance projected onto $CP_\mathsf{OM}$\nrelative to pre-perturbation')
axes[0,1].set_yticks([1, 50, 100])
axes[0,1].set_ylim([0.5, 55])
axes[0,1].set_title('B', loc='left', pad=0, weight='bold', fontsize=8)

for perturbation_type in ['WM', 'OM']:
    m = np.mean(angles[perturbation_type]['dVar_vs_VT'], axis=0)
    sem = np.std(angles[perturbation_type]['dVar_vs_VT'], axis=0, ddof=1) / angles[perturbation_type]['dVar_vs_VT'].shape[
        0] ** 0.5
    axes[1,0].plot(np.arange(len(m)), m, label=perturbation_type,
                             color=col_w if perturbation_type == 'WM' else col_o)
    axes[1,0].fill_between(np.arange(len(m)),
                                     m - 2 * sem,
                                     m + 2 * sem,
                                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
    axes[1,0].set_ylabel(f'Smallest principal angle $[deg]$\n' + r'between $d\mathbb{V}[\mathbf{v}]$ and $V^\mathsf{T}$')
    axes[1,0].set_xlabel('Weight update post-perturbation')
    axes[1,0].set_ylim([0, 90])
    axes[1,0].set_yticks([0, 45, 90])
    axes[1,0].set_xticks([0, 2000, 4000])
    axes[1,0].set_title('C', loc='left', pad=0, weight='bold', fontsize=8)

for perturbation_type in ['WM', 'OM']:
    m = np.mean(angles[perturbation_type]['UpperVar_vs_VT'], axis=0)
    sem = np.std(angles[perturbation_type]['UpperVar_vs_VT'], axis=0, ddof=1) / angles[perturbation_type]['UpperVar_vs_VT'].shape[
        0] ** 0.5
    axes[1,1].plot(np.arange(len(m)), m, label=perturbation_type,
                             color=col_w if perturbation_type == 'WM' else col_o)
    axes[1,1].fill_between(np.arange(len(m)),
                                     m - 2 * sem,
                                     m + 2 * sem,
                                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
    axes[1,1].set_ylabel(f'Smallest principal angle $[deg]$\n' + r'between dominant modes and $V^\mathsf{T}$')
    axes[1,1].set_xlabel('Weight update post-perturbation')
    axes[1,1].set_ylim([0, 90])
    axes[1,1].set_yticks([0, 45, 90])
    axes[1,1].set_xticks([0, 2000, 4000])
    axes[1,1].set_title('D', loc='left', pad=0, weight='bold', fontsize=8)
    axes[1,1].legend()

#for ax in axes.ravel():
#    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/fig3.png')
plt.close()
plt.show()

