"""
Description:
-----------
Plot fig 2.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
import seaborn as sns
plt.style.use('rnn4bci_plot_params.dms')

mpl.rcParams['font.size'] = 7

# Set load and save directories
load_dir = "data/egd/exponent_W0.55-lr0.001-M6"
save_fig_dir = "results/egd/exponent_W0.55-lr0.001-M6"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

# Import data
loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss_dict['loss']
loss_corr = loss_dict['loss_corr']
overlap = np.load(f"{load_dir}/normalized_variance_explained.npy", allow_pickle=True).item()
norm_gradW = np.load(f"{load_dir}/norm_gradW.npy", allow_pickle=True).item()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(85*units_convert['mm'], 85*units_convert['mm']/1.25), sharex=True)
# Panel A
m_wm, m_om = np.mean(loss['WM'], axis=0), np.mean(loss['OM'], axis=0)
std_wm, std_om = np.std(loss['WM'], axis=0, ddof=1), np.std(loss['OM'], axis=0, ddof=1)
axes[0,0].plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
axes[0,0].plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
axes[0,0].fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/loss['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/loss['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
axes[0,0].fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/loss['OM'].shape[0]**0.5,
                 m_om + 2*std_om/loss['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)
axes[0,0].set_title('A', loc='left', pad=0, weight='bold', fontsize=8)
axes[0,0].set_yticks([0, 0.2, 0.4, 0.6])
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
# Panel B
for perturbation_type in ['WM', 'OM']:
    mean_ = np.mean(loss_corr[perturbation_type], axis=0)
    std_ = np.std(loss_corr[perturbation_type], axis=0, ddof=1)
    axes[0,1].plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
    axes[0,1].fill_between(np.arange(len(mean_)),
                        mean_ - 2*std_/loss_corr[perturbation_type].shape[0]**0.5,
                        mean_ + 2 * std_ / loss_corr[perturbation_type].shape[0] ** 0.5,
                        color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
axes[0,1].plot(np.arange(len(mean_)), 0.5 * np.ones(len(mean_)), ':', lw=0.5, color='grey')
axes[0,1].set_yticks([0, 0.5])
axes[0,1].set_ylabel('Correlation component\nof the loss')
axes[0,1].set_title('B', loc='left', pad=0, weight='bold', fontsize=8)
# Panel C
for perturbation_type in ['WM', 'OM']:
    m = np.mean(overlap[perturbation_type], axis=0)
    sem = np.std(overlap[perturbation_type], axis=0, ddof=1) / overlap[perturbation_type].shape[0]**0.5
    axes[1,0].plot(np.arange(len(m)), m, label=perturbation_type, color=col_w if perturbation_type == 'WM' else col_o)
    axes[1,0].fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem,
                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
axes[1,0].set_xlabel('Weight update post-perturb.')
axes[1,0].set_ylabel('Manifold overlap')
#axes[1,0].set_xticks([0, 2000, 4000])
axes[1,0].set_yticks([0.9, 0.95, 1])
axes[1,0].set_title('C', loc='left', pad=0, weight='bold', fontsize=8)
# Panel D
m_wm, m_om = np.mean(norm_gradW['loss']['WM'], axis=0), np.mean(norm_gradW['loss']['OM'], axis=0)
std_wm, std_om = np.std(norm_gradW['loss']['WM'], axis=0, ddof=1), np.std(norm_gradW['loss']['OM'], axis=0, ddof=1)
axes[1,1].plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
axes[1,1].plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
axes[1,1].fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/norm_gradW['loss']['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/norm_gradW['loss']['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
axes[1,1].fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/norm_gradW['loss']['OM'].shape[0]**0.5,
                 m_om + 2*std_om/norm_gradW['loss']['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)
axes[1,1].set_ylim([0, 5])
axes[1,1].set_yticks(list(plt.gca().get_ylim()))
#axes[1,1].set_xticks([0, 2000, 4000])
axes[1,1].set_xlabel('Weight update post-perturb.')
axes[1,1].set_ylabel(r'$\|\nabla_W L\|_F$')
axes[1,1].set_title('D', loc='left', pad=0, weight='bold', fontsize=8)
#for ax in axes.ravel():
#    sns.despine(ax=ax)
plt.tight_layout()

for ax in axes.ravel():
    ax.set_xlim([0, 500])
    ax.set_xticks([0, 500])

plt.savefig(f'{save_fig_dir}/fig2.png')
plt.close()
plt.show()

