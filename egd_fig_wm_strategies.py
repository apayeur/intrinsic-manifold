"""
Description:
-----------
Plot the amount of covariability projected along the row space of D, denoted A.
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd-high-dim-input/plasticity-in-U-only-lower-lr"
save_fig_dir = "results/egd-high-dim-input/plasticity-in-U-only-lower-lr"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

A = np.load(f"{load_dir}/A.npy", allow_pickle=True).item()
labels = {'D': 'Original', 'DP_WM': 'Perturbed'}

selected_adaptation_times = np.array([100, 250,  500])  # np.array([200, 500, 1000, 2500, 5000])



_, axes = plt.subplots(nrows=2, figsize=(45*units_convert['mm'], 2*45*units_convert['mm']/1.25), sharex=True, sharey=True)
axes_dict = {'D': axes[0], 'DP_WM': axes[1]}
for projection_type in ['D', 'DP_WM']:
    relative_A = 100*(A[projection_type] - A[projection_type][:,[0]]) / A[projection_type][:,[0]]
    m = np.median(relative_A, axis=0)
    sem = np.std(relative_A, axis=0, ddof=1) / relative_A.shape[0]**0.5
    vp = axes_dict[projection_type].violinplot(relative_A[:, selected_adaptation_times-1],
                        positions=selected_adaptation_times,
                        showmedians=True, widths=50)
    for pc in vp['bodies']:
        pc.set_facecolor('grey' if projection_type == 'D' else col_w)
        pc.set_edgecolor(None)
        pc.set_linewidth(0.5)
        pc.set_alpha(0.5)
    vp['cbars'].set_color('grey' if projection_type == 'D' else col_w)
    vp['cmaxes'].set_color('grey' if projection_type == 'D' else col_w)
    vp['cmins'].set_color('grey' if projection_type == 'D' else col_w)
    vp['cmedians'].set_color('grey' if projection_type == 'D' else col_w)
    axes_dict[projection_type].set_title(labels[projection_type], pad=2)

    #for i in range(relative_A.shape[0]):
    #    plt.plot(np.arange(len(m)), relative_A[i], color='black' if projection_type == 'D' else (0.9, 0.6, 0), lw=0.2)
    #plt.plot(np.arange(len(m)), m, '-' if projection_type == 'D' else '--',
    #         label=labels[projection_type], color='black' if projection_type == 'D' else (0.9, 0.6, 0))
    #plt.fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem,
    #                 color='black' if projection_type == 'D' else (0.9, 0.6, 0), alpha=0.5, lw=0)
axes[1].set_xlabel('Weight update post-perturbation')
axes[0].set_ylabel('Change in decoder-projected\ncovariability (%)')
axes[1].set_ylabel('Change in decoder-projected\ncovariability (%)')
axes[0].set_ylim(ymax=1600)
axes[1].set_ylim(ymax=1600)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/Strategies.png')
plt.close()
