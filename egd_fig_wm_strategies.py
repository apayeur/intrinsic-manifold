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

# Functions
def find_adaptation_time(loss, loss_fractions):
    """Find adaptation times that correspond to the prescribed loss fractions."""
    adaptation_times = []
    for loss_frac in loss_fractions:
        relative_loss = np.abs(loss - (1-loss_frac) * loss[0])
        adaptation_times.append(np.argmin(relative_loss))
    return adaptation_times


# Loading data
load_dir = "data/egd-high-dim-input/plasticity-in-U-only-final-M6-matching-params"
save_fig_dir = "results/egd-high-dim-input/plasticity-in-U-only-final-M6-matching-params"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

A = np.load(f"{load_dir}/A.npy", allow_pickle=True).item()
loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss_wm = loss_dict['loss']['WM']
labels = {'D': 'Original', 'DP_WM': 'Perturbed'}

# Prescribed loss fractions where to evaluate change in decoder projected covariability
loss_fractions = [0.5, 0.75, 0.9]

_, axes = plt.subplots(nrows=2, figsize=(45*units_convert['mm'], 2*45*units_convert['mm']/1.25), sharex=True, sharey=True)
axes_dict = {'D': axes[0], 'DP_WM': axes[1]}
for projection_type in ['D', 'DP_WM']:
    adapt_times = []
    relative_A = np.empty((loss_wm.shape[0], len(loss_fractions)))
    for seed in range(loss_wm.shape[0]):
        print(len(loss_wm[seed]))
        adapt_times = find_adaptation_time(loss_wm[seed], loss_fractions)
        print(adapt_times)
        relative_A[seed] = 100 * (A[projection_type][seed][adapt_times] - A[projection_type][seed,0]) / A[projection_type][seed,0]
    m = np.median(relative_A, axis=0)
    print(f"{projection_type}: Median = {m}")
    sem = np.std(relative_A, axis=0, ddof=1) / relative_A.shape[0]**0.5
    vp = axes_dict[projection_type].violinplot(relative_A,
                                               positions=100 * np.array(loss_fractions),
                                               showmedians=True, widths=5)
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
axes[1].set_xlabel('Percentage decrease in loss')
axes[0].set_ylabel('Change in decoder-projected\ncovariability (%)')
axes[1].set_ylabel('Change in decoder-projected\ncovariability (%)')

for ax in axes:
    ax.set_xticks(100*np.array(loss_fractions))
    ax.set_xticklabels([int(100*l) for l in loss_fractions])

axes[0].set_ylim(ymax=1600)
axes[1].set_ylim(ymax=1600)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/Strategies.png')
plt.close()
