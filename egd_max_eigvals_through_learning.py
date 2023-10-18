import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')

output_fig_format = 'png'

load_dir = "data/egd/exponent_W0.55-lr0.001-M6-iterAdapt500-lrstudy"
save_fig_dir = "results/egd/exponent_W0.55-lr0.001-M6-iterAdapt500"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

max_eigvals_init = np.load(f"{load_dir}/eigenvals_init.npy")
max_eigvals = np.load(f"{load_dir}/eigenvals.npy", allow_pickle=True).item()

plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
# plot pre-perturbation eigvals
m = np.mean(max_eigvals_init, axis=0)
sem = np.std(max_eigvals_init, axis=0, ddof=1) / max_eigvals_init.shape[0] ** 0.5
plt.plot(np.arange(len(m)), m, label='initial', color='grey')
plt.fill_between(np.arange(len(m)),  m - 2 * sem, m + 2 * sem, color='grey', alpha=0.5, lw=0)

for perturbation_type in ['WM', 'OM']:
    m = np.mean(max_eigvals[perturbation_type], axis=0)
    sem = np.std(max_eigvals[perturbation_type], axis=0, ddof=1) / max_eigvals[perturbation_type].shape[0] ** 0.5
    plt.plot(np.arange(len(m), 2*len(m)), m, label=perturbation_type, color=col_w if perturbation_type == 'WM' else col_o)
    plt.fill_between(np.arange(len(m), 2*len(m)),
                                     m - 2 * sem,
                                     m + 2 * sem,
                                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)

plt.xlim([0, len(m)])
plt.xticks([0, len(m), 2*len(m)])
plt.gca().set_xticklabels([-len(m), 0, len(m)])
plt.ylim(ymax=1)
plt.yticks([0.8, 0.9, 1])
plt.xlabel('Weight update')
plt.ylabel('Max eigenvalue of $W$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/MaxEigvalsWThroughLearning.{output_fig_format}')
plt.close()