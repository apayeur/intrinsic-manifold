"""
Description:
-----------
Plot the manifold overlap and the ratio of projected variance
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/exponent_W0.55-lr0.001-M6-iterAdapt500-high-input"
save_fig_dir = "results/egd/exponent_W0.55-lr0.001-M6-iterAdapt500-high-input"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)


# Plot manifold overlap
overlap = np.load(f"{load_dir}/normalized_variance_explained.npy", allow_pickle=True).item()

plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
for perturbation_type in ['WM', 'OM']:
    m = np.mean(overlap[perturbation_type], axis=0)
    sem = np.std(overlap[perturbation_type], axis=0, ddof=1) / overlap[perturbation_type].shape[0]**0.5
    plt.plot(np.arange(len(m)), m, label=perturbation_type, color=col_w if perturbation_type == 'WM' else col_o)
    plt.fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem,
                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Manifold overlap, $f/f_0$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/ManifoldOverlap.png')
plt.close()
plt.show()

# Plot ratio of projected variance
R = np.load(f"{load_dir}/R.npy")

plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
m = np.mean(R, axis=0)
sem = np.std(R, axis=0, ddof=1) / R.shape[0]**0.5
plt.plot(np.arange(len(m)), m, color=col_o)
plt.fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem, color=col_o, alpha=0.5, lw=0)
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Ratio of projected variance, $R$')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/RatioProjectedVariance.png')
plt.close()
plt.show()