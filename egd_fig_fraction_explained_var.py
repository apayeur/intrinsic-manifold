"""
Description:
-----------
Plot the fraction of explained variance tr(C @ Var @ C.T) / tr(Var)
by the original manifold.
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


# Plot manifold overlap
f = np.load(f"{load_dir}/f.npy", allow_pickle=True).item()

plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
for perturbation_type in ['WM', 'OM']:
    m = np.mean(f[perturbation_type], axis=0)
    sem = np.std(f[perturbation_type], axis=0, ddof=1) / f[perturbation_type].shape[0]**0.5
    plt.plot(np.arange(len(m)), m, label=perturbation_type, color=col_w if perturbation_type == 'WM' else col_o)
    plt.fill_between(np.arange(len(m)), m - 2 * sem, m + 2 * sem,
                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Fraction of variance explained by $C$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/FractionExplainedVariance.png')
plt.close()
plt.show()