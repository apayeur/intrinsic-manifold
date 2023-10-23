import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')

output_fig_format = 'png'

exponents = [0.55, 1]
dir_suffixes = [f"exponent_W{expo}-lr0.001-M6-iterAdapt2000" for expo in exponents]

save_fig_dir = f"results/egd/{dir_suffixes[0]}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

for i, expo in enumerate(exponents):
    load_dir = f"data/egd/{dir_suffixes[0]}"

pr = {expo: np.load(f"data/egd/exponent_W{expo}-lr0.001-M6-iterAdapt2000/participation_ratio_during_training.npy", allow_pickle=True).item() for expo in exponents}

plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
for i, expo in enumerate(exponents):
    # plot pre-perturbation eigvals
    m = np.mean(pr[expo]['initial'], axis=0)
    sem = np.std(pr[expo]['initial'], axis=0, ddof=1) / pr[expo]['initial'].shape[0] ** 0.5
    plt.plot(np.arange(-len(m), 0), m, label='initial' if i==0 else None, color='grey')
    plt.fill_between(np.arange(-len(m), 0),  m - 2 * sem, m + 2 * sem, color='grey', alpha=0.5, lw=0)

    for perturbation_type in ['WM', 'OM']:
        m = np.mean(pr[expo][perturbation_type], axis=0)
        sem = np.std(pr[expo][perturbation_type], axis=0, ddof=1) / pr[expo][perturbation_type].shape[0] ** 0.5
        plt.plot(np.arange(len(m)), m, label=perturbation_type if i==0 else None, color=col_w if perturbation_type == 'WM' else col_o)
        plt.fill_between(np.arange(len(m)),
                                         m - 2 * sem,
                                         m + 2 * sem,
                                         color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)

        if perturbation_type == 'WM':
            plt.gca().text(100, m[0] + 2*sem[0]+0.1, rf'$\alpha = {expo}$', fontsize=5, ha='right')
plt.xlim([-pr[exponents[0]]['initial'].shape[1], pr[exponents[0]]['WM'].shape[1]])
plt.xticks([-pr[exponents[0]]['initial'].shape[1], 0, pr[exponents[0]]['WM'].shape[1]])
#plt.gca().set_xticklabels([-pr['initial'].shape[1], 0, len(m)])
#plt.ylim([1, 6])
#plt.yticks(np.arange(1, 6+1))
plt.xlabel('Weight update')
plt.ylabel('Participation ratio')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/PRThroughLearning.{output_fig_format}')
plt.close()
