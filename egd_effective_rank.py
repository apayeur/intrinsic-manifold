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
plot_relative = False
if plot_relative:
    assert len(exponents) == 1, "Only a single should be plotted exponent when plotting relative PR"

save_fig_dir = f"results/egd/{dir_suffixes[0]}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

er = {expo: np.load(f"data/egd/exponent_W{expo}-lr0.001-M6-iterAdapt2000/effective_rank.npy", allow_pickle=True).item() for expo in exponents}
nb_seeds, nb_iter_initial = er[exponents[0]]['initial'].shape
nb_iter_adapt = er[exponents[0]]['WM'].shape[1]

plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
for i, expo in enumerate(exponents):
    if plot_relative:
        pr_loc = er[expo]['initial'] / er[expo]['initial'][:, [-1]]
    else:
        pr_loc = er[expo]['initial']
    # plot pre-perturbation pr
    m = np.mean(pr_loc, axis=0)
    sem = np.std(pr_loc, axis=0, ddof=1) / nb_seeds ** 0.5
    # for seed in range(nb_seeds):
    #     plt.plot(np.arange(-len(m), 0), pr[expo]['initial'][seed], color='grey', alpha=0.3)
    plt.plot(np.arange(-nb_iter_initial, 0), m, label='initial' if i==0 else None, color='grey')
    plt.fill_between(np.arange(-nb_iter_initial, 0),  m - 2 * sem, m + 2 * sem, color='grey', alpha=0.5, lw=0)
    # plot adaptation pr
    for perturbation_type in ['WM', 'OM']:
        if plot_relative:
            pr_loc = er[expo][perturbation_type] / er[expo][perturbation_type][:, [-1]]
        else:
            pr_loc = er[expo][perturbation_type]
        m = np.mean(pr_loc, axis=0)
        sem = np.std(pr_loc, axis=0, ddof=1) / nb_seeds ** 0.5
        # for seed in range(nb_seeds):
        #     plt.plot(np.arange(len(m)),pr[expo][perturbation_type][seed], color=col_w if perturbation_type == 'WM' else col_o, alpha=0.3)
        plt.plot(np.arange(nb_iter_adapt), m,
                 label=perturbation_type if i == 0 else None, color=col_w if perturbation_type == 'WM' else col_o)
        plt.fill_between(np.arange(nb_iter_adapt), m - 2 * sem, m + 2 * sem,
                         color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)
        if not plot_relative:
            if perturbation_type == 'WM':
                plt.gca().text(100, m[0] + 2*sem[0]+0.1, rf'$\alpha = {expo}$', fontsize=5, ha='right')

plt.xlim([-nb_iter_initial, nb_iter_adapt])
plt.xticks([-nb_iter_initial, 0, nb_iter_adapt // 2, nb_iter_adapt])
#plt.gca().set_xticklabels([-pr['initial'].shape[1], 0, len(m)])
#plt.ylim([1, 6])
#plt.yticks(np.arange(1, 6+1))
plt.xlabel('Weight update')
plt.ylabel('Effective rank')
plt.legend()
plt.tight_layout()
outfile_name = f'{save_fig_dir}/EffectiveRank.{output_fig_format}'
plt.savefig(outfile_name)
plt.close()
