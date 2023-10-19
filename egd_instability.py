"""
Study of acquired instability during learning (max eigval becoming > 1) and irregular (noisy/oscillatory)
losses for different
learning rates.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')


lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
fraction_unstable_seeds = {'WM': [], 'OM': []}
fraction_irregular_stable_seeds = {'WM': [], 'OM': []}

output_fig_format = 'png'
model_type = "egd"
tag = "exponent_W0.55-M6-iterAdapt1000-lrstudy"
save_fig_dir = f"results/{model_type}/{tag}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

for lr in lrs:
    # Load data
    load_dir = f"data/{model_type}/exponent_W0.55-lr{lr}-M6-iterAdapt1000-lrstudy"

    max_eigvals = {'WM': np.load(f"{load_dir}/eigenvals_after_WMP.npy"),
                   'OM': np.load(f"{load_dir}/eigenvals_after_OMP.npy")}

    loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
    loss = loss_dict['loss']

    for perturbation_type in ['WM', 'OM']:
        # Proportion of unstable seeds
        nb_seeds = len(max_eigvals[perturbation_type])
        unstable_seeds = np.where(max_eigvals[perturbation_type] > 1)[0]
        nb_stable_seeds = nb_seeds - len(unstable_seeds)
        fraction_unstable_seeds[perturbation_type].append(len(unstable_seeds) / nb_seeds)

        # Proportion of irregular stable seeds
        seeds_to_include = [i for i in range(nb_seeds) if i not in unstable_seeds]
        loss[perturbation_type] = loss[perturbation_type][seeds_to_include]

        nb_stable_irregular_seeds = 0
        for i in range(loss[perturbation_type].shape[0]):
            if np.any(np.gradient(loss[perturbation_type][i]) > 0):
                nb_stable_irregular_seeds += 1
        fraction_irregular_stable_seeds[perturbation_type].append((nb_stable_irregular_seeds + len(unstable_seeds)) / nb_seeds)

# Plotting
fig, axes = plt.subplots(ncols=2, figsize=(85*units_convert['mm'], 85/2/1.25*units_convert['mm']), sharex=True, sharey=True)

# proportion of unstable seeds
for perturbation_type in ['WM', 'OM']:
    axes[0].plot(lrs, fraction_unstable_seeds[perturbation_type], label=perturbation_type,
                 marker='o' if perturbation_type=='WM' else 's', markersize=2.5, mew=0.3, mec='white',
                 color=col_w if perturbation_type=='WM' else col_o)
axes[0].set_xlabel('Learning rate')
axes[0].set_ylabel('Proportion of\nunstable seeds')
axes[0].set_xticks(lrs[3:])
axes[0].set_ylim([-0.05, 1.05])
axes[0].set_yticks([0, 0.5, 1])
axes[0].legend()

# proportion of irregular stable seeds
for perturbation_type in ['WM', 'OM']:
    axes[1].plot(lrs, fraction_irregular_stable_seeds[perturbation_type], label=perturbation_type,
                 marker='o' if perturbation_type == 'WM' else 's', markersize=2.5, mew=0.3, mec='white',
                 color=col_w if perturbation_type=='WM' else col_o)
axes[1].set_xlabel('Learning rate')
axes[1].set_ylabel('Proportion of\nirregular seeds')

plt.tight_layout()
plt.savefig(f'{save_fig_dir}/UnstableSeeds.{output_fig_format}')

