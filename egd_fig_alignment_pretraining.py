import matplotlib.pyplot as plt
import numpy as np
from seaborn import despine
from scipy.stats import wilcoxon
from scipy.stats import linregress
from utils import units_convert, col_o, col_w, convert_uneven_lists_to_array
import os

plt.style.use('rnn4bci_plot_params.dms')

exponents_W = [0.5, 1.]

xticklabels_ = ['Lazy', 'Rich']
if len(exponents_W) == 1:
    if abs(exponents_W[0] - 0.55) <= 0.1:
        xticklabels_.remove('Rich')
    elif abs(exponents_W[0] - 1) <= 0.1:
        xticklabels_.remove('Lazy')

output_fig_format = 'png'

data_representation_alignment = []
data_ntk_alignment = []
deltaW_norm = []
losses = []

for exponent_W in exponents_W:
    tag = f"pretraining-largeinitWboth-N100-Nreadouts100-activationtanh"
    model_type = "egd"
    load_dir = f"data/{model_type}/{tag}-expW{exponent_W}"
    save_fig_dir = f"results/{model_type}/{tag}"

    ra = np.load(f"{load_dir}/representation_alignment.npy")
    ntk_a = np.load(f"{load_dir}/tangent_kernel_alignment.npy")
    dW_n = np.load(f"{load_dir}/delta_W_norm.npy")
    data_representation_alignment.append(ra)
    data_ntk_alignment.append(ntk_a)
    deltaW_norm.append(dW_n)

    data = np.load(f"{load_dir}/data.npy", allow_pickle=True).item()
    loss_loc = data['initial']['loss']

    if isinstance(loss_loc, list):
        loss_loc = convert_uneven_lists_to_array(loss_loc)
    losses.append(loss_loc)

if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

# Figure 1 : representational alignment
plt.figure(figsize=(114 / 4 * units_convert['mm'], 114 / 3 * units_convert['mm'] / 1.15))
bp = plt.boxplot(data_representation_alignment, patch_artist=True,
                 flierprops={'markersize': 1, 'mew': 0.5}, boxprops={'lw': 0.5},
                 medianprops={'lw': 0.5, 'color': (0.9, 0.9, 0.9)},
                 capprops={'lw': 0.5}, whiskerprops={'lw': 0.5})

colors = ['k', 'k']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
despine(ax=plt.gca())

if len(exponents_W) == 2:
    max_y = max(max(data_representation_alignment[0]), max(data_representation_alignment[1]))
    r = wilcoxon(data_representation_alignment[0], data_representation_alignment[1], alternative='greater')
    plt.gca().text(1.5, max_y + 0.1, f"$p = {r.pvalue:.1e}$", horizontalalignment='center',
                   verticalalignment='bottom', fontsize=5)
    plt.gca().plot([1, 2], [max_y + 0.1, max_y + 0.1], 'grey', lw=0.3)

plt.gca().set_xticklabels(xticklabels_)
plt.xlabel('Regime')
plt.ylabel("Representational\nalignment")
plt.tight_layout()
despine()
plt.savefig(f'{save_fig_dir}/RepresentationAlignment.png')
plt.close()

# Figure 2 : tangent kernel alignment
align_types = ['$xx$', '$xy$', '$yx$', '$yy$']
_, axes = plt.subplots(ncols=4, figsize=(114 * units_convert['mm'], 114 / 3 * units_convert['mm'] / 1.15),
                       sharex=True, sharey=True)
for i, ax in enumerate(axes):
    data = [data_ntk_alignment[j][:, i] for j in range(len(exponents_W))]
    bp = ax.boxplot(data, patch_artist=True,
                    flierprops={'markersize': 1, 'mew': 0.5}, boxprops={'lw': 0.5},
                    medianprops={'lw': 0.5, 'color': (0.9, 0.9, 0.9)},
                    capprops={'lw': 0.5}, whiskerprops={'lw': 0.5})
    colors = ['k', 'k']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(align_types[i], pad=7)
    ax.set_xticks(list(range(1, 1+len(exponents_W))))
    ax.set_xticklabels(xticklabels_)
    ax.set_xlabel('Regime')
    despine(ax=ax)

    if len(exponents_W) == 2:
        r = wilcoxon(data[0], data[1], alternative='greater')
        ax.text(1.5, 1.1, f"$p = {r.pvalue:.1e}$", horizontalalignment='center',
                verticalalignment='bottom', fontsize=5)
        ax.plot([1, 2], 1.1 * np.ones(2), 'grey', lw=0.3)

axes[0].set_ylabel(f"Tangent kernel\nalignment")
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/TangentKernelAlignment.png')
plt.close()

# Figure 3 : norm of deltaW
plt.figure(figsize=(114 / 3 * units_convert['mm'], 114 / 3 * units_convert['mm'] / 1.15))
bp = plt.boxplot(deltaW_norm, patch_artist=True,
                 flierprops={'markersize': 1, 'mew': 0.5}, boxprops={'lw': 0.5},
                 medianprops={'lw': 0.5, 'color': (0.9, 0.9, 0.9)},
                 capprops={'lw': 0.5}, whiskerprops={'lw': 0.5})

colors = ['k', 'k']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.gca().set_xticklabels(xticklabels_)
despine(ax=plt.gca())

if len(exponents_W) == 2:
    max_y = max(max(deltaW_norm[0]), max(deltaW_norm[1]))
    r = wilcoxon(deltaW_norm[0], deltaW_norm[1], alternative='less')
    plt.gca().text(1.5, max_y+0.1, f"$p = {r.pvalue:.1e}$", horizontalalignment='center',
                   verticalalignment='bottom', fontsize=5)
    plt.gca().plot([1, 2], [max_y+0.1, max_y+0.1], 'grey', lw=0.3)

plt.xlabel('Regime')
plt.ylabel("Norm of total\nweight change")
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/NormOfTotalWeightChange.png')
plt.close()

# Figure 4: influence of initial loss
plt.figure(figsize=(114 / 3 * units_convert['mm'], 114 / 3 * units_convert['mm'] / 1.15))

if len(exponents_W) == 2:
    plt.plot(losses[0][:, 0], data_representation_alignment[0], lw=0, color='k', marker='o', markersize=1,
             label='lazy')
    plt.plot(losses[1][:, 0], data_representation_alignment[1], lw=0, color='grey', marker='x',
             markersize=1, label='rich')
    r = linregress(losses[0][:, 0], data_representation_alignment[0], alternative='two-sided')
    x = [min(losses[0][:, 0]), max(losses[0][:, 0])]

    plt.plot(x, [r.slope * x[0] + r.intercept, r.slope * x[1] + r.intercept], ':', color='blue', lw=0.5,
             label=f'p = {r.pvalue:.3f}', zorder=3)
elif xticklabels_[0] == 'Lazy':
    plt.plot(losses[0][:, 0], data_representation_alignment[0], lw=0, color='k', marker='o',
             markersize=1,
             label='lazy')
    r = linregress(losses[0][:, 0], data_representation_alignment[0], alternative='two-sided')
    x = [min(losses[0][:, 0]), max(losses[0][:, 0])]

    plt.plot(x, [r.slope * x[0] + r.intercept, r.slope * x[1] + r.intercept], ':', color='blue', lw=0.5,
             label=f'p = {r.pvalue:.3f}', zorder=3)
elif xticklabels_[0] == 'Rich':
    plt.plot(losses[1][:, 0], data_representation_alignment[1], lw=0, color='grey', marker='x',
             markersize=1, label='rich')
plt.xlabel('Starting loss, initial training')
plt.ylabel("Representational\nalignment")
plt.legend(frameon=True, labelspacing=0.1, borderpad=0.2)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/RA_vs_InitialLoss.png')
plt.close()
