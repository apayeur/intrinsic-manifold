import matplotlib.pyplot as plt
import numpy as np
from seaborn import despine
from scipy.stats import wilcoxon
from scipy.stats import linregress
from utils import units_convert, col_o, col_w, convert_uneven_lists_to_array
import os
plt.style.use('rnn4bci_plot_params.dms')

plt.rcParams.update({
    "text.usetex": False,
})
exponents_W = [0.55, 1.]
diff_relative_loss = {exponent_W: [] for exponent_W in exponents_W}
output_fig_format = 'png'
load_dir_suffix = ""  # "-lr0.001-M6-iterAdapt500"

data_representation_alignment = []
data_ntk_alignment = []
deltaW_norm = []
losses = []

for exponent_W in exponents_W:
    tag = f"pretraining-N100-Nreadouts100-activationtanh"
    model_type = "egd"
    load_dir = f"data/{model_type}/{tag}-expW{exponent_W}"
    save_fig_dir = f"results/{model_type}/{tag}"

    ra = np.load(f"{load_dir}/representation_alignment.npy", allow_pickle=True)
    ntk_a = np.load(f"{load_dir}/tangent_kernel_alignment.npy", allow_pickle=True)
    dW_n = np.load(f"{load_dir}/delta_W_norm.npy", allow_pickle=True)

    data_representation_alignment.append(ra)
    data_ntk_alignment.append(ntk_a)
    deltaW_norm.append(dW_n)

    data = np.load(f"{load_dir}/data.npy", allow_pickle=True).item()
    loss_init = data['pretraining']['loss']

    if isinstance(loss_init, list):
        loss_init = convert_uneven_lists_to_array(loss_init)
    losses.append(loss_init)

if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

# Figure 1 : representational alignment
plt.figure(figsize=(114/4*units_convert['mm'], 114/3*units_convert['mm']/1.15))
bp = plt.boxplot(data_representation_alignment, positions=[0, 1], patch_artist=True,
                 flierprops={'markersize':1, 'mew':0.5}, boxprops={'lw':0.5}, medianprops={'lw':0.5, 'color':(0.9, 0.9, 0.9)},
                 capprops={'lw':0.5}, whiskerprops={'lw':0.5})

colors = ['k', 'k']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.gca().set_xticklabels(['Lazy', 'Rich'])
plt.xlabel('Regime')
plt.ylabel("Representational\nalignment")
plt.tight_layout()
despine()
plt.savefig(f'{save_fig_dir}/RepresentationAlignment.png')
plt.close()


# Figure 2 : tangent kernel alignment
align_types = ['$xx$', '$xy$', '$yx$', '$yy$']
_, axes = plt.subplots(ncols=4, figsize=(114*units_convert['mm'], 114/3*units_convert['mm']/1.15), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    data = [data_ntk_alignment[0][:, i], data_ntk_alignment[1][:, i]]
    bp = ax.boxplot(data, patch_artist=True,
                     flierprops={'markersize':1, 'mew':0.5}, boxprops={'lw':0.5}, medianprops={'lw':0.5, 'color':(0.9, 0.9, 0.9)},
                     capprops={'lw':0.5}, whiskerprops={'lw':0.5})

    colors = ['k', 'k']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks([1,2])
    ax.set_title(align_types[i], pad=7)
    ax.set_xticklabels(['Lazy', 'Rich'])
    ax.set_xlabel('Regime')
    despine(ax=ax)

    r = wilcoxon(data[0], data[1], alternative='greater')
    ax.text(1.5, 1.1, f"$p = {r.pvalue:.1e}$", horizontalalignment='center',
            verticalalignment='bottom', fontsize=4)
    ax.plot([1, 2], 1.1*np.ones(2), 'grey', lw=0.3)

axes[0].set_ylabel(f"Tangent kernel\nalignment")
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/TangentKernelAlignment.png')
plt.close()


# Figure 3 : norm of deltaW
plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
bp = plt.boxplot(deltaW_norm, patch_artist=True,
                 flierprops={'markersize':1, 'mew':0.5}, boxprops={'lw':0.5}, medianprops={'lw':0.5, 'color':(0.9, 0.9, 0.9)},
                 capprops={'lw':0.5}, whiskerprops={'lw':0.5})

colors = ['k', 'k']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.gca().set_xticklabels(['Lazy', 'Rich'])
plt.xlabel('Regime')
plt.ylabel("Norm of total\nweight change")
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/NormOfTotalWeightChange.png')
plt.close()

# Figure 4: influence of initial loss
plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
plt.plot(losses[0][:, 0], data_representation_alignment[0], lw=0, color='k', marker='o', markersize=1, label='lazy')
plt.plot(losses[1][:, 0], data_representation_alignment[1], lw=0, color='grey', marker='x', markersize=1, label='rich')
r = linregress(losses[0][:, 0], data_representation_alignment[0], alternative='two-sided')
x = [min(losses[0][:, 0]), max(losses[0][:, 0])]
plt.plot(x, [r.slope*x[0] + r.intercept, r.slope*x[1] + r.intercept], ':', color='blue', lw=0.5, label=f'p = {r.pvalue:.3f}', zorder=3)
plt.xlabel('Starting loss, initial training')
plt.ylabel("Representational\nalignment")
plt.legend(frameon=True, labelspacing=0.1, borderpad=0.2)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/RA_vs_InitialLoss.png')
plt.close()