import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5

output_fig_format = 'pdf'

load_dir_prefix = "data/egd/exponent_W"
load_dir_suffix = "-lr0.001-M6-iterAdapt2000"
save_fig_dir = f"results/egd/eigvals-study{load_dir_suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

threshold = 0.25

exponent_Ws = [0.55, 0.6, 0.7, 0.8, 0.9, 1]
max_eigvals = {expo: {'W0': None, 'W_intuit': None, 'W_WM': None, 'W_OM': None} for expo in exponent_Ws}
dims = {expo: None for expo in exponent_Ws}
losses = {expo: {'WM': None, 'OM': None} for expo in exponent_Ws}
sum_diff = {expo: [] for expo in exponent_Ws}
dims = {expo: None for expo in exponent_Ws}

for expo in exponent_Ws:
    load_dir = f"{load_dir_prefix}{expo}{load_dir_suffix}"
    max_eigvals[expo]['W_intuit'] = np.load(f"{load_dir}/eigenvals_init.npy")
    max_eigvals[expo]['W_WM'] = np.load(f"{load_dir}/eigenvals_after_WMP.npy")
    max_eigvals[expo]['W_OM'] = np.load(f"{load_dir}/eigenvals_after_OMP.npy")

    dims[expo] = np.load(f"{load_dir}/participation_ratio.npy", allow_pickle=True).item()

    loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
    loss = loss_dict['loss']
    for perturbation_type in ['WM', 'OM']:
        losses[expo][perturbation_type] = loss[perturbation_type] / loss[perturbation_type][:, 0:1]

    nb_seeds = len(max_eigvals[expo]['W_intuit'])
    print(losses[expo]['OM'].shape)
    for seed_i in range(nb_seeds):
        indices = losses[expo]['OM'][seed_i, :] > threshold
        d = losses[expo]['OM'][seed_i, indices] - losses[expo]['WM'][seed_i, indices]
        sum_diff[expo].append(np.sum(d) / len(indices))

for expo in exponent_Ws:
    load_dir = f"{load_dir_prefix}{expo}{load_dir_suffix}"
    max_eigvals[expo]['W_intuit'] = np.load(f"{load_dir}/eigenvals_init.npy")
    max_eigvals[expo]['W_WM'] = np.load(f"{load_dir}/eigenvals_after_WMP.npy")
    max_eigvals[expo]['W_OM'] = np.load(f"{load_dir}/eigenvals_after_OMP.npy")

    dims[expo] = np.load(f"{load_dir}/participation_ratio.npy", allow_pickle=True).item()


# Plot max sum_diff vs manifold dimension
fig, axes = plt.subplots(ncols=2, figsize=(2*45*units_convert['mm'], 36*units_convert['mm']))
min_, max_ = 100, 0
min_diff = 10
x, y = [], []
for expo in exponent_Ws:
    x += list(dims[expo]['initial'])
    y += list(sum_diff[expo])
    axes[0].plot(dims[expo]['initial'], sum_diff[expo], lw=0, marker='o', markersize=2.5, mec='white', mew=0.3, color='k')
    min_ = min(dims[expo]['initial']) if min_ > min(dims[expo]['initial']) else min_
    max_ = max(dims[expo]['initial']) if max_ < max(dims[expo]['initial']) else max_
    min_diff = min(sum_diff[expo]) if min_diff > min(sum_diff[expo]) else min_diff
r = linregress(x, y, alternative='less')
axes[0].fill_between([min_, max_], [min_diff, min_diff], [0,0], color='grey', zorder=-1, alpha=0.3, lw=0)
#axes[0].plot([min_, max_], [r.slope*min_+r.intercept, r.slope*max_+r.intercept], color='black')
#axes[0].text(0.05, 0.1, f"p = {r.pvalue:.1}", ha='left', transform=axes[0].transAxes, fontsize=5)
#axes[0].text(0.05, 0.3, f"R$^2$ = {r.rvalue**2:.2}", ha='left', va='top', transform=axes[0].transAxes, fontsize=5)
#axes[0].plot([min_, max_], [0, 0], ':', color='grey', zorder=-1)
#axes[0].set_yticks([0., 0.1])
axes[0].set_xlabel('Manifold dimension\nbefore adaptation')
axes[0].set_ylabel(r'Average $\frac{L^{(\mathsf{OM})}}{L^{(\mathsf{OM})}_0} - \frac{L^{(\mathsf{WM})}}{L^{(\mathsf{WM})}_0}$') #\n(normalized)')

# Plot dim vs max_eigval intuit
for expo in exponent_Ws:
    axes[1].plot(max_eigvals[expo]['W_intuit'], dims[expo]['initial'],
             lw=0, marker='o', markersize=2.5, mec='white', mew=0.3, color='k')
axes[1].set_xlabel('Max eigenvalue $W$\nbefore adaptation')
axes[1].set_ylabel('Manifold dimension\nbefore adaptation')
#plt.yticks([0.5, 1])
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/ManifoldDimension.{output_fig_format}')
plt.close()

# Plot dim Wm and Om vs dim initial
# for expo in exponent_Ws:
#     for perturbation_type in ['WM', 'OM']:
#         axes[1].plot(dims[expo]['initial'], dims[expo][perturbation_type],
#                  lw=0, marker='o' if perturbation_type=='WM' else 's', markersize=2, mec='white', mew=0.1,
#                  color=col_w if perturbation_type=='WM' else col_o, label=perturbation_type if expo==0.55 else None)
# axes[1].set_xlabel('Manifold dimension\nbefore adaptation')
# axes[1].set_ylabel('Manifold dimension\nafter adaptation')
# #plt.yticks([0.5, 1])
# axes[1].legend()
# plt.tight_layout()
#