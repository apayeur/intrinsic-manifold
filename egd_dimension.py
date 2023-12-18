import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from seaborn import despine
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')


output_fig_format = 'png'

def bins_with_same_number_of_points(x, nb_points=10):
    sorted_x = np.sort(x)
    bins = sorted_x[::nb_points]
    bins = np.append(bins, sorted_x[-1])
    return bins


load_dir_prefix = "data/egd/exponent_W"
load_dir_suffix = "-lr0.001-M6-iterAdapt2000"
save_fig_dir = f"results/egd/eigvals-study{load_dir_suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

threshold = 0.5

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
plt.figure( figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
min_, max_ = 100, 0
min_diff = 10
x, y = [], []
for expo in exponent_Ws:
    x += list(dims[expo]['initial'])
    y += list(sum_diff[expo])
    plt.plot(dims[expo]['initial'], sum_diff[expo], lw=0, marker='o', markersize=1.5, mec='white', mew=0.1, color='grey')
    min_ = min(dims[expo]['initial']) if min_ > min(dims[expo]['initial']) else min_
    max_ = max(dims[expo]['initial']) if max_ < max(dims[expo]['initial']) else max_
    min_diff = min(sum_diff[expo]) if min_diff > min(sum_diff[expo]) else min_diff
r = linregress(x, y, alternative='less')
plt.fill_between([min_, max_], [min_diff, min_diff], [0,0], color='grey', zorder=-1, alpha=0.2, lw=0)
#axes[0].plot([min_, max_], [r.slope*min_+r.intercept, r.slope*max_+r.intercept], color='black')
#axes[0].text(0.05, 0.1, f"p = {r.pvalue:.1}", ha='left', transform=axes[0].transAxes, fontsize=5)
#axes[0].text(0.05, 0.3, f"R$^2$ = {r.rvalue**2:.2}", ha='left', va='top', transform=axes[0].transAxes, fontsize=5)
#axes[0].plot([min_, max_], [0, 0], ':', color='grey', zorder=-1)
#axes[0].set_yticks([0., 0.1])
# compute histogram
nb_points = 24
bins = bins_with_same_number_of_points(x, nb_points=nb_points)
m, _, _ = plt.hist(x, bins=bins, weights=np.array(y)/nb_points, histtype='step', lw=0.5, color='k', zorder=10)
bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
s, _ = np.histogram(x, bins=bins, weights=np.array(y)**2/nb_points)
sem = np.sqrt((nb_points-1)/nb_points * (s - m**2)) / nb_points**0.5
plt.errorbar(bin_centers, m, yerr=2*sem, lw=0, elinewidth=0.5, ecolor='k')
plt.xlabel('Manifold dimension\nbefore adaptation')
plt.ylabel(r'Average $\frac{L^{(\mathsf{OM})}}{L^{(\mathsf{OM})}_0} - \frac{L^{(\mathsf{WM})}}{L^{(\mathsf{WM})}_0}$') #\n(normalized)')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/AvgDiff_vs_ManifoldDimension.{output_fig_format}')
plt.close()

plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
# Plot dim vs max_eigval intuit
for expo in exponent_Ws:
    plt.plot(max_eigvals[expo]['W_intuit'], dims[expo]['initial'],
             lw=0, marker='o', markersize=1.5, mec='white', mew=0.1, color='grey')
plt.xlabel('Max eigenvalue $W$\nbefore adaptation')
plt.ylabel('Manifold dimension\nbefore adaptation')
#plt.yticks([0.5, 1])
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/ManifoldDimension_vs_MaxEigval.{output_fig_format}')
plt.close()

# Plot max eigval before Wm and Om vs dim initial
# plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
# for expo in exponent_Ws:
#      for perturbation_type in ['WM', 'OM']:
#         axes[1].plot(dims[expo]['initial'], dims[expo][perturbation_type],
#                  lw=0, marker='o' if perturbation_type=='WM' else 's', markersize=2, mec='white', mew=0.1,
#                  color=col_w if perturbation_type=='WM' else col_o, label=perturbation_type if expo==0.55 else None)
# axes[1].set_xlabel('Manifold dimension\nbefore adaptation')
# axes[1].set_ylabel('Manifold dimension\nafter adaptation')
# #plt.yticks([0.5, 1])
# axes[1].legend()
# plt.tight_layout()

# Plot dim vs alpha
plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
manifold_dims = np.empty(shape=(len(dims[exponent_Ws[0]]['initial']), len(exponent_Ws)))
# Plot dim vs max_eigval intuit
m = []
sem = []
for i, expo in enumerate(exponent_Ws):
    m.append(np.mean(dims[expo]['initial']))
    sem.append(np.std(dims[expo]['initial'], ddof=1) / len(dims[expo]['initial'])**0.5)
    manifold_dims[:, i] = dims[expo]['initial']
plt.errorbar(exponent_Ws, m, yerr=2*np.array(sem), color='k', marker='o', markersize=3, mec='white', mew=0.5, lw=1)
#plt.boxplot(manifold_dims, positions=exponent_Ws, manage_ticks=False, widths=0.02, flierprops={'marker':'o', 'markersize':0.5})
plt.xlabel(r'$\alpha$')
plt.ylabel('Manifold dimension\nbefore adaptation')
plt.yticks([3,4])
#plt.yticks([0.5, 1])
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/ManifoldDimension_vs_Alpha.{output_fig_format}')
plt.close()

# Empirical distribution of max eigvals for each alphas
"""
n_samples = int(1e3)
plt.figure(figsize=(85/2*units_convert['mm'], 85/2*units_convert['mm']))
rng = np.random.default_rng()
for expo in exponent_Ws:
    print(f"alpha = {expo}")
    max_eigval_samples = np.empty(n_samples)
    for i in range(n_samples):
        w = rng.standard_normal(size=(100, 100)) / 100 ** expo
        max_eigval_samples[i] = np.max(np.abs(np.linalg.eigvals(w)))
    plt.hist(max_eigval_samples, bins='auto', density=True, label=rf'$\alpha = {expo}$')
plt.ylabel('PDF')
plt.xlabel('Max initial eigenvalue of $W$')
#plt.yticks([0.5, 1])
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/MaxEigVal_vs_Alpha.{output_fig_format}')
plt.close()
"""
