import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')


from seaborn import despine
output_fig_format = 'pdf'

load_dir_prefix = "data/egd/exponent_W"
load_dir_suffix = "-lr0.001-M6-iterAdapt2000"
save_fig_dir = f"results/egd/eigvals-study{load_dir_suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

threshold = 0.25

exponent_Ws = [0.55, 0.6, 0.7, 0.8, 0.9, 1]
max_eigvals = {expo: {'W0': None, 'W_intuit': None, 'W_WM': None, 'W_OM': None} for expo in exponent_Ws}
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

# Plot max sum_diff vs manifold dimension
plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
min_, max_ = 100, 0
x, y = [], []
for expo in exponent_Ws:
    x += list(dims[expo]['initial'])
    y += list(sum_diff[expo])
    plt.plot(dims[expo]['initial'], sum_diff[expo], lw=0, marker='o', markersize=2, mec='white', mew=0.2, color='k')
    min_ = min(dims[expo]['initial']) if min_ > min(dims[expo]['initial']) else min_
    max_ = max(dims[expo]['initial']) if max_ < max(dims[expo]['initial']) else max_
r = linregress(x, y, alternative='less')
plt.plot([min_, max_], [r.slope*min_+r.intercept, r.slope*max_+r.intercept], color='black')
plt.gca().text(0.05, 0.1, f"p = {r.pvalue:.1}", ha='left', transform=plt.gca().transAxes, fontsize=5)
plt.gca().text(0.05, 0.3, f"R$^2$ = {r.rvalue**2:.2}", ha='left', va='top', transform=plt.gca().transAxes, fontsize=5)
plt.plot([min_, max_], [0, 0], ':', color='grey',zorder=-1)
plt.yticks([0., 0.1])
plt.xlabel('Manifold dimension\nbefore adaptation')
plt.ylabel(r'Average $\frac{L^{(\mathsf{OM})}}{L^{(\mathsf{OM})}_0} - \frac{L^{(\mathsf{WM})}}{L^{(\mathsf{WM})}_0}$') #\n(normalized)')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/SumDiff_Vs_ManifoldDimension.{output_fig_format}')
plt.close()

# Plot max sum_diff vs W_intuit
plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
min_, max_ = 1, 0
x, y = [], []
for expo in exponent_Ws:
    x += list(max_eigvals[expo]['W_intuit'])
    y += list(sum_diff[expo])
    plt.plot(max_eigvals[expo]['W_intuit'], sum_diff[expo], lw=0, marker='o', markersize=2, mec='white', mew=0.1, color='k')
    min_ = min(max_eigvals[expo]['W_intuit']) if min_ > min(max_eigvals[expo]['W_intuit']) else min_
    max_ = max(max_eigvals[expo]['W_intuit']) if max_ < max(max_eigvals[expo]['W_intuit']) else max_
r = linregress(x, y, alternative='greater')
plt.plot([min_, max_], [r.slope*min_+r.intercept, r.slope*max_+r.intercept], color='black')
plt.gca().text(0.95, 0.1, f"p = {r.pvalue:.1}", ha='right', transform=plt.gca().transAxes, fontsize=5)
plt.gca().text(0.95, 0.3, f"R$^2$ = {r.rvalue**2:.2}", ha='right', va='top', transform=plt.gca().transAxes, fontsize=5)
plt.plot([min_, max_], [0, 0], ':', color='grey')
#plt.yticks([0., 0.15])
plt.xlabel(r'Max eigenvalue $W_\mathsf{intuit}$')
plt.ylabel(r'Average $\frac{L^{(\mathsf{OM})}}{L^{(\mathsf{OM})}_0} - \frac{L^{(\mathsf{WM})}}{L^{(\mathsf{WM})}_0}$') #\n(normalized)')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/SumDiff_Vs_MaxEigVal.{output_fig_format}')
plt.close()


# Plot max_eig WM and OM vs max_eigval intuit
plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
plt.gca().set_aspect('equal')
for expo in exponent_Ws:
    for perturbation_type in ['WM', 'OM']:
        plt.plot(max_eigvals[expo]['W_intuit'], max_eigvals[expo][f'W_{perturbation_type}'], lw=0,
                marker='o' if perturbation_type=='WM' else 's', markersize=2, mec='white', mew=0.1 ,
                 color=col_w if perturbation_type=='WM' else col_o, label=perturbation_type if expo==0.6 else None)
plt.xlabel('Max eigenvalue $W$\nbefore adaptation')
plt.ylabel('Max eigenvalue $W$\nafter adaptation')
plt.yticks([0.5, 1])
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/MaxEigVal.{output_fig_format}')
plt.close()

# Plot max_eig WM and OM vs max_eigval intuit
plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
change_max_eigvals = {'WM': [], 'OM':[]}
relative_max_eigvals = []
for i, expo in enumerate(exponent_Ws):
    load_dir = f"{load_dir_prefix}{expo}{load_dir_suffix}"
    max_eigvals[expo]['W_intuit'] = np.load(f"{load_dir}/eigenvals_init.npy")

    relative_max_eigvals += list((max_eigvals[expo][f'W_OM'] - max_eigvals[expo][f'W_WM']))

    for perturbation_type in ['WM', 'OM']:
        max_eigvals[expo][f'W_{perturbation_type}'] = np.load(f"{load_dir}/eigenvals_after_{perturbation_type}P.npy")
        change_max_eigvals[perturbation_type] += list((max_eigvals[expo][f'W_{perturbation_type}'] - max_eigvals[expo]['W_intuit']))

plt.hist(relative_max_eigvals, bins='auto', color ='k', rwidth=0.9)
#for perturbation_type in ['WM', 'OM']:
#    plt.hist(change_max_eigvals[perturbation_type], bins='auto', cumulative=True, histtype='stepfilled', density=True,
#             label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o, alpha=0.5)
plt.xlabel('Difference in max eigenvalue\nafter adapt. (OM $-$ WM)')
plt.ylabel('Count')
#plt.yticks([0.5, 1])
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/HistChangeMaxEigval.{output_fig_format}')
plt.close()