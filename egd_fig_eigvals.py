import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')
mpl.rcParams['font.size'] = 7

save_fig_dir = "results/egd/test-rich-eigvals-study"
load_dir_prefix = "data/egd/test-rich-exponent_W"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

threshold = 0.

exponent_Ws = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'infinity']
max_eigvals = {expo: {'W0': None, 'W_intuit': None, 'W_WM': None, 'W_OM': None} for expo in exponent_Ws}
losses = {expo: {'WM': None, 'OM': None} for expo in exponent_Ws}
sum_diff = {expo: [] for expo in exponent_Ws}

for expo in exponent_Ws:
    load_dir = f"{load_dir_prefix}{expo}"
    max_eigvals[expo]['W_intuit'] = np.load(f"{load_dir}/eigenvals_init.npy")
    max_eigvals[expo]['W_WM'] = np.load(f"{load_dir}/eigenvals_after_WMP.npy")
    max_eigvals[expo]['W_OM'] = np.load(f"{load_dir}/eigenvals_after_OMP.npy")

    loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
    loss = loss_dict['loss']
    for perturbation_type in ['WM', 'OM']:
        losses[expo][perturbation_type] = loss[perturbation_type] / loss[perturbation_type][:, 0:1]

    nb_seeds = len(max_eigvals[expo]['W_intuit'])
    for seed_i in range(nb_seeds):
        indices = losses[expo]['OM'][:, seed_i] > threshold
        d = losses[expo]['OM'][indices, seed_i] - losses[expo]['WM'][indices, seed_i]
        sum_diff[expo].append(np.sum(d) / len(indices))

# Plot max sum_diff vs W_intuit
plt.figure(figsize=(45*units_convert['mm'], 36*units_convert['mm']))
min_, max_ = 1, 0
x, y = [], []
for expo in exponent_Ws[1:]:
    x += list(max_eigvals[expo]['W_intuit'])
    y += list(sum_diff[expo])
    plt.plot(max_eigvals[expo]['W_intuit'], sum_diff[expo], lw=0, marker='o', markersize=1, color='k')
    min_ = min(max_eigvals[expo]['W_intuit']) if min_ > min(max_eigvals[expo]['W_intuit']) else min_
    max_ = max(max_eigvals[expo]['W_intuit']) if max_ < max(max_eigvals[expo]['W_intuit']) else max_
r = linregress(x, y, alternative='greater')
plt.plot([min_, max_], [r.slope*min_+r.intercept, r.slope*max_+r.intercept], color='black')
#plt.gca().text(0.02, 0.7, f"p = {r.pvalue:.2}", ha='left', transform=plt.gca().transAxes, fontsize=5)
plt.gca().text(0.02, 0.6, f"R$^2$ = {r.rvalue**2:.2}", ha='left', transform=plt.gca().transAxes, fontsize=5)
plt.plot([min_, max_], [0, 0], ':', color='grey')
plt.yticks([0., 0.15])
plt.xlabel(r'Max eigenvalue $W_\mathsf{intuit}$')
plt.ylabel('Average of\n$L_\mathsf{OM}$ - $L_\mathsf{WM}$ (normalized)')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/SumDiff_Vs_MaxEigVal.png')
plt.close()


# Plot max_eig WM and OM vs max_eigval intuit
plt.figure(figsize=(45*units_convert['mm'], 36*units_convert['mm']))
for expo in exponent_Ws[1:]:
    for perturbation_type in ['WM', 'OM']:
        plt.plot(max_eigvals[expo]['W_intuit'], max_eigvals[expo][f'W_{perturbation_type}'],
                 lw=0, marker='o' if perturbation_type=='WM' else 's', markersize=1,
                 color=col_w if perturbation_type=='WM' else col_o, label=perturbation_type if expo==0.6 else None)
plt.xlabel('Max eigenvalue $W_\mathsf{intuit}$')
plt.ylabel('Max eigenvalue $W$')
plt.yticks([0.5, 1])
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/MaxEigVal.png')
plt.close()