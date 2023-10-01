import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')
mpl.rcParams['font.size'] = 7

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

for expo in exponent_Ws:
    load_dir = f"{load_dir_prefix}{expo}{load_dir_suffix}"
    max_eigvals[expo]['W_intuit'] = np.load(f"{load_dir}/eigenvals_init.npy")
    max_eigvals[expo]['W_WM'] = np.load(f"{load_dir}/eigenvals_after_WMP.npy")
    max_eigvals[expo]['W_OM'] = np.load(f"{load_dir}/eigenvals_after_OMP.npy")

    dims[expo] = np.load(f"{load_dir}/participation_ratio.npy", allow_pickle=True).item()

# Plot dim vs max_eigval intuit
plt.figure(figsize=(45*units_convert['mm'], 36*units_convert['mm']))
for expo in exponent_Ws:
    plt.plot(max_eigvals[expo]['W_intuit'], dims[expo]['initial'],
             lw=0, marker='o', markersize=2, mec='white', mew=0.1, color='k')
plt.xlabel('Max eigenvalue $W_\mathsf{intuit}$')
plt.ylabel('Manifold dimension\nbefore adaptation')
#plt.yticks([0.5, 1])
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/InitialManifoldDim_vs_MaxInitEigval.{output_fig_format}')
plt.close()

# Plot dim Wm and Om vs dim initial
plt.figure(figsize=(45*units_convert['mm'], 36*units_convert['mm']))
for expo in exponent_Ws:
    for perturbation_type in ['WM', 'OM']:
        plt.plot(dims[expo]['initial'], dims[expo][perturbation_type],
                 lw=0, marker='o' if perturbation_type=='WM' else 's', markersize=2, mec='white', mew=0.1,
                 color=col_w if perturbation_type=='WM' else col_o, label=perturbation_type if expo==0.55 else None)
plt.xlabel('Manifold dimension\nbefore adaptation')
plt.ylabel('Manifold dimension\nafter adaptation')
#plt.yticks([0.5, 1])
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/ManifoldDimension.{output_fig_format}')
plt.close()