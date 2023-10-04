"""
Description:
-----------
Code for figure 1B: Distribution of the pre-adaptation losses for WM and OM. Indicate median loss.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5

exponent_W = 0.55
tag = f"exponent_W{exponent_W}-lr0.001-M6-iterAdapt2000"
load_dir = f"data/egd/{tag}"
save_fig_dir = f"results/egd/{tag}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

wm = np.load(f"{load_dir}/candidate_wm_perturbations.npy")
om = np.load(f"{load_dir}/candidate_om_perturbations.npy")
print(wm.shape)
print(om.shape)
median_ = np.median(np.concatenate((wm[:, None], om[:, None])))

plt.figure(figsize=(40*units_convert['mm'], 40*units_convert['mm']))
plt.hist(wm, bins=50, label='WM', color=col_w, alpha=0.5)
plt.hist(om, bins=100, label='OM', color=col_o, alpha=0.5)
plt.gca().axvline(median_, color='grey', lw=0.5, label='combined\nmedian')
plt.xlabel('Loss')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossDistributionsCandidate.pdf')
plt.close()
plt.show()