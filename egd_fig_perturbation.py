"""
Description:
-----------
Code for figure 1B: Distribution of the pre-adaptation losses for WM and OM. Indicate median loss.
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/final_more_seeds"
save_fig_dir = "results/egd/final_more_seeds"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

wm = np.load(f"{load_dir}/candidate_wm_perturbations.npy")
om = np.load(f"{load_dir}/candidate_om_perturbations.npy")
print(wm.shape)
print(om.shape)
median_ = np.median(np.concatenate((wm[:, None], om[:, None])))

plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
plt.hist(wm, bins=50, label='WM', color=col_w, alpha=0.5)
plt.hist(om, bins=100, label='OM', color=col_o, alpha=0.5)
plt.gca().axvline(median_, color='grey', lw=0.5)
plt.xlabel('Loss')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossDistributionsCandidate.png')
plt.close()
plt.show()