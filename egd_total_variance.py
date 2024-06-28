import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

"""Plot change in total variance across learning for rich and lazy regimes."""

save_fig_dir = f"results/egd/fig2-relu-m5-zscoreTrue-zeroedavgxFalse-fitinterTrue"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)
tot_var_lazy = np.load("data/egd/fig2-relu-m5-zscoreTrue-zeroedavgxFalse-fitinterTrue-expW0.55/tot_var.npy", allow_pickle=True).item()
#tot_var_rich = np.load("data/egd/new-fig2-m5-zscoreTrue-zeroedavgxFalse-fitinterTrue-expW1.0/tot_var.npy", allow_pickle=True).item()


fig, axes = plt.subplots(ncols=2, figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
m = tot_var_lazy['WM'].mean(axis=0)
axes[0].plot()

print(tot_var_lazy['WM'][:,0])
print(tot_var_lazy['OM'][:,0])