import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd-high-dim-input/dim_vs_lr"
save_fig_dir = "results/egd-high-dim-input/dim_vs_lr"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

data = np.load(f"{load_dir}/real_dims.npy", allow_pickle=True).item()
lrs = []
dims = []
for k in data.keys():
    lrs.append(k)
    dims.append(data[k])
dims = np.array(dims)

plt.figure(figsize=(45*units_convert['mm'], 45/1.25*units_convert['mm']))
for seed in range(dims.shape[1]):
    plt.plot(lrs, dims[:,seed], color='grey', lw=0.5, alpha=0.5)
print(np.std(dims, axis=1))
plt.errorbar(lrs, np.mean(dims, axis=1), yerr=np.std(dims, axis=1) / dims.shape[1]**0.5, lw=1, color='k')
plt.xlabel("Learning rate")
plt.ylabel("Dimension")
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/DimensionVSLearningRate.png')
plt.close()
plt.show()
