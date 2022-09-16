"""
Description:
-----------
Figure showing manifold dimension vs seed.
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/fc_like_test"
save_fig_dir = "results/egd/fc_like_test"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

dims = np.load(f"{load_dir}/real_dims.npy")
print(f"Mean dimension = {np.mean(dims)}")
print(f"Mean dimension = {np.std(dims, ddof=1)}")

plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
plt.plot(dims, 'o-', color='k', markersize=3)
plt.ylabel('Manifold dimension')
plt.xlabel('Seed')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/ManifoldDimensions.png')
plt.close()
plt.show()