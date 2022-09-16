"""
Description:
-----------
Figure(s) about the effect of the learning rate on WM and OM adaptation.
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/effect-of-lr"
save_fig_dir = "results/egd/effect-of-lr"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss_dict['loss']


params = np.load(f"{load_dir}/params.npy", allow_pickle=True).item()
nb_iter = params['nb_iter']
nb_iter_adapts = params['nb_iter_adapts']
lr_adapts = params['lr_adapts']


# Plot adaptation loss for each seed
for seed_id in range(3):
    _, axes = plt.subplots(ncols=3, figsize=(3*45*units_convert['mm'], 45*units_convert['mm']))
    for lr_i, lr in enumerate(lr_adapts[:-1]):
        axes[lr_i].semilogy(loss['WM'][lr_i][seed_id], color=col_w, label='WM')
        axes[lr_i].semilogy(loss['OM'][lr_i][seed_id], color=col_o, label='OM')
        axes[lr_i].set_title("Learning rate = {:1.0e}".format(lr), pad=2)
    axes[1].set_xlabel('Weight update post-perturbation')
    axes[0].set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/LossSeed{seed_id}.png')
    plt.close()

