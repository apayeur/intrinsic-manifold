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

load_dir = "data/egd/effect-of-lr-weights"
save_fig_dir = "results/egd/effect-of-lr-weights"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss_dict['loss']


params = np.load(f"{load_dir}/params.npy", allow_pickle=True).item()
nb_iter = params['nb_iter']
nb_iter_adapts = params['nb_iter_adapts']
lr_adapts = params['lr_adapts']

fig_all, axes_all = plt.subplots(ncols=len(lr_adapts),
                                 figsize=(len(lr_adapts)*45*units_convert['mm'], 45*units_convert['mm']/1.25),
                                 sharey=True)
print(lr_adapts)
# Plot adaptation loss for each seed
for seed_id in [0]: #range(20):
    fig, axes = plt.subplots(ncols=len(lr_adapts), figsize=(len(lr_adapts)*45*units_convert['mm'], 45*units_convert['mm']/1.25),  sharey=True)
    for lr_i, lr in enumerate(lr_adapts):
        print(lr)
        axes[lr_i].semilogy(loss['WM'][lr_i][seed_id], color=col_w, label='WM')
        axes[lr_i].semilogy(loss['OM'][lr_i][seed_id], color=col_o, label='OM')
        axes[lr_i].set_title(r"$\eta_W = $ {:1.0e}".format(lr), y=0.8)
        axes[lr_i].set_xlabel('Weight update post-perturbation')

        axes_all[lr_i].semilogy(loss['WM'][lr_i][seed_id], color=col_w, lw=0.2, alpha=0.3, label='WM' if seed_id==0 else None)
        axes_all[lr_i].semilogy(loss['OM'][lr_i][seed_id], color=col_o, lw=0.2, alpha=0.3, label='OM' if seed_id==0 else None)

    ymin_ = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ymax_ = min(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim([ymin_, ymax_])
        ax.legend()
    axes[0].set_ylabel('Loss')
    fig.tight_layout()
    fig.savefig(f'{save_fig_dir}/LossSeed{seed_id}.png')
    plt.close(fig=fig)

ymin_ = min(axes_all[0].get_ylim()[0], axes_all[1].get_ylim()[0])
ymax_ = min(axes_all[0].get_ylim()[1], axes_all[1].get_ylim()[1])
for lr_i, lr in enumerate(lr_adapts):
    axes_all[lr_i].set_ylim([ymin_, ymax_])
    axes_all[lr_i].set_title(r"$\eta_W = $ {:1.0e}".format(lr), y=0.8)
    axes_all[lr_i].set_xlabel('Weight update post-perturbation')
    axes_all[lr_i].legend(loc='upper right')
axes_all[0].set_ylabel('Loss')
fig_all.tight_layout()
fig_all.savefig(f'{save_fig_dir}/AllLosses.png')