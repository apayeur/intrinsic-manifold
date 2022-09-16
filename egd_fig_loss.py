import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/final_low_lr"
save_fig_dir = "results/egd/final_low_lr"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss_dict['loss']
loss_init = loss_dict['loss_init']
loss_var = loss_dict['loss_var']
loss_exp = loss_dict['loss_exp']
loss_corr = loss_dict['loss_corr']
loss_proj = loss_dict['loss_proj']
loss_vbar = loss_dict['loss_vbar']

params = np.load(f"{load_dir}/params.npy", allow_pickle=True).item()
nb_iter = params['nb_iter']
nb_iter_adapt = params['nb_iter_adapt']


# ------------------------ Loss-related figures ------------------------ #
# Plot initial loss
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
for l in loss_init:
    plt.semilogy(l, color='black', lw=0.5)
plt.xlabel('Weight update')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/InitialLoss.png')
plt.close()

# Plot adaptation loss for each seed
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
for i in range(loss['WM'].shape[0]):
    line_, = plt.semilogy(loss['WM'][i], label='WM' if i == 0 else None, lw=0.5)
    plt.semilogy(loss['OM'][i], '--', label='OM' if i == 0 else None, color=line_.get_color(), lw=0.5)
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossAdaptForEachSeed.png')
plt.close()

# Plot mean +/- 2SEM adaptation loss
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
m_wm, m_om = np.mean(loss['WM'], axis=0), np.mean(loss['OM'], axis=0)
std_wm, std_om = np.std(loss['WM'], axis=0, ddof=1), np.std(loss['OM'], axis=0, ddof=1)
plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
plt.fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/loss['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/loss['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
plt.fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/loss['OM'].shape[0]**0.5,
                 m_om + 2*std_om/loss['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossAdapt.png')
plt.close()


# Plot subsampled relative performance
subsampling = nb_iter_adapt // 10
shift = nb_iter_adapt // 50  # for clearer plot
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
for i in range(loss['WM'].shape[0]):
    perf_wm = 1 - loss['WM'][i] / loss['WM'][i][0]
    perf_om = 1 - loss['OM'][i] / loss['OM'][i][0]
    line_, = plt.plot(np.arange(0 - shift, len(perf_wm) - shift, subsampling), perf_wm[::subsampling],
                      'o-', markersize=3, markeredgewidth=0,
                      label='WM' if i == 0 else None, lw=0., alpha=0.5)
    plt.plot(np.arange(0 + shift, len(perf_om) + shift, subsampling), perf_om[::subsampling],
             'x-', markersize=3, markeredgewidth=0.5, label='OM' if i == 0 else None,
             color=line_.get_color(), lw=0., alpha=0.5)
plt.yticks([0, 1])
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Retraining performance')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/SubsampledPerformance.png')
plt.close()

# Plot subsampled relative performance mean +/- SEM
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
for perturbation_type in ['WM', 'OM']:
    perf = 1 - loss[perturbation_type] / loss[perturbation_type][:,0:1]
    m, s = np.mean(perf, axis=0), np.std(perf, axis=0, ddof=1)
    # plt.plot(np.arange(1, 1 + loss['WM'].shape[1]), m_wm, color=col_w, zorder=0, label='WM', lw=0.5)
    # plt.plot(np.arange(1, 1 + loss['OM'].shape[1]), m_om, color=col_o, zorder=0, label='OM', lw=0.5)
    # plt.boxplot(loss['WM'][:, ::subsampling], positions=np.arange(1, 1 + loss['WM'].shape[1], subsampling))
    # plt.boxplot(loss['OM'][:, ::subsampling], positions=np.arange(1, 1 + loss['OM'].shape[1], subsampling))
    plt.errorbar(np.arange(0, loss[perturbation_type].shape[1], subsampling), m[::subsampling],
                 yerr=2*s[::subsampling] / s.shape[0]**0.5, fmt='o-',
                 color=col_w if perturbation_type == 'WM' else col_o,
                 markersize=3, markeredgewidth=0, label=perturbation_type, lw=0.5)
plt.yticks([0, 1])
plt.xlabel('Weight update post-perturbation')
plt.ylabel('Retraining performance')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/SubsampledPerformance_Stat.png')
plt.close()

# Plot loss components
fig, (ax_var, ax_exp) = plt.subplots(nrows=2, figsize=(45*units_convert['mm'], 2*45*units_convert['mm']/1.5), sharex=True)
for perturbation_type in ['WM', 'OM']:
    mean_ = np.mean(loss_var[perturbation_type], axis=0)
    std_ = np.std(loss_var[perturbation_type], axis=0, ddof=1)
    ax_var.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
    ax_var.fill_between(np.arange(len(mean_)),
                        mean_ - 2*std_/loss_var[perturbation_type].shape[0]**0.5,
                        mean_ + 2 * std_ / loss_var[perturbation_type].shape[0] ** 0.5,
                        color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
for perturbation_type in ['WM', 'OM']:
    mean_ = np.mean(loss_exp[perturbation_type], axis=0)
    std_ = np.std(loss_exp[perturbation_type], axis=0, ddof=1)
    ax_exp.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
    ax_exp.fill_between(np.arange(len(mean_)),
                        mean_ - 2*std_/loss_exp[perturbation_type].shape[0]**0.5,
                        mean_ + 2 * std_ / loss_exp[perturbation_type].shape[0] ** 0.5,
                        color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
ax_var.set_ylabel('Variance component\nof the loss')
ax_exp.set_ylabel('Expectation component\nof the loss')
#ax_var.set_xlabel('Weight update post-perturbation')
ax_exp.set_xlabel('Weight update post-perturbation')
ax_var.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossComponents.png')
plt.close()

# Plot loss components correlation and projection
fig, (ax_var, ax_exp) = plt.subplots(nrows=2, figsize=(45*units_convert['mm'], 2*45*units_convert['mm']/1.5), sharex=True)
for perturbation_type in ['WM', 'OM']:
    mean_ = np.mean(loss_corr[perturbation_type], axis=0)
    std_ = np.std(loss_corr[perturbation_type], axis=0, ddof=1)
    ax_var.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
    ax_var.fill_between(np.arange(len(mean_)),
                        mean_ - 2*std_/loss_var[perturbation_type].shape[0]**0.5,
                        mean_ + 2 * std_ / loss_var[perturbation_type].shape[0] ** 0.5,
                        color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
for perturbation_type in ['WM', 'OM']:
    mean_ = np.mean(loss_proj[perturbation_type], axis=0)
    std_ = np.std(loss_proj[perturbation_type], axis=0, ddof=1)
    ax_exp.plot(np.arange(len(mean_)), mean_, '-' if perturbation_type=='WM' else '--',
                label=perturbation_type, color=col_w if perturbation_type=='WM' else col_o)
    ax_exp.fill_between(np.arange(len(mean_)),
                        mean_ - 2*std_/loss_exp[perturbation_type].shape[0]**0.5,
                        mean_ + 2 * std_ / loss_exp[perturbation_type].shape[0] ** 0.5,
                        color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
ax_var.set_ylabel('Correlation component\nof the loss')
ax_exp.set_ylabel('Projection component\nof the loss')
#ax_var.set_xlabel('Weight update post-perturbation')
ax_exp.set_xlabel('Weight update post-perturbation')
ax_var.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossComponentsCorrProj.png')
plt.close()

# Plot total-covariance loss component vs vbar-related loss component
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
ratio = {'WM': loss_vbar['WM'],
         'OM': loss_vbar['OM']}
m_wm, m_om = np.mean(ratio['WM'], axis=0), np.mean(ratio['OM'], axis=0)
std_wm, std_om = np.std(ratio['WM'], axis=0, ddof=1), np.std(ratio['OM'], axis=0, ddof=1)
plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
plt.fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/ratio['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/ratio['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
plt.fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/ratio['OM'].shape[0]**0.5,
                 m_om + 2*std_om/ratio['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)

plt.xlabel('Weight update post-perturbation')
plt.ylabel(r'$\frac{1}{2}\|\|V \bar{\mathbf{v}} \|  \|^2$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/Loss_vbar.png')
plt.close()


plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
ratio = {'WM': loss_corr['WM'] - loss_vbar['WM'],
         'OM': loss_corr['OM'] - loss_vbar['OM']}

m_wm, m_om = np.mean(ratio['WM'], axis=0), np.mean(ratio['OM'], axis=0)
std_wm, std_om = np.std(ratio['WM'], axis=0, ddof=1), np.std(ratio['OM'], axis=0, ddof=1)
plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
plt.fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/loss['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/loss['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
plt.fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/loss['OM'].shape[0]**0.5,
                 m_om + 2*std_om/loss['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)

plt.xlabel('Weight update post-perturbation')
plt.ylabel(r'$\frac{1}{2}$tr$\{V \mathbb{V}[\bar{\mathbf{v}}] V^\mathsf{T}\}$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/Loss_TotalVariance.png')
plt.close()


