import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

exponent_W = 0.55
file_suffix = f"exponent_W{exponent_W}-lr0.001-M6-iterAdapt500"
load_dir = f"data/egd/{file_suffix}"
save_fig_dir = f"results/egd/{file_suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

norm_gradW = np.load(f"{load_dir}/norm_gradW.npy", allow_pickle=True).item()

# ------------------------ Learning-signatures-related figures ------------------------ #
# Plot mean +/- 2SEM of the Frobenius norm of dL/dW.T
plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
m_wm, m_om = np.mean(norm_gradW['loss']['WM'], axis=0), np.mean(norm_gradW['loss']['OM'], axis=0)
std_wm, std_om = np.std(norm_gradW['loss']['WM'], axis=0, ddof=1), np.std(norm_gradW['loss']['OM'], axis=0, ddof=1)
plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
plt.fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/norm_gradW['loss']['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/norm_gradW['loss']['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
plt.fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/norm_gradW['loss']['OM'].shape[0]**0.5,
                 m_om + 2*std_om/norm_gradW['loss']['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)
if exponent_W == 0.55:
    plt.gca().text(0.5, 0.9, 'Lazy', ha='center', va='center', transform=plt.gca().transAxes)
elif exponent_W == 1:
    plt.gca().text(0.5, 0.9, 'Rich', ha='center', va='center', transform=plt.gca().transAxes)
#plt.xlim([0, len(m_wm)])
plt.xticks([0, len(m_wm)])
plt.ylim([0, 5])
#plt.ylim(ymax=15)
plt.yticks([0, 5])
plt.gca().set_yticklabels(['0.0', '5.0'])
plt.xlabel('Weight update post-perturb.')
#plt.ylabel(r'$\|\nabla_W L\|_F$')
plt.ylabel("Gradient norm")
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/NormGradWTotal.png')
plt.close()

# Plot mean +/- 2SEM of the Frobenius norm of dL/dW.T
plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
m_wm, m_om = np.mean(norm_gradW['loss_tot_var']['WM'], axis=0), np.mean(norm_gradW['loss_tot_var']['OM'], axis=0)
std_wm, std_om = np.std(norm_gradW['loss_tot_var']['WM'], axis=0, ddof=1), np.std(norm_gradW['loss_tot_var']['OM'], axis=0, ddof=1)
plt.plot(np.arange(m_wm.shape[0]), m_wm, label='WM', color=col_w, lw=0.5)
plt.plot(np.arange(m_om.shape[0]), m_om, '--', label='OM', color=col_o, lw=0.5)
plt.fill_between(np.arange(m_wm.shape[0]),
                 m_wm - 2*std_wm/norm_gradW['loss_tot_var']['WM'].shape[0]**0.5,
                 m_wm + 2*std_wm/norm_gradW['loss_tot_var']['WM'].shape[0]**0.5,
                 color=col_w, lw=0, alpha=0.5)
plt.fill_between(np.arange(m_om.shape[0]),
                 m_om - 2*std_om/norm_gradW['loss_tot_var']['OM'].shape[0]**0.5,
                 m_om + 2*std_om/norm_gradW['loss_tot_var']['OM'].shape[0]**0.5,
                 color=col_o, lw=0, alpha=0.5)
plt.ylim(ymin=0)
plt.ylim(ymax=150)

plt.yticks(list(plt.gca().get_ylim()))
plt.xlabel('Weight update post-perturbation')
plt.ylabel(r'$\| \nabla_W L_{\mathbb{V}[\mathbf{v}]}\|_F$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/NormGradWVar.png')
plt.close()