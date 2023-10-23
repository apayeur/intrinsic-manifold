import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/exponent_W0.55-lr0.001-M6-iterAdapt2000"
save_fig_dir = "results/egd/exponent_W0.55-lr0.001-M6-iterAdapt2000"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

norm_gradW = np.load(f"{load_dir}/norm_gradW.npy", allow_pickle=True).item()
pr = np.load(f"{load_dir}/participation_ratio_during_training.npy", allow_pickle=True).item()
diff_pr = np.diff(pr['OM'], axis=1)
print(pr['OM'].shape)

plt.figure(figsize=(85/2*units_convert['mm'], 85/2*units_convert['mm']/1.25))
for i in range(norm_gradW['loss']['OM'].shape[0]):
    plt.plot(norm_gradW['loss']['OM'][i, :-1], diff_pr[i], lw=0, marker='.', markersize=1, alpha=0.3, color=col_o)
plt.xlabel(r'$\|\nabla_W L\|_F$')
plt.ylabel(r'$\Delta$ participation ratio')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/DeltaPR_vs_GradW_OM.png')
plt.close()
