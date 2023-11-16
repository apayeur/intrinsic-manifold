import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

"""
Plot of $\Delta W$ across learning for rich and lazy regime (Fig. 2F).
"""

save_fig_dir = f"results/egd/rich_vs_lazy"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

delta_W_lazy = np.load("data/egd/exponent_W0.55-lr0.001-M6-iterAdapt2000/total_change_W_Fnorm.npy", allow_pickle=True).item()
delta_W_rich = np.load("data/egd/exponent_W1-lr0.001-M6-iterAdapt2000/total_change_W_Fnorm.npy", allow_pickle=True).item()

data = []
data.append(delta_W_lazy['WM'])
data.append(delta_W_lazy['OM'])
data.append(delta_W_rich['WM'])
data.append(delta_W_rich['OM'])

plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
bp = plt.boxplot(data, positions=[0, 1, 3, 4], patch_artist=True,
                 flierprops={'markersize':1, 'mew':0.5}, boxprops={'lw':0.5}, medianprops={'lw':0.5, 'color':(0.9, 0.9, 0.9)},
                 capprops={'lw':0.5}, whiskerprops={'lw':0.5})

colors = [col_w, col_o, col_w, col_o]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.xticks([0.5, 3.5])
plt.gca().set_xticklabels(['Lazy', 'Rich'])
plt.xlabel('Regime')
#plt.ylabel(r'$\|\nabla_W L\|_F$')
plt.ylabel("Norm of total\nweight change")
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/NormOfTotalWeightChange.png')
plt.close()