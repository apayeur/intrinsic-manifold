import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
import seaborn as sns
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/incremental-training-no-partial-increment"
save_fig_dir = "results/egd/incremental-training-no-partial-increment"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

loss_dict_no_incr_training = np.load(f"{load_dir}/loss_no_incremental_training.npy", allow_pickle=True).item()
loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()

n_seeds = len(loss_dict['loss'])


endloss = {'Incremental training': [],
           'No incremental training': []}

fig, ax = plt.subplots(figsize=(30*units_convert['mm'], 45*units_convert['mm']/1.25))
for i in range(n_seeds):
    endloss['Incremental training'].append(loss_dict['loss'][i][-1])
    endloss['No incremental training'].append(loss_dict_no_incr_training['loss'][i][-1])
    if i == 0:
        ax.plot([0, 1], [loss_dict_no_incr_training['loss'][i][-1], loss_dict['loss'][i][-1]],
                lw=0.5, marker='o', markersize=2, color='k', alpha=0.4)
    else:
        ax.plot([0, 1], [loss_dict_no_incr_training['loss'][i][-1], loss_dict['loss'][i][-1]],
                lw=0.5, marker='o', markersize=2, color=col_o, alpha=0.4)
ax.set_xlabel('Incremental training')
ax.set_ylabel('Final loss')
ax.set_xlim([-0.1, 1.1])
ax.set_xticks([0,1])
ax.set_xticklabels(['No', 'Yes'])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/IncrementalTraining.png')
plt.close()