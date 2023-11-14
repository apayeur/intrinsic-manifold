import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
import seaborn as sns
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd/incremental-training-exponent0.55"
save_fig_dir = "results/egd/incremental-training-exponent0.55"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

loss_dict_no_incr_training = np.load(f"{load_dir}/loss_no_incremental_training.npy", allow_pickle=True).item()
loss_dict = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()

n_seeds = len(loss_dict['loss'])
seeds = np.arange(n_seeds, dtype=int)
seeds = np.delete(seeds, 18)

endloss = {'Incremental training': [],
           'No incremental training': []}

fig, ax = plt.subplots(figsize=(30*units_convert['mm'], 85/2*units_convert['mm']/1.25))

for i in seeds:
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


# Plot loss (mean +/- 2SEM)
fig, ax = plt.subplots(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
m = np.mean(np.array(loss_dict['loss'])[seeds], axis=0)
s = np.std(np.array(loss_dict['loss'])[seeds], axis=0, ddof=1) / len(seeds)**0.5
plt.plot(np.arange(m.shape[0]), m,  color=col_o, label='incr. training')
plt.fill_between(np.arange(m.shape[0]),
                 m - 2 * s,
                 m + 2 * s,
                 color=col_o, lw=0, alpha=0.5)
m = np.mean(np.array(loss_dict_no_incr_training['loss'])[seeds], axis=0)
s = np.std(np.array(loss_dict_no_incr_training['loss'])[seeds], axis=0, ddof=1) / len(seeds)**0.5
plt.plot(np.arange(m.shape[0]), m,  ':', color=[1.3 * col_i for col_i in col_o], label='no incr. training')
plt.fill_between(np.arange(m.shape[0]),
                 m - 2 * s ,
                 m + 2 * s,
                 color=[1.3 * col_i for col_i in col_o], lw=0, alpha=0.5)
#ax.set_ylim([0,0.5])
ax.set_xlabel('Weight update after perturbation')
ax.set_ylabel('Loss')
ax.legend()
#sns.despine(trim=True)
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/LossAcrossSeeds.png')
plt.close()

# Plot loss each seed
for i in seeds:
    fig, ax = plt.subplots(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    plt.plot(np.arange(m.shape[0]), loss_dict['loss'][i], color=col_o, label='incr. training')
    plt.plot(np.arange(m.shape[0]), loss_dict_no_incr_training['loss'][i],
             color=[1.3 * col_i for col_i in col_o], label='no incr. training')
    ax.set_xlabel('Weight update after perturbation')
    ax.set_ylabel('Loss')
    ax.set_xlim([0, m.shape[0]])
    ax.set_xticks([0, m.shape[0]])
    ax.legend()
    #sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/CompareLoss_Seed{i}.png')
    plt.close()
