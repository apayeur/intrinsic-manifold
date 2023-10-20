import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')

output_fig_format = 'png'

dir_suffix = "exponent_W1-lr0.001-M6-iterAdapt500"
load_dir = f"data/egd/{dir_suffix}"
save_fig_dir = f"results/egd/{dir_suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

pr = np.load(f"{load_dir}/participation_ratio_during_training.npy", allow_pickle=True).item()

plt.figure(figsize=(85/2*units_convert['mm'], 85/2/1.25*units_convert['mm']))
# plot pre-perturbation eigvals
print(pr['initial'][0][0], pr['initial'][0][-1])
print(pr['WM'][0][0], pr['WM'][0][-1])
print(pr['OM'][0][0], pr['OM'][0][-1])

m = np.mean(pr['initial'], axis=0)
sem = np.std(pr['initial'], axis=0, ddof=1) / pr['initial'].shape[0] ** 0.5
plt.plot(np.arange(len(m)), m, label='initial', color='grey')
plt.fill_between(np.arange(len(m)),  m - 2 * sem, m + 2 * sem, color='grey', alpha=0.5, lw=0)

for perturbation_type in ['WM', 'OM']:
    m = np.mean(pr[perturbation_type], axis=0)
    sem = np.std(pr[perturbation_type], axis=0, ddof=1) / pr[perturbation_type].shape[0] ** 0.5
    plt.plot(np.arange(len(m), 2*len(m)), m, label=perturbation_type, color=col_w if perturbation_type == 'WM' else col_o)
    plt.fill_between(np.arange(len(m), 2*len(m)),
                                     m - 2 * sem,
                                     m + 2 * sem,
                                     color=col_w if perturbation_type == 'WM' else col_o, alpha=0.5, lw=0)

plt.xlim([0, len(m)])
plt.xticks([0, len(m), 2*len(m)])
plt.gca().set_xticklabels([-len(m), 0, len(m)])
#plt.ylim(ymax=1)
plt.yticks([2, 3, 4])
plt.xlabel('Weight update')
plt.ylabel('Manifold dimension')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/PRThroughLearning.{output_fig_format}')
plt.close()
