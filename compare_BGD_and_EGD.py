import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

exponents_W = [0.55] #, 0.6, 0.7, 0.8, 0.9, 1]
diff_relative_loss = {exponent_W: [] for exponent_W in exponents_W}
output_fig_format = 'png'

filename_egd = f"data/egd/exponent_W0.55-lr0.001-M6-iterAdapt500"
filename_bgd = f"data/batch-sgd/exponent_W0.55-lr0.001-M6-iterAdapt1000"

save_fig_dir = f"results/compare_egd_bgd"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

# Load data
losses = dict()
loss_dict = np.load(f"{filename_egd}/loss.npy", allow_pickle=True).item()
losses['egd'] = loss_dict['loss']
loss_dict = np.load(f"{filename_bgd}/loss.npy", allow_pickle=True).item()
losses['bgd'] = loss_dict['loss']


colors = {'egd': {'WM': col_w, 'OM': col_o}, 'bgd': {'WM': [c + 0.2 for c in col_w], 'OM': [c + 0.2 for c in col_o]}}


# Plot mean +/- 2SEM adaptation loss
plt.figure(figsize=(114/3*units_convert['mm'], 114/3*units_convert['mm']/1.15))
plot_relative_loss = False
for learning_type in ['egd', 'bgd']:
    for perturbation_type in ['WM', 'OM']:
        if plot_relative_loss:
            perf = losses[learning_type][perturbation_type] / losses[learning_type][perturbation_type][:,0:1]
            print(perf.shape)
            print(f"{learning_type.capitalize()}: Loss {perturbation_type} at update 150", np.mean(perf[:, 150]),
                  "+/-", 2*np.std(perf[:, 150], ddof=1)/perf.shape[0]**0.5)
            print(f"{learning_type.capitalize()}: Loss {perturbation_type} at update 67", np.mean(perf[:, 67]),
                  "+/-", 2*np.std(perf[:, 67], ddof=1)/perf.shape[0]**0.5)
        else:
            perf = losses[learning_type][perturbation_type]
        m = np.mean(perf, axis=0)
        std = np.std(perf, axis=0, ddof=1)
        plt.plot(np.arange(m.shape[0]), m, '-' if perturbation_type=='WM' else '--', label=f"{learning_type.upper()}, {perturbation_type}",
                 color=colors[learning_type][perturbation_type], lw=0.5)
        plt.fill_between(np.arange(m.shape[0]),
                         m- 2*std/losses[learning_type][perturbation_type].shape[0]**0.5,
                         m + 2*std/losses[learning_type][perturbation_type].shape[0]**0.5,
                         color=colors[learning_type][perturbation_type], lw=0, alpha=0.25)
if plot_relative_loss:
    plt.ylim([0,1])
    plt.yticks([0,0.5,1])
    plt.ylabel('$L/L_0$')
else:
    plt.ylim([0, 0.55])
    plt.yticks([0, 0.5])
    plt.ylabel('Loss')
#plt.xlim([0, len(m)])
plt.xlim([0-10, 500])
plt.xticks([0, 500])
#plt.xticks(plt.gca().get_xlim())

plt.xlabel('Weight update (EGD)\nor epoch (BGD)' )
plt.legend()
plt.tight_layout()
outfile_name = f'{save_fig_dir}/LossAdapt.{output_fig_format}' if not plot_relative_loss else f'{save_fig_dir}/LossAdaptRelative.{output_fig_format}'
plt.savefig(outfile_name)
plt.close()