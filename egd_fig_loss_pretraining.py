import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w, convert_uneven_lists_to_array
import os
plt.style.use('rnn4bci_plot_params.dms')

exponents_W = [0.55, 1.0] #, 0.6, 0.7, 0.8, 0.9, 1]
diff_relative_loss = {exponent_W: [] for exponent_W in exponents_W}
output_fig_format = 'png'
load_dir_suffix = ""  # "-lr0.001-M6-iterAdapt500"

for exponent_W in exponents_W:
    tag = f"pretraining-N100-Nreadouts100-activationlinear-V1-expW{exponent_W}"
    model_type = "egd"
    load_dir = f"data/{model_type}/{tag}"
    save_fig_dir = f"results/{model_type}/{tag}"
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)

    params = np.load(f"{load_dir}/params.npy", allow_pickle=True).item()

    data = np.load(f"{load_dir}/data.npy", allow_pickle=True).item()

    loss_init = data['pretraining']['loss']

    if isinstance(loss_init, list):
        loss_init = convert_uneven_lists_to_array(loss_init)

    x_label = 'Weight update post-perturb.' if 'egd' in load_dir else 'Epoch'

    # ------------------------ Loss-related figures ------------------------ #
    # Plot initial loss
    plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']/1.25))
    for i, l in enumerate(loss_init):
        #plt.semilogy(l, color='black', lw=0.5)
        plt.semilogy(l, lw=0.5, label=f'seed {i}')
    plt.legend()
    plt.xlabel('Weight update')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'{save_fig_dir}/InitialLoss.{output_fig_format}')
    plt.close()
