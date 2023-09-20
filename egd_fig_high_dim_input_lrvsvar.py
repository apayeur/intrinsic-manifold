import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
plt.style.use('rnn4bci_plot_params.dms')

load_dir = "data/egd-high-dim-input/dim_vs_lr_higher_private_noise"
save_fig_dir = "results/egd-high-dim-input/dim_vs_lr_higher_private_noise"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

# Dimension from eigenvalue threshold
data_dims = np.load(f"{load_dir}/real_dims.npy", allow_pickle=True).item()
data_dims_input = np.load(f"{load_dir}/real_dims_input.npy", allow_pickle=True).item()

lrs = []
dims = []
dims_input = []
for k in data_dims.keys():
    lrs.append(k)
    dims.append(data_dims[k])
    dims_input.append(data_dims_input[k])
del data_dims
del data_dims_input
dims = np.array(dims)
dims_input = np.array(dims_input)

# Participation ratios
data_prs = np.load(f"{load_dir}/participation_ratios.npy", allow_pickle=True).item()
data_prs_input = np.load(f"{load_dir}/participation_ratios_input.npy", allow_pickle=True).item()

prs, prs_input = [], []
for k in data_prs.keys():
    prs.append(data_prs[k])
    prs_input.append(data_prs_input[k])
del data_prs
del data_prs_input
prs = np.array(prs)
prs_input = np.array(prs_input)

# Eigenvalues
data_evs = np.load(f"{load_dir}/eigenvalsW.npy", allow_pickle=True).item()
data_svs = np.load(f"{load_dir}/singvalsU.npy", allow_pickle=True).item()
mean_ev, mean_sv = {}, {}
sem_ev, sem_sv = {}, {}
#print(data_evs)

for k in data_evs.keys():
    data_evs[k] = [list(i) for i in data_evs[k]]
    data_evs[k] = np.abs(data_evs[k])
    mean_ev.update({k: np.mean(data_evs[k], axis=0)})
    sem_ev.update({k: np.std(data_evs[k], ddof=1) / data_evs[k].shape[0]**0.5})

    data_svs[k] = [list(i) for i in data_svs[k]]
    mean_sv.update({k: np.mean(data_svs[k], axis=0)})
    sem_sv.update({k: np.std(data_svs[k], ddof=1) / len(data_svs[k]) ** 0.5})

# Loss
loss = np.load(f"{load_dir}/loss.npy", allow_pickle=True).item()
loss = loss['loss_init']
init_loss = []
max_ev = []
for k in data_evs.keys():
    for seed in loss[k].keys():
        init_loss.append(loss[k][seed][0])
        max_ev.append(np.max(np.abs(data_evs[k][seed-1])))
print(max_ev)
plt.loglog(init_loss, max_ev, lw=0, marker='o', markersize=2)
plt.tight_layout()
plt.show()
plt.semilogy(max_ev, lw=0, marker='o', markersize=1)
plt.xlabel('All seeds across learning rates')
plt.ylabel("Maximum abs. eigenvalue of $W$")
plt.tight_layout()
plt.show()

# ---- Figures ---- #
# Dimension
for i, d in enumerate([dims, dims_input]):
    plt.figure(figsize=(45*units_convert['mm'], 45/1.25*units_convert['mm']))
    for seed in range(d.shape[1]):
        plt.plot(lrs, d[:,seed], color='grey', lw=0.5, alpha=0.5)
    plt.errorbar(lrs, np.mean(d, axis=1), yerr=np.std(d, axis=1) / d.shape[1]**0.5, lw=1, color='k')
    plt.xlabel("Learning rate")
    plt.ylabel("Dimension" if i == 0 else "Input dimension")
    plt.tight_layout()
    filename = f'{save_fig_dir}/DimensionVSLearningRate.png' if i == 0 else f'{save_fig_dir}/DimensionInputVSLearningRate.png'
    plt.savefig(filename)
    plt.close()
    plt.show()

# Eigenvals
plt.figure(figsize=(45*units_convert['mm'], 45/1.25*units_convert['mm']))
for k,v in mean_ev.items():
    #plt.errorbar(np.arange(1, len(v)+1), v, yerr=sem_ev[k], lw=1, label=k)
    plt.semilogy(np.arange(1, len(v)+1), v,lw=0.5, label=k)
plt.xlabel("Dimension")
plt.ylabel("Abs. eigenvalue")
plt.tight_layout()
plt.legend()
filename = f'{save_fig_dir}/EigenvalsW.png'
plt.savefig(filename)
plt.close()
plt.show()

# Participation ratio
for i, p in enumerate([prs, prs_input]):
    plt.figure(figsize=(45*units_convert['mm'], 45/1.25*units_convert['mm']))
    for seed in range(p.shape[1]):
        plt.plot(lrs, p[:,seed], color='grey', lw=0.5, alpha=0.5)
    plt.errorbar(lrs, np.mean(p, axis=1), yerr=np.std(p, axis=1) / p.shape[1]**0.5, lw=1, color='k')
    plt.xlabel("Learning rate")
    plt.ylabel("Participation ratio" if i == 0 else "Input PR")
    plt.tight_layout()
    filename = f'{save_fig_dir}/PRVSLearningRate.png' if i == 0 else f'{save_fig_dir}/PRInputVSLearningRate.png'
    plt.savefig(filename)
    plt.close()
    plt.show()