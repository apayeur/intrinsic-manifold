import matplotlib.pyplot as plt
import numpy as np
from utils import units_convert, col_o, col_w
import os
import copy
plt.style.use('rnn4bci_plot_params.dms')

output_fig_format = 'png'

exponent = 0.6
suffix = f"exponent_W{exponent}-lr0.001-M6-iterAdapt1000"
load_dir = f"data/egd/{suffix}"
save_fig_dir = f"results/egd/{suffix}"
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

evs = np.load(f"{load_dir}/follow_eigvals.npy", allow_pickle=True).item()
_, nb_updates, _ = evs['initial'].shape
nb_seeds, nb_updates_adapt, nb_units = evs['WM'].shape
nb_seeds=1
# Check whether any eigvals became unstable
for type_ in ['initial', 'WM', 'OM']:
    for seed in range(nb_seeds):
        if np.max(np.abs(evs[type_][seed])) >= 1:
            print("Found EVs greater than 1")


# Plot nb of real values vs updates
plt.figure(figsize=(85/2*units_convert['mm'], 85/2*units_convert['mm']/1.25))
# initial
nb_real_eigvals = np.empty(shape=(nb_seeds, nb_updates))
for seed in range(nb_seeds):
    for i in range(nb_updates):
        nb_real_eigvals[seed, i] = np.sum(abs(np.imag(evs['initial'][seed, i, :])) < 1e-8)
m = np.mean(nb_real_eigvals, axis=0)
sem = np.std(nb_real_eigvals, axis=0, ddof=1) / nb_seeds**0.5
plt.plot(np.arange(-nb_updates, 0), m, color='grey', label='initial')
plt.fill_between(np.arange(-nb_updates, 0), m - 2*sem, m + 2*sem, color='grey', alpha=0.5, lw=0)
# adapt
for perturbation_type in ['WM', 'OM']:
    nb_real_eigvals = np.empty(shape=(nb_seeds, nb_updates_adapt))
    for seed in range(nb_seeds):
        for i in range(nb_updates_adapt):
            nb_real_eigvals[seed, i] = np.sum(abs(np.imag(evs[perturbation_type][seed, i, :])) < 1e-8)
    m = np.mean(nb_real_eigvals, axis=0)
    sem = np.std(nb_real_eigvals, axis=0, ddof=1) / nb_seeds ** 0.5
    plt.plot(np.arange(nb_updates_adapt), m, color=col_w if perturbation_type=='WM' else col_o, label=perturbation_type)
    plt.fill_between(np.arange(nb_updates_adapt), m - 2 * sem, m + 2 * sem, color=col_w if perturbation_type=='WM' else col_o, alpha=0.5, lw=0)
plt.xlim([-nb_updates, nb_updates_adapt])
plt.xticks([-nb_updates, 0, nb_updates_adapt])
plt.xlabel('Weight update post-perturb.')
plt.ylabel('Nb of real eigenvalues $W$')
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/NbRealValuesOM.{output_fig_format}')
plt.close()


# Plot eigval dynamics during adaptation
selected_seed = 0
perturbation_type = 'OM'
# re-order evs
evs = evs[perturbation_type][selected_seed]
for i in range(evs.shape[0]):
    evs[i] = np.sort_complex(evs[i])
"""
nb_real_eigvals = np.zeros(nb_updates_adapt, dtype=int)
real_eigvals = [evs[0, np.abs(np.imag(evs[0])) < 1e-8]]
nb_real_eigvals[0] = len(real_eigvals[0])

for i in range(1, evs.shape[0]):
    evs_ordered = 10*np.ones(evs.shape[1], dtype=complex)  # 10 is arbitrary, as a placeholder for dealing w/ creation/annihilation of evs
    nb_real_eigvals[i] = np.sum(np.abs(np.imag(evs[i])) < 1e-8)
    real_eigvals.append(evs[i, np.abs(np.imag(evs[i])) < 1e-8])

    if nb_real_eigvals[i-1] != nb_real_eigvals[i]:
        if nb_real_eigvals[i] > nb_real_eigvals[i-1]:
            print(f"{nb_real_eigvals[i] - nb_real_eigvals[i-1]} real eigenvalues created")
            appeared_evs = []
            appeared_evs_indices = []
            for vp_i, vp in enumerate(real_eigvals[i]):
                if np.sum(np.isclose(vp, real_eigvals[i], atol=1e-2)) >= 2:
                    appeared_evs.append(vp)
                    appeared_evs_indices.append(vp_i)
            for j in range(evs.shape[1]):
                if j not in appeared_evs_indices:
                    index_min = np.argmin(np.abs(evs[i - 1] - evs[i, j]))
                    evs_ordered[index_min] = evs[i, j]
            indices_tmp = np.where(np.isclose(evs_ordered, 10))[0]
            print(indices_tmp)
            print(appeared_evs_indices)
            for k, indices_tmp_i in enumerate(indices_tmp):
                evs_ordered[indices_tmp_i] = appeared_evs[k]

        else:
            print(f"{nb_real_eigvals[i-1] - nb_real_eigvals[i]} real eigenvalues destroyed")
            disappeared_evs = []
            disappeared_evs_indices = []
            for vp_i, vp in enumerate(real_eigvals[i-1]):
                if np.sum(np.isclose(vp, real_eigvals[i-1], atol=1e-2)) >= 2:
                    disappeared_evs.append(vp)
                    disappeared_evs_indices.append(vp_i)
            for j in range(evs.shape[1]):
                if j not in disappeared_evs_indices:
                    index_min = np.argmin(np.abs(evs[i] - evs[i-1, j]))
                    evs_ordered[j] = evs[i, index_min]
            indices_tmp = np.where(np.isclose(evs_ordered, 10))[0]
            for k, indices_tmp_i in enumerate(indices_tmp):
                evs_ordered[indices_tmp_i] = disappeared_evs[k]
    else:
        for j in range(evs.shape[1]):
            index_min = np.argmin(np.abs(evs[i-1] - evs[i, j]))
            evs_ordered[index_min] = evs[i, j]
    evs[i] = evs_ordered 
"""
plt.figure(figsize=(85/2*units_convert['mm'], 85/2*units_convert['mm']))
plt.gca().set_aspect('equal')
plt.plot(np.real(evs[0, :]), np.imag(evs[0, :]), lw=0, marker='o', markersize=1, mew=0, color='k')

for i in range(evs.shape[1]):
    plt.plot(np.real(evs[:, i]), np.imag(evs[:, i]), lw=0.5, color=col_o)
    #plt.plot(np.real(evs[:, i]), lw=0.3,label='real')
    #plt.plot(np.imag(evs[:, i]), lw=0.3, label='imag')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xticks([-1,1])
plt.yticks([-1,1])
#plt.plot(np.imag(evs[:, i]), lw=0.5, alpha=0.5, label='imag')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.tight_layout()
plt.savefig(f'{save_fig_dir}/FollowEVs_{perturbation_type}_seed{selected_seed}.{output_fig_format}')
plt.close()
