"""
Description:
-----------
Simplified implementation of a BCI task with a RNN trained using node perturbation, with manifold-based perturbations.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import time
from numba import njit
import copy
from utils import target_colors
plt.style.use('../../rnn4bci_plot_params.dms')


# --- "GLOBAL" PARAMETERS AND VARIABLES --- #
dt = 0.01                                                       # integration time step


# --- TASK DEFINITION --- #
# Parameters
endpoint_workspace_dimension = 2                                # number of dimensions of endpoint workspace (2 => 2 dimensions)
nb_targets = 6                                                  # number of targets
duration = 100                                                  # movement duration
distance = 1                                                    # distance of peripheral reach targets from center (cm)
# Targets
targets = np.empty((nb_targets, endpoint_workspace_dimension))
targets[:, 0] = distance * np.cos(2*np.pi*np.arange(nb_targets)/nb_targets)
targets[:, 1] = distance * np.sin(2*np.pi*np.arange(nb_targets)/nb_targets)
# Input lines
inp = np.eye(nb_targets)


# --- NETWORK PARAMETERS --- #
n_in, n_rec, n_out = nb_targets, 100, endpoint_workspace_dimension      # dimensions of network
tau_m = 0.05/dt                                                         # single-neuron time constant
noise_amplitude = tau_m*0.01*50/n_rec                                   # single-neuron noise (for noise perturbation)


# --- LEARNING PARAMETERS --- #
lr = 0.5e-1                                                         # scalar learning rate
# Cost/reward-related parameters
alpha_reward = 0.3                                              # filtering factor of reward traces
rel_hyperparams = (0.1, 0.)                                     # relative weight of velocity- and force-loss


@njit
def bci_states(velocities, ics):
    """BCI dynamics, with state s = [x, y, xdot, ydot]

    :param velocities:   velocities [shape : (duration, 2)]
    :param ics:     initial conditions [shape : (4, )]

    :return:
        arm states for all times [shape : (duration, 4)]
    """
    s = np.zeros((velocities.shape[0]+1, 4))
    s[0] = np.zeros(4)
    s[1:,2:] = velocities
    for t in range(velocities.shape[0]):
        s[t+1, :2] = s[t, :2] + dt * s[t, 2:]
    return s

@njit
def end_cost(s_end, target, relative_hyperparameters):
    return (s_end[:2] - target) @ (s_end[:2] - target) + \
           relative_hyperparameters[0] * s_end[2:4] @ s_end[2:4]

@njit
def forward(x, U, W, V, b):
    u = np.zeros(n_rec)
    controls = np.zeros((duration, 2))
    a = 1 / tau_m

    for t in range(1, duration + 1):
        xi = noise_amplitude * np.random.randn(n_rec)
        u = (1 - a) * u + a * (U @ x + W @ np.tanh(u) + b + xi)
        h = np.tanh(u)
        o = V @ h
        controls[t - 1] = o
    return controls

def get_average_loss(U, W, V, b):
    a = 1 / tau_m
    loss = 0.
    for i in range(nb_targets):
        controls = np.zeros((duration, 2))
        v = np.zeros(n_rec)
        for t in range(1, duration + 1):
            v = (1 - a) * v + a * (U @ inp[i] + W @ np.tanh(v) + b)
            controls[t-1] = V @ np.tanh(v)
        s = bci_states(controls, np.zeros(4))
        loss += end_cost(s[-1], targets[i], rel_hyperparams)
    return loss/nb_targets

@njit
def sample_model(U, W, V, b, nb_epochs=1):
    activities = np.empty(shape=(nb_epochs*nb_targets*duration, n_rec))
    velocities = np.empty(shape=(nb_epochs*nb_targets*duration, n_out))
    a = 1 / tau_m
    sample_count = 0
    for e in range(nb_epochs):
        for i in range(nb_targets):
            v = np.zeros(n_rec)
            for t in range(1, duration + 1):
                xi = noise_amplitude * np.random.randn(n_rec)
                v = (1 - a) * v + a * (U @ inp[i] + W @ np.tanh(v) + b + xi)
                h = np.tanh(v)
                velocities[sample_count, :] = V @ h
                activities[sample_count, :] = h
                sample_count += 1
    return activities, velocities

@njit
def compute_eligibility_traces(x, target, U, W, V, b):
    total_loss = 0.

    # Initialize network variables and controls
    u = np.zeros(n_rec)
    h = u  # np.tanh(u)
    o = V @ h
    controls = np.zeros((duration, len(o)))
    etU, etW, etb = np.zeros_like(U), np.zeros_like(W), np.zeros_like(b)

    # Run trial
    for t in range(1, duration + 1):
        h_prev, o_prev = h, o
        a = 1 / tau_m
        xi = noise_amplitude * np.random.randn(n_rec)
        u = (1 - a) * u + a * (U @ x + W @ np.tanh(u) + b + xi)
        h = np.tanh(u)
        o = V @ h
        controls[t - 1] = o
        # Update traces
        etW += np.outer(xi, h_prev)
        etU += np.outer(xi, x)
        etb += xi
    # Reward
    R = -total_loss / duration
    s = bci_states(controls, np.zeros(4))
    R -= end_cost(s[-1], target, rel_hyperparams)
    return etU, etW, etb, R

def train_epoch(U, W, V, b, learning_rates, epoch, reward_traces, learning_after_each_example):
    total_reward = 0

    gradU, gradW, gradb = np.zeros_like(U), np.zeros_like(W), np.zeros_like(b)

    for ex_id, target in enumerate(targets):
        etU, etW, etb, reward = compute_eligibility_traces(inp[ex_id], target, U, W, V, b)
        total_reward += reward

        if epoch > 5:
            gradU += (reward - reward_traces[ex_id]) * etU
            gradW += (reward - reward_traces[ex_id]) * etW
            gradb += (reward - reward_traces[ex_id]) * etb

            if learning_after_each_example:
                U += learning_rates[0] * gradU
                W += learning_rates[1] * gradW
                b += learning_rates[3] * gradb
                gradU, gradW, gradb = np.zeros_like(U), np.zeros_like(W), np.zeros_like(b)
        else:
            gradU, gradW, gradb = np.zeros_like(U), np.zeros_like(W), np.zeros_like(b)

        reward_traces[ex_id] = alpha_reward * reward_traces[ex_id] + (1 - alpha_reward) * reward  # update reward filter

    if epoch > 5 and not learning_after_each_example:
        U += learning_rates[0] * gradU
        W += learning_rates[1] * gradW
        b += learning_rates[3] * gradb
    return U, W, b, total_reward / nb_targets

def train(epochs, U, W, V, b, learning_rates=(lr, lr, 0, lr), verbose=False, interval_verbose=1, learning_after_each_example=True):
    loss = []

    reward_traces = np.zeros(nb_targets)

    for i in range(epochs):
        U, W, b, single_epoch_loss = train_epoch(U, W, V, b, learning_rates, i, reward_traces, learning_after_each_example)
        if verbose:
            if i % interval_verbose == 0:
                print('Epoch {}: loss = {}'.format(i + 1, abs(single_epoch_loss)))
        loss.append(single_epoch_loss)
    return loss, U, W, b

def fit_decoder(U, W, V, b, intrinsic_manifold_dim=None, threshold=0.99):
    activities, velocities = sample_model(U, W, V, b, nb_epochs=1)
    centered_activities = activities - np.mean(activities, axis=0)
    cov = centered_activities.T @ centered_activities / (centered_activities.shape[0] - 1)
    _, s, vt = np.linalg.svd(cov)
    del centered_activities

    cum_var = np.cumsum(s)
    #plt.semilogy(cum_var/cum_var[-1])
    #plt.show()
    indice_closest_to_threshold = np.argmin(np.abs(cum_var - threshold*cum_var[-1]))
    if cum_var[indice_closest_to_threshold] < threshold*cum_var[-1]:  # to make sure we represent at least `threshold` of total variance
        indice_closest_to_threshold += 1
    print('Dim for {} of total variance = {}'.format(threshold, indice_closest_to_threshold+1))

    if intrinsic_manifold_dim is None:
        intrinsic_manifold_dim = indice_closest_to_threshold+1

    # Projection matrix
    C = vt[:intrinsic_manifold_dim, :]

    velocities -= np.mean(velocities, axis=0, keepdims=True)
    X = activities @ C.T

    decoder = lm.LinearRegression(fit_intercept=False)
    decoder.fit(X, velocities)
    y = decoder.predict(X)
    mse = np.mean((y - velocities) ** 2)
    print('MSE = %.4f' % mse)

    # Define D and redefine V
    D = decoder.coef_
    V = D @ C

    return intrinsic_manifold_dim, D, C, V

def om_perturb(U, W, V, b, D, C, nb_samples=200, target_loss=None):
    all_losses = np.empty(nb_samples)
    indices_to_permute = np.arange(n_rec)

    if target_loss is None:
        all_indices = []

        for i in range(nb_samples):
            indices = copy.deepcopy(indices_to_permute)
            np.random.shuffle(indices)
            V = D @ C[:,indices]

            loc_loss = get_average_loss(U, W, V, b)
            all_losses[i] = loc_loss
            all_indices.append(indices)
        mean_pert_index = np.argmin(np.abs(all_losses - np.mean(all_losses)))
        V = D @ C[:, all_indices[mean_pert_index]]
    else:
        all_losses = []
        loc_loss = get_average_loss(U, W, V, b)
        while abs(loc_loss - target_loss) > 1e-3:
            indices = copy.deepcopy(indices_to_permute)
            np.random.shuffle(indices)
            V = D @ C[:, indices]
            loc_loss = get_average_loss(U, W, V, b)

    return all_losses, V

def wm_perturb(U, W, V, b, D, C, manifold_dim=10, nb_samples=200, target_loss=None):
    all_losses = np.empty(nb_samples)
    indices_to_permute = np.arange(manifold_dim)

    if target_loss is None:
        all_indices = []

        for i in range(nb_samples):
            indices = copy.deepcopy(indices_to_permute)
            np.random.shuffle(indices)
            V = D @ C[indices,:]

            loc_loss = get_average_loss(U, W, V, b)
            all_losses[i] = loc_loss
            all_indices.append(indices)
        mean_pert_index = np.argmin(np.abs(all_losses - np.mean(all_losses)))
        V = D @ C[all_indices[mean_pert_index], :]
    else:
        all_losses = []
        loc_loss = get_average_loss(U, W, V, b)
        indices = np.arange(manifold_dim)
        while abs(loc_loss - target_loss) > 1e-3:
            np.random.shuffle(indices)
            V = D @ C[indices, :]
            loc_loss = get_average_loss(U, W, V, b)

    return all_losses, V


if __name__ == '__main__':
    # Pre-training the network
    start = time.time()
    seed = 89
    np.random.seed(seed)
    W = np.random.randn(n_rec, n_rec) / n_rec ** 0.5
    U = np.random.uniform(-1, 1, size=(n_rec, n_in)) / n_in ** 0.5
    b = np.zeros(n_rec)
    V = np.random.randn(n_out, n_rec) / n_rec ** 0.5 # initial output weights matrix
    _, U, W, b = train(int(2e4), U, W, V, b, verbose=True, interval_verbose=1000, learning_after_each_example=False)
    end = time.time()
    print("Elapsed time = %s" % (end - start))

    # Plotting pre-training trajectories
    _, (ax_traj, ax_vel) = plt.subplots(ncols=2, figsize=(3, 1.5))
    for i in range(nb_targets):
        vel = forward(inp[i], U, W, V, b)
        s = bci_states(vel, np.zeros(4))
        ax_traj.plot(s[:, 0], s[:, 1], lw=1, color=target_colors[i])
        ax_traj.scatter(targets[i, 0], targets[i, 1], s=3, color=target_colors[i])
        ax_vel.plot(dt * np.arange(duration), np.sqrt(s[1:, 2] ** 2 + s[1:, 3] ** 2), color=target_colors[i])
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (a.u.)')
    ax_traj.axis('off')
    sns.despine(trim=True, ax=ax_vel)
    plt.tight_layout()
    plt.savefig(f"../../results/PreTrainingTrajectory_seed{seed}.png")
    plt.close()

    # Fit decoder
    intrinsic_manifold_dim, D, C, V_BCI = fit_decoder(U, W, V, b, intrinsic_manifold_dim=10, threshold=0.99)

    # Retrain
    start = time.time()
    _, U, W, b = train(int(1e3), U, W, V_BCI, b, verbose=True, interval_verbose=100, learning_after_each_example=False)
    end = time.time()
    print("Elapsed time = %s" % (end - start))

    # WM perturbation
    _, V_WM = wm_perturb(U, W, V_BCI, b, D, C, manifold_dim=10, nb_samples=int(1e3), target_loss=1.)

    # OM perturbation
    _, V_OM = om_perturb(U, W, V_BCI, b, D, C, nb_samples=int(1e3), target_loss=1.)

    U_copy = copy.deepcopy(U)
    W_copy = copy.deepcopy(W)
    b_copy = copy.deepcopy(b)

    # Adapt WM
    print("\nAdapt to WM perturbation")
    loss_wm, U_WM, W_WM, b_WM = train(int(2e4), U, W, V_WM, b, verbose=True, interval_verbose=1000,
                          learning_after_each_example=False)

    # Adapt OM
    print("\nAdapt to OM perturbation")
    loss_om, U_OM, W_OM, b_OM = train(int(2e4), U_copy, W_copy, V_OM, b_copy, verbose=True, interval_verbose=1000,
                          learning_after_each_example=False)

    # Plotting the trajectories WM
    _, (ax_traj, ax_vel) = plt.subplots(ncols=2, figsize=(3, 1.5))
    for i in range(nb_targets):
        vel = forward(inp[i], U_WM, W_WM, V_WM, b_WM)
        s = bci_states(vel, np.zeros(4))
        ax_traj.plot(s[:, 0], s[:, 1], lw=1, color=target_colors[i])
        ax_traj.scatter(targets[i, 0], targets[i, 1], s=3, color=target_colors[i])
        ax_vel.plot(dt*np.arange(duration), np.sqrt(s[1:, 2]**2 + s[1:, 3]**2), color=target_colors[i])
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (a.u.)')
    ax_traj.axis('off')
    sns.despine(trim=True, ax=ax_vel)
    plt.tight_layout()
    plt.savefig(f"../../results/WMTrajectory_seed{seed}.png")
    plt.close()

    # Plotting the trajectories OM
    _, (ax_traj, ax_vel) = plt.subplots(ncols=2, figsize=(3, 1.5))
    for i in range(nb_targets):
        vel = forward(inp[i], U_OM, W_OM, V_OM, b_OM)
        s = bci_states(vel, np.zeros(4))
        ax_traj.plot(s[:, 0], s[:, 1], lw=1, color=target_colors[i])
        ax_traj.scatter(targets[i, 0], targets[i, 1], s=3, color=target_colors[i])
        ax_vel.plot(dt*np.arange(duration), np.sqrt(s[1:, 2]**2 + s[1:, 3]**2), color=target_colors[i])
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (a.u.)')
    ax_traj.axis('off')
    sns.despine(trim=True, ax=ax_vel)
    plt.tight_layout()
    plt.savefig(f"../../results/OMTrajectory_seed{seed}.png")
    plt.close()

    # Plotting loss
    fig, ax = plt.subplots(figsize=(2, 2/1.25))
    ax.semilogy(-np.array(loss_wm), lw=0.5, label='WM')
    ax.semilogy(-np.array(loss_om), lw=0.5, label='OM')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    sns.despine(trim=True, ax=ax)
    plt.tight_layout()
    plt.savefig(f"../../results/LossWM_OM_seed{seed}.png")
    plt.close()