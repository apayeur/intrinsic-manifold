import numpy as np
import matplotlib.pyplot as plt
target_colors = [(200./255, 0, 0),  # red
                 (0.9, 0.6, 0),  # orange
                 (0.95, 0.9, 0.25),  # yellow
                 (0, 158./255, 115./255),  # bluish green
                 (86./255, 180./255, 233./255),  # sky_blue
                 (0, 0.45, 0.7),  # blue
                 (75. / 255, 0., 146. / 255),  # purple
                 (0.8, 0.6, 0.7)]  # pink
col_o = (0, 0.45, 0.7)
col_w = (200./255, 0, 0)
perturbation_colors = {'WM': (200./255, 0, 0), 'OM': (0, 0.45, 0.7)}
units_convert = {'cm': 1 / 2.54, 'mm': 1 / 2.54 / 10}


class SVD:
    def __init__(self, X):
        self.X = X
        self.U, self.S, self.VT = np.linalg.svd(X)


def principal_angles(A, B):
    """Follows the Björk & Golub algorithm"""
    assert (A.shape[0] == B.shape[0])
    if A.shape[1] < B.shape[1]:
        tmpA, tmpB = A, B
        A, B = tmpB, tmpA
        del tmpA
        del tmpB
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)
    M = Q_A.T @ Q_B
    Y, c, ZT = np.linalg.svd(M)

    # Check for round-off errors
    for i in range(len(c)):
        if c[i] > 1.:
            c[i] = 1.
        elif c[i] < 0.:
            c[i] = 0.
    return np.arccos(c)



def get_permutation_matrix(size, rng=np.random.default_rng()):
    indices = np.arange(size)
    rng.shuffle(indices)
    P = np.zeros(shape=(size, size))
    for i in range(size):
        P[i, indices[i]] = 1.
    return P, indices

def heap_permutation(k, arr):
    """Heap's algorithm to generate permutations"""
    if k == 1:
        yield arr
    else:
        heap_permutation(k-1, arr)
        for i in range(k-1):
            if k % 2 == 0:
                arr[i], arr[k-1] = arr[k-1], arr[i]
            else:
                arr[0], arr[k-1] = arr[k-1], arr[0]
            heap_permutation(k - 1, arr)

def gram_schmidt(list_of_vectors):
    def vector_projection(a, u):
        return (a @ u) / np.linalg.norm(u) ** 2 * u

    ortho_list = []
    for i in range(len(list_of_vectors)):
        if i == 0:
            ortho_list.append(list_of_vectors[i] / np.linalg.norm(list_of_vectors[i]))
        else:
            sum_of_proj = np.zeros_like(list_of_vectors[i])
            for j in range(i):
                sum_of_proj += vector_projection(list_of_vectors[i], list_of_vectors[j])
            list_of_vectors[i] -= sum_of_proj
            ortho_list.append(list_of_vectors[i] / np.linalg.norm(list_of_vectors[i]))
    return ortho_list

# --- Plotting functions --- #
def plot_variances(iter_var, pop_shared_var, pop_private_var, dimension_var, outfile_name):
    #f'{results_folder}/SharedVariance.png'
    fig, axes = plt.subplots(ncols=2, figsize=(4, 2 / 1.25), sharex=True)
    axes[0].plot(iter_var, pop_shared_var, label='Shared')
    axes[0].plot(iter_var, pop_private_var, label='Private')
    axes[1].plot(iter_var, dimension_var)
    for ax in axes:
        ax.set_xlabel('Iteration')
    axes[0].legend()
    axes[0].set_ylabel('Fraction of\npopulation variance')
    axes[1].set_ylabel('Dimension')
    plt.tight_layout()
    plt.savefig(outfile_name)
    plt.close()

def plot_projections(potent_space_projections, null_space_projections, outfile_name):
    fig, axes = plt.subplots(ncols=2, figsize=(4, 2 / 1.25), sharex=True, sharey=True)
    for k in potent_space_projections.keys():
        nb_iterations = len(potent_space_projections[k])
        if k not in ['b', 'db']:
            axes[0].plot(np.arange(nb_iterations), potent_space_projections[k], label=k, lw=0.5)
    for k in null_space_projections.keys():
        nb_iterations = len(potent_space_projections[k])
        if k not in ['b', 'db']:
            axes[1].plot(np.arange(nb_iterations), null_space_projections[k], label=k, lw=0.5)
    for ax in axes:
        ax.set_xlabel(r'Iteration')
        # ax.set_xlim([0, 1000])
    axes[0].legend()
    axes[0].set_ylabel('Normalized Frobenius norm\nof output-potent projection')
    axes[1].set_ylabel('Normalized Frobenius norm\nof output-null projection')
    plt.tight_layout()
    plt.savefig(outfile_name)
    plt.close()

def plot_svd(singular_values, outfile_name):
    fig, axes = plt.subplots(ncols=2, figsize=(4, 2 / 1.25))
    N = len(singular_values['dW'][0])
    for i, iteration_num in enumerate(singular_values['iteration']):
        axes[0].semilogy(np.arange(1, N + 1), singular_values['dW'][i], 'o', linestyle='solid',
                         label=f'dW | iteration {iteration_num}', lw=0.5, markersize=2,
                         color=target_colors[i % len(target_colors)])
        axes[0].semilogy(np.arange(1, N + 1), singular_values['dU'][i], 'x', linestyle='dashed',
                         label=f'dU | iteration {iteration_num}', lw=0.5, markersize=2,
                         color=target_colors[i % len(target_colors)])
        axes[1].semilogy(np.arange(1, N + 1), singular_values['W'][i], 'o', linestyle='solid',
                         label=f'W | iteration {iteration_num}',
                         lw=0.5, markersize=2, color=target_colors[i % len(target_colors)])
        axes[1].semilogy(np.arange(1, N + 1), singular_values['U'][i], 'x', linestyle='dashed',
                         label=f'U | iteration {iteration_num}',
                         lw=0.5, markersize=2, color=target_colors[i % len(target_colors)])
    axes[0].set_ylabel('Singular value')
    for ax in axes:
        ax.set_xlabel('Rank')
        ax.legend()
    axes[0].set_xlim([1 - 0.1, N//5 + 0.1])
    axes[1].set_xlim([1 - 0.5, N + 0.5])
    plt.tight_layout()
    plt.savefig(outfile_name)
    plt.close()

def plot_performance(u, targets, freqs, filename):
    plt.figure(figsize=(2, 2))
    if u.shape[0] == 2:
        for i in range(len(freqs) - 1):
            plt.scatter(u[0, freqs[i]:freqs[i + 1]], u[1, freqs[i]:freqs[i + 1]], s=5,
                        facecolor=target_colors[i], edgecolors='white', lw=0.2)
            plt.scatter(targets[0, freqs[i]:freqs[i + 1]], targets[1, freqs[i]:freqs[i + 1]], s=10,
                        facecolor=target_colors[i], edgecolors='black', lw=0.4, zorder=10)
    elif u.shape[0] == 1:
        for i in range(len(freqs) - 1):
            plt.scatter(u[freqs[i]:freqs[i + 1]], np.zeros_like(u[freqs[i]:freqs[i + 1]]), s=5,
                        facecolor=target_colors[i], edgecolors='white', lw=0.2)
            plt.scatter(targets[freqs[i]:freqs[i + 1]], np.zeros_like(targets[freqs[i]:freqs[i + 1]]), s=10,
                        facecolor=target_colors[i], edgecolors='black', lw=0.4, zorder=10)
    plt.xlabel('$u_x$')
    plt.ylabel('$u_y$')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

