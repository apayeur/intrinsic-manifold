import numpy as np
import matplotlib.pyplot as plt
from utils import units_convert

plt.style.use('rnn4bci_plot_params.dms')

Ns = [10, 50, 100, 150]
nb_reals = int(5e2)

exponent_Ws = np.arange(0.5, 1.05, 0.05) #[0.5, 0.6, 0.7, 0.8, 0.9, 1]

prob_max_abs_eig_greater_than_one = {N: [0 for expo in exponent_Ws] for N in Ns}

for N in Ns:
    print(f"Computing for N = {N}")
    for j, expo in enumerate(exponent_Ws):
        print(f"   Computing for expo = {expo}")
        for i in range(nb_reals):
            max_abs_eigval = np.max(np.abs(np.linalg.eigvals(np.random.randn(N, N)/N**expo)))
            prob_max_abs_eig_greater_than_one[N][j] += int(max_abs_eigval > 1)
        prob_max_abs_eig_greater_than_one[N][j] /= nb_reals

# plot
plt.figure(figsize=(45*units_convert['mm'], 36 * units_convert['mm']))
for N in Ns:
    plt.plot(exponent_Ws, prob_max_abs_eig_greater_than_one[N], label=f'N = {N}')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"Probability max $|\lambda(W)| > 1$")
plt.tight_layout()
plt.legend()
plt.savefig("results/ProbMaxAbsEigvalGreaterThanOne.png")