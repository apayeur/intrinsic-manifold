import numpy as np
import matplotlib.pyplot as plt
from utils import units_convert
from scipy.stats import linregress
plt.style.use('rnn4bci_plot_params.dms')

# Ns = [100, 1000] #, int(1e4)]  #np.logspace(2, 4, 4, endpoint=True, dtype=int)
# v = np.empty(len(Ns))
# m = np.empty(len(Ns))
# v_W = np.empty(len(Ns))
# Fs = []
# for i, N in enumerate(Ns):
#     W = np.random.randn(N, N) / N**0.5
#     v_W[i] = W.var(ddof=1)
#     F = np.linalg.inv(np.eye(N) - W)
#     Fs.append(F)
#     m[i] = np.mean(F)
#     v[i] = F.var(ddof=1)
#     print(N)
# print('v_W', v_W)
# print('v_F', v)


#plt.plot(Ns, m, label='mean')
#plt.plot(Ns, v, label='var')
#plt.legend()
#plt.show()


# fig, axes = plt.subplots(nrows=len(Ns),
#                          figsize=(45 * units_convert['mm'], len(Ns)*45/1.6 * units_convert['mm']))#, sharex=True)
# for i, ax in enumerate(axes.ravel()):
#     d = Fs[i].ravel()
#     ax.hist(Fs[i].ravel(), bins='auto', density=True)
#     x = np.arange(min(d), max(d), 0.01)
#     ax.plot(x, 1/(2*np.pi*v[i])**0.5 * np.exp(-0.5*(x - m[i])**2 / v[i]), color='k')
#     ax.set_title(f'N = {Ns[i]}', pad=-10)
# plt.tight_layout()
#plt.savefig(f'results/InitialRecWeightDist.png')

rng = np.random.default_rng(432)
nb_samples = 100

plt.figure(figsize=(45 * units_convert['mm'], 45/1.25 * units_convert['mm']))
Ns = [100, 200, 500]
for N in Ns:
    exponents = list(np.arange(0.5, 0.545, 0.01)) + list(np.linspace(0.55, 1., 5, endpoint=True))
    #var_F = {exp: [np.std(np.linalg.inv(np.eye(N) - rng.standard_normal(size=(N, N)) / N**exp)) for _ in range(nb_samples)] for exp in exponents}
    var_F = {exp: [] for exp in exponents}
    for _ in range(nb_samples):
        raw_W = rng.standard_normal(size=(N, N))
        for exp in exponents:
            var_F[exp].append(np.std(np.linalg.inv(np.eye(N) - raw_W / N**exp)))

    m = []
    sem = []
    for exp in exponents:
        m.append(np.mean(var_F[exp]))
        sem.append(np.std(var_F[exp], ddof=1) / nb_samples ** 0.5)
        #m.append(np.mean(np.asarray(var_F[exp])/np.asarray(var_F[1.])))
        #sem.append(np.std(np.asarray(var_F[exp])/np.asarray(var_F[1.]), ddof=1) / nb_samples ** 0.5)

    #plt.loglog(exponents, m, label=f"N = {N}")
    plt.plot(exponents, m, label=f"N = {N}")
    #plt.errorbar(exponents, m, yerr=sem, label=f"N = {N}")
plt.xlabel('Exponent')
plt.ylabel('Std of $(I - W)^{-1}$')
plt.legend()
plt.tight_layout()
plt.savefig(f'results/StdQ_vs_Exp.png')

"""
exp = 0.75
Ns = np.arange(100, 1050, 100) #[100, 200, 500, 1000]
var_F = {N: [np.var(np.linalg.inv(np.eye(N) - rng.standard_normal(size=(N, N)) / N**exp)) for _ in range(nb_samples)] for N in Ns}
m = []
sem = []
for N in Ns:
    m.append(np.mean(var_F[N]))
    sem.append(np.std(var_F[N], ddof=1) / nb_samples ** 0.5)

result = linregress(np.log10(Ns), np.log10(m))
print(result.slope)

plt.figure(figsize=(45 * units_convert['mm'], 45/1.25 * units_convert['mm']))
#plt.errorbar(Ns, m, yerr=sem, color='k')
plt.loglog(Ns, m)
plt.loglog(Ns, 10**(result.intercept + result.slope*np.array(np.log10(Ns))), ':', color='orange', lw=1.5)
plt.xlabel('N')
plt.ylabel('Var of $(I - W)^{-1}$')
plt.text(0.9, 0.6, f"slope = {result.slope:.3}", ha='right', transform=plt.gca().transAxes)
plt.title(f'Exponent = {exp}', pad=0)
plt.tight_layout()
plt.savefig(f'results/VarQ_vs_N_Exp{exp}.png')
"""