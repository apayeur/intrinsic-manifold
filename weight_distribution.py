import numpy as np
import matplotlib.pyplot as plt
from utils import units_convert

plt.style.use('rnn4bci_plot_params.dms')

Ns = [100, 1000] #, int(1e4)]  #np.logspace(2, 4, 4, endpoint=True, dtype=int)
v = np.empty(len(Ns))
m = np.empty(len(Ns))
v_W = np.empty(len(Ns))
Fs = []
for i, N in enumerate(Ns):
    W = np.random.randn(N, N) / N**0.5
    v_W[i] = W.var(ddof=1)
    F = np.linalg.inv(np.eye(N) - W)
    Fs.append(F)
    m[i] = np.mean(F)
    v[i] = F.var(ddof=1)
    print(N)
print('v_W', v_W)
print('v_F', v)


#plt.plot(Ns, m, label='mean')
#plt.plot(Ns, v, label='var')
#plt.legend()
#plt.show()


fig, axes = plt.subplots(nrows=len(Ns),
                         figsize=(45 * units_convert['mm'], len(Ns)*45/1.6 * units_convert['mm']))#, sharex=True)
for i, ax in enumerate(axes.ravel()):
    d = Fs[i].ravel()
    ax.hist(Fs[i].ravel(), bins='auto', density=True)
    x = np.arange(min(d), max(d), 0.01)
    ax.plot(x, 1/(2*np.pi*v[i])**0.5 * np.exp(-0.5*(x - m[i])**2 / v[i]), color='k')
    ax.set_title(f'N = {Ns[i]}', pad=-10)
plt.tight_layout()
#plt.savefig(f'results/InitialRecWeightDist.png')


