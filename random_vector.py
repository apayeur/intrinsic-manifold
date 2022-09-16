import numpy as np

class MultivariateNormal:
    """Mostly a wrapper for np.random.multivariate_normal"""
    def __init__(self, mean=None, cov=None, size=None, rng=None):
        """Define a multivariate normal instance with vector-mean `mean` and covariance matrix `cov`
        TODO: rework this!!!
        :param mean: mean of the distribution
        :param cov: covariance matrix
        :param size: number of elements of the random vector
        """
        if mean is None and cov is None and size is None:
            self.size = 1
            self.mean = 0
            self.cov = 1
        elif size is not None and mean is None and cov is None:
            self.size = size
            self.mean = np.zeros(size)
            self.cov = np.eye(size)
        elif size is None and cov is None and mean is not None:
            self.size = len(mean)
            self.mean = np.array(mean)
            self.cov = np.eye(self.size)
        elif size is None and mean is None and cov is not None:
            self.size = np.shape(cov)[0]
            self.cov = np.array(cov)
            self.mean = np.zeros(self.size)
        elif size is None and mean is not None and cov is not None:
            self.size = len(mean)
            self.mean = mean
            self.cov = cov
        else:
            raise RuntimeError('Must provide mean and/or cov, or size')

        self.rng = np.random.default_rng() if rng is None else rng

    def __repr__(self):
        return f'Multivariate random vector with mean {self.mean} and covariance {self.cov}'

    def draw(self, size=1):
        """Draw sample of size `size` of the multivariate Gaussian
        :param size: (int) size of the sample
        :return: 2D array with shape (size, self.size)
        """
        return self.rng.multivariate_normal(self.mean, self.cov, size=size)


class GaussianMixture:
    def __init__(self, means, covs, p=None, rng=None):
        self.means = means
        self.covs = covs
        self.K = len(means)
        self.size = len(self.means[0])
        if len(covs) != self.K:
            raise RuntimeError(f'Mismatch between len(means)={self.K} and len(covs)={len(covs)}')
        if p is None:
            self.p = [1/self.K]*self.K
        else:
            self.p = p
        self.components = [MultivariateNormal(m, c) for m, c in zip(means, covs)]
        self.rng = np.random.default_rng() if rng is None else rng

    def draw(self, size=1):
        """Draw sample of size `size` of the Gaussian mixture
        :param size: (int) size of the sample
        :return: 2D array with shape (size, self.size)
        """
        components_freq = self.rng.multinomial(size, self.p)
        sample = np.empty(shape=(size, self.size))
        count = 0
        for c, s in enumerate(components_freq):
            sample[count:count+s, :] = self.components[c].draw(s)
            count += s
        return sample, components_freq

    def draw_component(self, component_i, size=1):
        return self.components[component_i].draw(size)

    def draw_each_component_once(self):
        sample = np.empty(shape=(self.K, self.size))
        for component_i in range(self.K):
            sample[component_i, :] = self.components[component_i].draw(1)
        return sample

    def global_mean(self):
        m = np.zeros_like(self.means[0])
        for i, p in enumerate(self.p):
            m += p * self.means[i]
        return m