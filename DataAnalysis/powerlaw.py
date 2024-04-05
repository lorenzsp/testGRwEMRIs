import numpy as np
import scipy.stats as stats
from scipy.stats import powerlaw
import matplotlib.pyplot as plt

from eryn.prior import ProbDistContainer
from eryn.moves import DistributionGenerate

import os
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
os.environ["OPENBLAS_NUM_THREADS"] = str(2)
os.system("OPENBLAS_NUM_THREADS=2")

class PowerLawDistribution(object):
    """Generate power law distribution between ``min_val`` and ``max_val``

    Args:
        min_val (double): Minimum value in the power law distribution
        max_val (double): Maximum value in the power law distribution
        alpha (double): Exponent of the power law distribution
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False``, use NumPy.
            (default: ``False``)

    Raises:
        ValueError: Issue with inputs.

    """

    def __init__(self, min_val, max_val, alpha, use_cupy=False):
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val.")

        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha

        self.diff = max_val**(alpha+1) - min_val**(alpha+1)

        self.pdf_val = (alpha+1) / self.diff
        self.logpdf_val = np.log(self.pdf_val)

        self.use_cupy = use_cupy
        if use_cupy:
            try:
                import cupy as cp
            except ImportError:
                raise ValueError("CuPy not found.")

    def rvs(self, size=1):
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")

        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        rand_unif = xp.random.rand(*size)

        out = ((self.diff * rand_unif + self.min_val**(self.alpha+1)))**(1/(self.alpha+1))

        return out

    def pdf(self, x):
        out = self.pdf_val * (x**(self.alpha))

        return out

    def logpdf(self, x):
        xp = np if not self.use_cupy else cp

        out = xp.zeros_like(x)
        out[x >= self.min_val] = self.logpdf_val + self.alpha * np.log(x[x >= self.min_val])
        out[x < self.min_val] = -np.inf
        return out

    def copy(self):
        return deepcopy(self)


def powerlaw_dist(min, max, alpha=-2, use_cupy=False):
    """Generate uniform distribution between ``min`` and ``max``

    Args:
        min (double): Minimum in the uniform distribution
        max (double): Maximum in the uniform distribution
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Returns:
        :class:`UniformDistribution`: Uniform distribution.


    """
    dist = PowerLawDistribution(min, max, alpha, use_cupy=use_cupy)

    return dist
# # Parameters for the power law distribution
# min_val = 1.0
# max_val = 10.0
# alpha = -2
# sample_size = 10000

# # Create an instance of the PowerLawDistribution
# powerlaw_dist = PowerLawDistribution(min_val, max_val, alpha)

# # Draw samples from the distribution
# samples = powerlaw_dist.rvs(size=sample_size)

# # Plot histogram of samples
# plt.figure(figsize=(8, 6))
# plt.hist(samples, bins=50, density=True, alpha=0.5, color='blue', label='Histogram')

# # Plot the probability density function (PDF)
# x = np.linspace(min_val, max_val, 1000)
# pdf = powerlaw_dist.pdf(x)
# plt.plot(x, pdf, color='red', lw=2, label='PDF')

# plt.title('Histogram and PDF of Power Law Distribution')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()

from sklearn.mixture import GaussianMixture

class SklearnGaussianMixtureModel(object):
    """Generate samples from a Gaussian Mixture Model (GMM) using scikit-learn.

    Args:
        n_components (int): The number of Gaussian components in the mixture model.

    """

    def __init__(self, n_components, max_iter=1000, reg_covar=1e-20, **kwargs):
        self.gmm = GaussianMixture(n_components=n_components,max_iter=max_iter, reg_covar=reg_covar, **kwargs)

    def fit(self, data):
        self.gmm.fit(data)
        self.labels = self.gmm.predict(data)

    def rvs(self, size=1):
        return self.gmm.sample(*size)[0]

    def pdf(self, x):
        return np.exp(self.gmm.score_samples(x))

    def logpdf(self, x):
        return self.gmm.score_samples(x)

    def copy(self):
        return deepcopy(self)


# fname = "results_paper/mcmc_nonoise_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed26011996_nw16_nt1/samples.npy"
# data = np.load(fname)

# Fit Gaussian Mixture Model
# sklearn_gmm = SklearnGaussianMixtureModel(n_components=4)  # You can adjust the number of components as needed
# sklearn_gmm.fit(data)

# pdc_gmm = ProbDistContainer({(0,1,2,3,4,5,6,7,8,9,10,11,12): sklearn_gmm})

# move_gmm = DistributionGenerate({"emri":pdc_gmm})

# samp = pdc_gmm.rvs(1000)
# pdc_gmm.logpdf(samp)

# pdc = move_gmm.generate_dist["emri"]
# pdc.priors_in[(0,1,2,3,4,5,6,7,8,9,10,11,12)]
# pdc.priors_in[(0,1,2,3,4,5,6,7,8,9,10,11,12)].fit