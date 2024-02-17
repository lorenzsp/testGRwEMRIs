import numpy as np
import scipy.stats as stats
from scipy.stats import powerlaw
import matplotlib.pyplot as plt

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