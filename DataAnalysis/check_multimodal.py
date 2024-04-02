import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

labels = [r'$\ln (M/{\rm M}_\odot$)', r'$\ln (\mu / M_{\odot})$', r'$a$', r'$p_0 \, [M]$', r'$e_0$', 
            r'$D_L \, [{\rm Gpc}]$',
            r"$\cos \theta_S$",r"$\phi_S$",
            r"$\cos \theta_K$",r"$\phi_K$",
        r'$\Phi_{\varphi 0}$', r'$\Phi_{r 0}$',
            r"$d$",
        ]

CORNER_KWARGS = dict(
    labels=labels,
    bins=40,
    label_kwargs=dict(fontsize=35),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    show_titles=False,
    max_n_ticks=4,
    truth_color='k',
    labelpad=0.3,
)

import corner
import os
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
os.environ["OPENBLAS_NUM_THREADS"] = str(2)
os.system("OPENBLAS_NUM_THREADS=2")

# Generate some sample data
# Replace this with your actual list of samples
fname = "results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed26011996_nw16_nt1/samples.npy"
data = np.load(fname)[:]#,8:11]
# data = np.load(fname)[:,6:12]
# Fit Gaussian Mixture Model
n_components = 4
gmm = GaussianMixture(n_components=n_components, max_iter=200, reg_covar=1e-20)
gmm.fit(data)

# Print the parameters of the Gaussian components
print("GMM Means:\n", gmm.means_)
print("GMM Covariances:\n", gmm.covariances_)
print("GMM Weights:\n", gmm.weights_)

# Write GMM means and weights to a file
np.save('check_multimodal_gmm_means.npy', np.asarray(gmm.means_))

# Plot corner plot for GMM
# fig = corner.corner(data, truths=gmm.means_[0], color='b')
# for i in range(1, n_components):
#     fig = corner.corner(data[:100], truths=gmm.means_[i], fig=fig, color='b')
# plt.savefig('check_multimodal_gmm.png')

# Draw samples from GMM
import time
tic = time.time()
gmm_samples = gmm.sample(data.shape[0])[0]
toc = time.time()
print("sampled ", data.shape, "in ", toc-tic, "seconds")

# Plot corner plot for GMM samples
fig = corner.corner(data, color='b', **CORNER_KWARGS)
fig = corner.corner(gmm_samples, fig=fig, color='r', **CORNER_KWARGS)
plt.savefig('check_multimodal_gmm_samples.png')