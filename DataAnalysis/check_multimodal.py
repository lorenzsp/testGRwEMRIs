import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import corner
import os
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
os.environ["OPENBLAS_NUM_THREADS"] = str(2)
os.system("OPENBLAS_NUM_THREADS=2")

# Generate some sample data
# Replace this with your actual list of samples
fname = "results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed26011996_nw16_nt1/samples.npy"
data = np.load(fname)[:1000]#,8:11]
# data = np.load(fname)[:,6:12]
# Fit Gaussian Mixture Model
n_components = 4
gmm = GaussianMixture(n_components=n_components,max_iter=1000)  # You can adjust the number of components as needed
gmm.fit(data)

# Predict the labels
labels = gmm.predict(data)

fig = corner.corner(data, truths=gmm.means_[0]); 
for ii in range(1,n_components):
    fig = corner.corner(data, truths=gmm.means_[ii], fig=fig)
plt.savefig('check_multimodal.png')

# Print the parameters of the Gaussian components
print("Means:\n", gmm.means_)
# print("Covariances:\n", gmm.covariances_)
print("Weights:\n", gmm.weights_)

# Write means and weights to a file
np.save('check_multimodal_gmm_means.npy', np.asarray(gmm.means_))

# draw samples
num_samples = data.shape[0]
samples = gmm.sample(num_samples)[0]

fig = corner.corner(data, truths=gmm.means_[0], color='b'); 
fig = corner.corner(samples, fig=fig, color='r')
plt.savefig('check_multimodal_post.png')