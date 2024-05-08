import numpy as np
import glob

cov_files = glob.glob("mcmc_runs/*covariance_new.npy")
print(cov_files)
# # List of covariance files
# cov_files = [
#     "final_results/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy",
#     "results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy",
#     "results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy",
#     "results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy",
#     "results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy",
#     "results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy",
#     "results_paper/mcmc_rndStart_M5e+05_mu5.0_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_covariance.npy"
# ]

# Load the first covariance matrix
combined_cov = np.load(cov_files[0])

# Load and add the rest of the covariance matrices
for cov_file in cov_files[1:]:
    cov = np.load(cov_file)
    combined_cov += cov

# Divide by the number of covariance matrices to get the average
combined_cov /= len(cov_files)

# Save the combined covariance matrix
np.save("covariance.npy", combined_cov)