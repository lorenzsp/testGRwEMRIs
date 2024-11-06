import pandas as pd
import corner
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

list_fpath = ['search_results/search15_rndStart_M1e+06_mu5.0_a0.95_p4.8_e0.35_x1.0_delta0.1_SNR15.0_T0.5_seed2601_nw32_wind86400.0_best_values.txt',
'search_results/search25_rndStart_M1e+06_mu5.0_a0.95_p4.8_e0.35_x1.0_delta0.1_SNR25.0_T0.5_seed2601_nw32_wind86400.0_best_values.txt',
'search_results/search50_rndStart_M1e+06_mu5.0_a0.95_p4.8_e0.35_x1.0_delta0.1_SNR50.0_T0.5_seed2601_nw32_wind86400.0_best_values.txt']
# Load the data from the file
# file_path = 'search_results/search50_rndStart_M1e+06_mu5.0_a0.95_p4.8_e0.35_x1.0_delta0.1_SNR50.0_T0.5_seed2601_nw32_wind86400.0_best_values.txt'
for file_path in list_fpath:
    data = pd.read_csv(file_path, sep=', ')

    # Extract the relevant columns
    variables = ['lnM', 'lnmu', 'a', 'p0', 'e0', 'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 'Phi_r0']
    params_subset = data[variables]
    print(params_subset.head())
    truths = params_subset.iloc[0]
    # optimization function
    opt_fun = data['injTFstat'].to_numpy()
    # duration window
    dwindow = data['duration'].to_numpy()
    mask_window = (dwindow == 86400.0)
    true_opt_fun = opt_fun[mask_window][0]
    print('True optimization function:', true_opt_fun)

    # plto the optimization function as a function of the iteration
    plt.figure()
    # for loop over different windows
    duration_window = 86400.0
    for dw in [duration_window/4, duration_window/2, duration_window, duration_window*2, duration_window*4, duration_window*8, duration_window*16]:
        mask_window = dwindow == dw
        plt.plot(np.arange(len(opt_fun[mask_window]))[1:], -opt_fun[mask_window][1:], '.', label=f'window {dw/duration_window:.2f} days')
    # plt.colorbar()
    plt.xlabel('Iteration')
    plt.ylabel('Matched SNR')
    plt.legend()
    plt.savefig(file_path.replace('.txt', '_optimization_function.png'))

    # print information about the data
    print('iterations', params_subset.shape[0])

    start, end = 1, params_subset.shape[0]
    mask = np.arange(start, end)
    # Create the corner plot
    fig = corner.corner(params_subset.to_numpy()[mask], color='white', 
                        truths=truths, labels=variables, hist_kwargs={'color':'k'},
                        show_titles=True, title_fmt=".2e", plot_datapoints=False, show_contours=False, 
                        bins=30)

    # Plot the path of the optimized variables
    for i in range(len(variables)):
        for j in range(i):
            ax = fig.axes[i * len(variables) + j]
            ax.plot(params_subset[variables[j]][mask], params_subset[variables[i]][mask], 'o-', markersize=2, lw=0.5)

    plt.savefig(file_path.replace('.txt', '_corner.png'))