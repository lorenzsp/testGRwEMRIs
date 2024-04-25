import glob
# from eryn.backends import HDFBackend
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
import matplotlib as mpl
import re
import sys
sys.path.append('../DataAnalysis/')
# import matplotlib.style as style
# style.use('tableau-colorblind10')

default_width = 5.78853 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

import matplotlib.ticker as mticker

vals = [0.000001,0.00001,0.0001,0.01,1.0]

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

g = lambda x,pos : "${}$".format(f.set_scientific('%1.10e' % x))
fmt = mticker.FuncFormatter(g)

from scipy.constants import golden
import corner
import pandas as pd
inv_golden = 1. / golden

px = 2*0.0132

mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": 'pdflatex',
    "pgf.rcfonts": False,
    "font.family": "serif",
    "figure.figsize": [246.0*px, inv_golden * 246.0*px],
  'legend.fontsize': 12,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
  'legend.title_fontsize' : 12,

# "axes.formatter.min_exponent": 1
# "axes.formatter.offset_threshold": 10
})


import matplotlib.lines as mlines

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

labels = [r'$\Delta \ln M$', r'$\Delta \ln \mu$', r'$\Delta a$', r'$\Delta p_0 \, [M]$', r'$\Delta e_0$', 
            r'$\Delta D_L \, [{\rm Gpc}]$',
            r"$\Delta \cos \theta_S$",r"$\Delta \phi_S$",
            r"$\Delta \cos \theta_K$",r"$\Delta \phi_K$",
        r'$\Delta \Phi_{\varphi 0}$', r'$\Delta \Phi_{r 0}$',
            r"$\Delta d$",
        ]

CORNER_KWARGS = dict(
    labels=labels,
    bins=40,
    truths=np.zeros(len(labels)),
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
def overlaid_corner(samples_list, sample_labels, name_save=None, corn_kw=None, title=None, ylim=None):
    """
    Plots multiple corners on top of each other.

    Parameters:
    - samples_list: list of numpy arrays
        List of MCMC samples for each corner plot.
    - sample_labels: list of strings
        List of labels for each set of samples.
    - name_save: string, optional
        Name of the file to save the plot. If not provided, the plot will be displayed.
    - corn_kw: dict, optional
        Additional keyword arguments to pass to the corner.corner function.
    - title: string, optional
        Title for the plot.
    - ylim: tuple, optional
        The y-axis limits for the marginalized corners.

    Returns:
    - None (if name_save is not provided) or saves the plot as a PDF file.

    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('Set1',)
    colors = [cmap(i) for i in range(n)]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Define the plot range for each dimension
    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )

    # Update corner plot keyword arguments
    corn_kw = corn_kw or {}
    corn_kw.update(range=plot_range)
    list_maxy = []
    # Create the first corner plot
    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        weights=get_normalisation_weight(len(samples_list[0]), max_len),
        **corn_kw
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))
    maxy = [axes[i, i].get_ybound()[-1] for i in range(ndim)]
    # append maxy
    list_maxy.append(maxy)
    
    # Overlay the remaining corner plots
    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **corn_kw
        )
        axes = np.array(fig.axes).reshape((ndim, ndim))
        maxy = [axes[i, i].get_ybound()[-1] for i in range(ndim)]
        # append maxy
        list_maxy.append(maxy)
    list_maxy =np.asarray(list_maxy)


    # Set y-axis limits for the marginalized corners
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes[i, i].set_ylim((0.0,np.max(list_maxy,axis=0)[i]))

    # Add legend
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=35,
        frameon=False,
        bbox_to_anchor=(0.5, ndim+1),
        loc="upper right",
        title=title,
        title_fontsize=35,
    )

    # Adjust plot layout
    plt.subplots_adjust(left=-0.1, bottom=-0.1, right=None, top=None, wspace=None, hspace=0.15)


    # Save or display the plot
    if name_save is not None:
        plt.savefig(name_save+".pdf", pad_inches=0.2, bbox_inches='tight')
    else:
        plt.show()

########################### preparation of the data #############################################
init_name = 'results_paper/mcmc_rndStart_M*_charge0.0*seed2601*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
# sort datasets by charge
datasets = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1.h5', ]
# create the list of injected parameters
pars_inj = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', ]

for ii in range(len(datasets)):
    datasets[ii] = '../DataAnalysis/' +  datasets[ii]
    pars_inj[ii] = '../DataAnalysis/' +  pars_inj[ii]
    
print("len names", len(datasets),len(pars_inj))
cmap = plt.cm.get_cmap('Set1',)
colors = [cmap(i) for i in range(len(datasets))]#['black','red', 'royalblue']#
ls = ['-','--','-.',':',(0, (3, 1, 1, 1, 3))]

list_chains,labs = [], []
list_dict = []
# Scalar plot
for filename, inj_params, color in zip(datasets, pars_inj, colors):
    # Get repository name
    repo_name = inj_params.split('_injected_pars.npy')[0]
    
    # Load injected parameters
    truths = np.load(inj_params)
    
    # Load MCMC samples
    list_chains.append(np.load(repo_name + '/samples.npy') - truths[None,:])
    # obtain median and 95% credible interval from the samples
    med = np.median(np.load(repo_name + '/samples.npy'), axis=0)
    low = np.percentile(np.load(repo_name + '/samples.npy'), 2.5, axis=0)
    high = np.percentile(np.load(repo_name + '/samples.npy'), 97.5, axis=0)
    # Create a DataFrame with the parameter information
    data = {
        'estimator': ['true', 'median', 'percentile 2.5 perc', 'percentile 97.5 perc', 'precision 68%'],
        'ln M': [truths[0], med[0], low[0], high[0], np.std(np.load(repo_name + '/samples.npy'), axis=0)[0] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[0]],
        'ln mu': [truths[1], med[1], low[1], high[1], np.std(np.load(repo_name + '/samples.npy'), axis=0)[1] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[1]],
        'a': [truths[2], med[2], low[2], high[2], np.std(np.load(repo_name + '/samples.npy'), axis=0)[2] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[2]],
        'p0': [truths[3], med[3], low[3], high[3], np.std(np.load(repo_name + '/samples.npy'), axis=0)[3] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[3]],
        'e0': [truths[4], med[4], low[4], high[4], np.std(np.load(repo_name + '/samples.npy'), axis=0)[4] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[4]],
        'DL': [truths[5], med[5], low[5], high[5], np.std(np.load(repo_name + '/samples.npy'), axis=0)[5] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[5]],
        'costhetaS': [truths[6], med[6], low[6], high[6], np.std(np.load(repo_name + '/samples.npy'), axis=0)[6] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[6]],
        'phiS': [truths[7], med[7], low[7], high[7], np.std(np.load(repo_name + '/samples.npy'), axis=0)[7] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[7]],
        'costhetaK': [truths[8], med[8], low[8], high[8], np.std(np.load(repo_name + '/samples.npy'), axis=0)[8] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[8]],
        'phiK': [truths[9], med[9], low[9], high[9], np.std(np.load(repo_name + '/samples.npy'), axis=0)[9] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[9]],
        'Phivarphi0': [truths[10], med[10], low[10], high[10], np.std(np.load(repo_name + '/samples.npy'), axis=0)[10] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[10]],
        'Phir0': [truths[11], med[11], low[11], high[11], np.std(np.load(repo_name + '/samples.npy'), axis=0)[11] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[11]],
        'd': [truths[12], med[12], low[12], high[12], np.std(np.load(repo_name + '/samples.npy'), axis=0)[12] / np.mean(np.load(repo_name + '/samples.npy'), axis=0)[12]]
    }

    df = pd.DataFrame(data)
    
    # Save the DataFrame as a PDF table
    # Save the DataFrame as a LaTeX table
    pd.set_option('display.float_format', lambda x: '%.8e' % x)
    df.to_markdown('./posterior_summary/parameter_table'+repo_name.split('/')[-1]+'.md', floatfmt=".8e")

    # Parse parameters from repo_name
    params = repo_name.split('_')[3:]
    params_dict = {}
    
    for param in params:
        name_to_split = re.match(r'([a-zA-Z]+)', param).groups()[0]
        key, value = name_to_split, float(param.split(name_to_split)[1])
        params_dict[key] = value

    # labels
    label = '('

    # label += f"{params_dict.get('T')}"
    label += fr"{params_dict.get('M')/1e6}$\times 10^6$"
    if int(params_dict.get('mu'))==5:
        label += f", $\, \, \,${int(params_dict.get('mu'))}"
    else:
        label += f", {int(params_dict.get('mu'))}"
    label += f", {params_dict.get('a'):.2f}"
    label += f", {params_dict.get('e')}"
    if params_dict.get('charge') == 0.0:
        label += f", $0.0$"
    else:
        label += fr", {params_dict.get('charge')*1e3} $\times 10^{{-3}}$"
    
    label += ')'
    print(params_dict)
    # update dict with med and 95% CI
    params_dict.update({f"median":med})
    params_dict.update({f"perc2.5":low})
    params_dict.update({f"perc97.5":high})
    # append params_dict to list_dict
    list_dict.append(params_dict)
    
    labs.append(label)

########################### plot all #############################################
# create a txt and store the information in list_dict

overlaid_corner(list_chains, labs, './figures/plot_all_parameters_posteriors', corn_kw=CORNER_KWARGS, title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, d)$')
# indintr = np.asarray([0,1,2,3,4,10,11,12])
# c_kw = CORNER_KWARGS.copy()
# c_kw['labels'] = [labels[i] for i in indintr]
# overlaid_corner(  [el[:,indintr] for el in list_chains], labs, './plot_paper/intr_parameters_posteriors', corn_kw=c_kw, title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, d)$')
# c_kw = CORNER_KWARGS.copy()
# indext = np.asarray([5,6,7,8,9,])
# c_kw['labels'] = [labels[i] for i in indext]
# overlaid_corner(  [el[:,indext] for el in list_chains], labs, './plot_paper/extr_parameters_posteriors', corn_kw=c_kw, title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, d)$')