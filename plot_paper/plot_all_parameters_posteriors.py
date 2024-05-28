import glob
# from eryn.backends import HDFBackend
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
from few.utils.utility import get_fundamental_frequencies
from few.trajectory.inspiral import EMRIInspiral
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

traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
def get_Ncycles_Dephasing(logM,logmu,a,p0,e0, charge):
    M = np.exp(logM)
    mu = np.exp(logmu)
    x0 = 1.0
    Lambda = charge**2 /4.
    Phi_phi = traj(M, mu, a, p0, e0, x0, Lambda, T=2.0, dt=10.0)[4]
    Phi_phi_zero = traj(M, mu, a, p0, e0, x0, 0.0, T=2.0, dt=10.0)[4]
    return Phi_phi_zero[-1]/(2*np.pi), Phi_phi[-1]-Phi_phi_zero[-1]

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

labels = [r'$\Delta \ln M$', r'$\Delta \ln \mu$', r'$\Delta a$', r'$\Delta p_0 \, [M]$', r'$\Delta e_0$', 
            r'$\Delta D_L \, [{\rm Gpc}]$',
            r"$\Delta \cos \theta_S$",r"$\Delta \phi_S$",
            r"$\Delta \cos \theta_K$",r"$\Delta \phi_K$",
        r'$\Delta \Phi_{\varphi 0}$', r'$\Delta \Phi_{r 0}$',
            r"$\Delta \Lambda$",
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
    max_n_ticks=3,
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

def get_log10alpha_weights(lnmu,d):
    """
    Calculate the log10alpha weights based on the given parameters.

    Parameters:
    lnmu (float): The natural logarithm of the parameter mu.
    d (float): The value of the parameter d.

    Returns:
    log10alpha (float): The calculated log10alpha value.
    w_sqrtalpha (float): The calculated weight for log10alpha value.
    sqrtalpha (float): The calculated sqrtalpha value.
    w (float): The calculated weight for sqrtalpha value.
    """
    mu = np.exp(lnmu)
    w = mu / np.sqrt(d)
    sqrtalpha = np.sqrt(2*d)*mu*MRSUN_SI/1e3 # kilometers
    log10alpha = np.log10(sqrtalpha)
    return log10alpha, w/sqrtalpha, sqrtalpha, w


def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

########################### preparation of the data #############################################
init_name = '../DataAnalysis/paper_runs/MCMC*'
datasets = ['../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.0_SNR50.0_T0.5_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu5.0_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.4_e0.2_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T4.0_seed2601_nw26_nt1.h5'
]

pars_inj =['../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.0_SNR50.0_T0.5_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu5.0_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.4_e0.2_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T4.0_seed2601_nw26_nt1_injected_pars.npy'
]
print("len names", len(datasets),len(pars_inj))
cmap = plt.cm.get_cmap('Set1',)
colors = [cmap(i) for i in range(len(datasets))]
ls = ['-','--','-.',':',(0, (3, 1, 1, 1, 3))]

list_chains,labs = [], []
list_dict = []
list_cyc_precision = []

# Comparison
table_comparison = {'Run Name':['95 Bound Charge', '95 Percent Bound Sqrt Alpha', 'dimensionless velocity' ,'Ncycles vacuum']}
# Scalar plot
for filename, inj_params, color in zip(datasets, pars_inj, colors):
    # Get repository name
    repo_name = inj_params.split('_injected_pars.npy')[0]
    
    # Load injected parameters
    truths = np.load(inj_params)
    
    # Load MCMC samples
    temp_samp = np.load(repo_name + '/samples.npy')
    Lambda = temp_samp[:,-1]
    temp_samp = temp_samp[Lambda>0.0]
    list_chains.append(temp_samp - truths[None,:])
    # obtain median and 95% credible interval from the samples
    med = np.median(temp_samp, axis=0)
    low = np.percentile(temp_samp, 2.5, axis=0)
    high = np.percentile(temp_samp, 97.5, axis=0)
    # add information
    Lambda = temp_samp[:,-1]
    charge = np.sqrt(4 * Lambda)
    mu = np.exp(temp_samp[:,1])
    sqrt_alpha = 2*mu*MRSUN_SI/1e3*Lambda**(1/4)
    weights = mu * Lambda**(-3/4) * 0.5

    sqrtalpha_quantiles = weighted_quantile(sqrt_alpha, [0.025, 0.5, 0.975,  0.95,], sample_weight=weights)
    charge = np.sqrt(4 * Lambda)
    weights_ch=1/np.sqrt(Lambda)
    charge_quantiles = weighted_quantile(charge, [0.025, 0.5, 0.975,  0.95,], sample_weight=weights_ch)
    
    # Create a DataFrame with the parameter information
    # add correlation coefficient
    corrcoef = np.corrcoef(temp_samp.T)
    data = {
        'variable': ['ln M', 'ln mu', 'a', 'p0', 'e0', 'DL', 'costhetaS', 'phiS', 'costhetaK', 'phiK', 'Phivarphi0', 'Phir0', 'Lambda'],
        'median': [med[0], med[1], med[2], med[3], med[4], med[5], med[6], med[7], med[8], med[9], med[10], med[11], med[12]],
        'true': [truths[0], truths[1], truths[2], truths[3], truths[4], truths[5], truths[6], truths[7], truths[8], truths[9], truths[10], truths[11], truths[12]],
        'percentile 2.5 perc': [low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7], low[8], low[9], low[10], low[11], low[12]],
        'percentile 97.5 perc': [high[0], high[1], high[2], high[3], high[4], high[5], high[6], high[7], high[8], high[9], high[10], high[11], high[12]],
        'one sigma relative precision': [np.std(temp_samp, axis=0)[0] / np.mean(temp_samp, axis=0)[0], np.std(temp_samp, axis=0)[1] / np.mean(temp_samp, axis=0)[1], np.std(temp_samp, axis=0)[2] / np.mean(temp_samp, axis=0)[2], np.std(temp_samp, axis=0)[3] / np.mean(temp_samp, axis=0)[3], np.std(temp_samp, axis=0)[4] / np.mean(temp_samp, axis=0)[4], np.std(temp_samp, axis=0)[5] / np.mean(temp_samp, axis=0)[5], np.std(temp_samp, axis=0)[6] / np.mean(temp_samp, axis=0)[6], np.std(temp_samp, axis=0)[7] / np.mean(temp_samp, axis=0)[7], np.std(temp_samp, axis=0)[8] / np.mean(temp_samp, axis=0)[8], np.std(temp_samp, axis=0)[9] / np.mean(temp_samp, axis=0)[9], np.std(temp_samp, axis=0)[10] / np.mean(temp_samp, axis=0)[10], np.std(temp_samp, axis=0)[11] / np.mean(temp_samp, axis=0)[11], np.std(temp_samp, axis=0)[12] / np.mean(temp_samp, axis=0)[12]],
        'correlation coefficient with Lambda': [corrcoef[-1,0], corrcoef[-1,1], corrcoef[-1,2], corrcoef[-1,3], corrcoef[-1,4], corrcoef[-1,5], corrcoef[-1,6], corrcoef[-1,7], corrcoef[-1,8], corrcoef[-1,9], corrcoef[-1,10], corrcoef[-1,11], corrcoef[-1,12]],
    }
    
    true_charge = np.sqrt(4 * truths[12])
    true_sqrtalpha = np.sqrt(2*true_charge)*np.exp(truths[1])*MRSUN_SI/1e3
    # another table only for the alpha contraints
    data_alpha = {
        'estimator': ['true', 'median', 'percentile 2.5 perc', 'percentile 97.5 perc', 'percentile 95 perc'],
        'sqrtalpha': [true_sqrtalpha, sqrtalpha_quantiles[1], sqrtalpha_quantiles[0], sqrtalpha_quantiles[2], sqrtalpha_quantiles[3]],
        'charge': [true_charge, charge_quantiles[1], charge_quantiles[0], charge_quantiles[2], charge_quantiles[3]],
        'Delta Phi_phi': ['None', 'None', 'None', 'None', 'None'],
        'Ncycles vacuum': ['None', 'None', 'None', 'None', 'None'],
    }
    # dephasing information 
    Ncyc, Deph = get_Ncycles_Dephasing(truths[0],truths[1],truths[2],truths[3],truths[4],data_alpha["charge"][-1])

    data_alpha['Delta Phi_phi'][-1] = Deph
    data_alpha['Ncycles vacuum'][0] = Ncyc

    df = pd.DataFrame(data)
    # Save the DataFrame as a LaTeX table
    df.to_markdown('./posterior_summary/table_'+repo_name.split('/')[-1]+'.md', floatfmt=".10e")
    
    df_alpha = pd.DataFrame(data_alpha)
    df_alpha.to_markdown('./posterior_summary/alpha_table_'+repo_name.split('/')[-1]+'.md', floatfmt=".10e")

    # Parse parameters from repo_name
    params = repo_name.split('_')[2:]
    params_dict = {}
    
    for param in params:
        name_to_split = re.match(r'([a-zA-Z]+)', param).groups()[0]
        key, value = name_to_split, float(param.split(name_to_split)[1])
        params_dict[key] = value

    # labels
    label = '('

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
    
    label += f",{params_dict.get('T')}"
    # add another lable for the cycles
    label += f", {Ncyc:.2f}"
    label += ')'
    print(params_dict)
    # update dict with med and 95% CI
    params_dict.update({f"median":med})
    params_dict.update({f"perc2.5":low})
    params_dict.update({f"perc97.5":high})
    # append params_dict to list_dict
    list_dict.append(params_dict)
    
    labs.append(label)
    
    # append to list the precision on the intrinsic parameter and the number of cycles
    precision = np.std(temp_samp, axis=0)[:5] / np.mean(temp_samp, axis=0)[:5]
    precision[:2] *= np.mean(temp_samp, axis=0)[:2]
    list_cyc_precision.append(np.append(precision,Ncyc))
    # construct a table where the first axis is the
    v = get_fundamental_frequencies(truths[2], truths[3], truths[4], 1.0)[0]**(1/3)
    print(v**(-2))
    table_comparison[repo_name.split('_noise0.0_')[-1].split('_seed')[0]] = [charge_quantiles[3], sqrtalpha_quantiles[3],v,Ncyc]

table_comparison
pd.DataFrame(table_comparison).T.to_markdown('./posterior_summary/comparison_table_for_bounds.md', floatfmt=".10e")

########################### plot all #############################################
# plot corner plots
overlaid_corner(list_chains, labs, './figures/plot_all_parameters_posteriors', corn_kw=CORNER_KWARGS, title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, \Lambda, T [{\rm yrs}], N_{\rm cycles})$')

# pl
# # plot the precision as a function of the number of cycles
# list_cyc_precision = np.asarray(list_cyc_precision)

# cmap = plt.cm.get_cmap('Set1',)
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# plt.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": 'pdflatex',
#     "pgf.rcfonts": False,
#     "font.family": "serif",
#     "figure.figsize": [246.0*px, inv_golden * 246.0*px],
#     'legend.fontsize': 12,
#     'xtick.labelsize': 18,
#     'ytick.labelsize': 18,
#     'legend.title_fontsize' : 12,
# })

# # plt.figure()
# # var_list = [r"$M$", r"$\mu$", r"$a$", r"$p_0$", r"$e_0$"]
# # run_list = np.arange(5)
# # marker_list = ['o','s','^','D','P']
# # for var in range(5):
# #     for run_ind in run_list:
# #         plt.scatter(list_cyc_precision[run_ind,-1], list_cyc_precision[run_ind,:-1][var], label=var_list[var], color=colors[run_ind], marker=marker_list[var],alpha=0.7)
# # plt.yscale('log')
# # plt.xlabel(r'$N_{\rm cycles}$', fontsize=20)
# # plt.ylabel(r'$\sigma_\theta / \theta$', fontsize=20)
# # # define custom legend with markers only
# # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=r'$M$', markerfacecolor='black', markersize=10),
# #                    plt.Line2D([0], [0], marker='s', color='w', label=r'$\mu$', markerfacecolor='black', markersize=10),
# #                    plt.Line2D([0], [0], marker='^', color='w', label=r'$a$', markerfacecolor='black', markersize=10),
# #                    plt.Line2D([0], [0], marker='D', color='w', label=r'$p_0$', markerfacecolor='black', markersize=10),
# #                    plt.Line2D([0], [0], marker='P', color='w', label=r'$e_0$', markerfacecolor='black', markersize=10)]
# # # now add the labels of labs
# # plt.legend(handles=legend_elements, loc='upper right')
# # plt.tight_layout()
# # plt.grid()
# # plt.savefig('./figures/plot_precision_vs_Ncycles.pdf')