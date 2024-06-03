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

labels = [r'$ \ln M$', r'$ \ln \mu$', r'$ a$', r'$ p_0 \, [M]$', r'$ e_0$', 
            r'$ D_L \, [{\rm Gpc}]$',
            r"$ \cos \theta_S$",r"$ \phi_S$",
            r"$ \cos \theta_K$",r"$ \phi_K$",
        r'$ \Phi_{\phi 0}$', r'$ \Phi_{r 0}$',
            # r"$\Delta \Lambda$",
            # r"$\Delta d$",
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

def overlaid_corner(samples_list, sample_labels, name_save=None, corn_kw=None, title=None, ylim=None, weights=None,):
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
    
    if weights is None:
        weights = [get_normalisation_weight(len(samples_list[idx]), max_len) for idx in range(0, n)]
    else:
        weights = [get_normalisation_weight(len(samples_list[idx]), max_len)*weights[idx] for idx in range(0, n)]
    # Create the first corner plot
    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        weights=weights[0],
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
            weights=weights[idx],
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
datasets = ['../DataAnalysis/paper_runs/devGR_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.025_SNR50.0_T0.5_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/bias_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.025_SNR50.0_T0.5_seed2601_nw26_nt1_vacuum.h5',
]

pars_inj =['../DataAnalysis/paper_runs/devGR_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.025_SNR50.0_T0.5_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/bias_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.025_SNR50.0_T0.5_seed2601_nw26_nt1_vacuum_injected_pars.npy',
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
    # list_chains.append(temp_samp - truths[None,:])
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
    # append charge at the end of the samples
    # temp_samp = np.hstack((temp_samp,sqrt_alpha[:,None]))
    # temp_samp = np.hstack((temp_samp,weights[:,None]))
    if filename == datasets[0]:
        list_chains.append(temp_samp[:,:-1])
        # histogram of the charge
        fig, ax = plt.subplots(1, 1, figsize=(default_width, default_width * default_ratio))
        ax.hist(charge, weights=weights_ch, bins=40, histtype='step', density=True, lw=2)#, label='Scalar charge')
        ax.axvline(charge_quantiles[0], ls='--', lw=2)
        ax.axvline(0.025, ls=':', lw=2, label='Injected scalar charge')
        ax.axvline(charge_quantiles[2], ls='--', lw=2, label='95\% credible interval')
        
        ax.axvline(0.035, ls='-', color='orange', lw=2, label='Upper bound GW230529')
        print(charge_quantiles)
        plt.xlabel(r'Scalar charge $d$',fontsize=18)
        plt.ylabel('Marginalized posterior',fontsize=18)
        plt.legend()
        plt.savefig('./figures/charge_histogram.pdf', bbox_inches='tight')
    else:
        list_chains.append(temp_samp[:,:])
    
# table_comparison
# pd.DataFrame(table_comparison).T.to_markdown('./posterior_summary/comparison_table_for_bounds.md', floatfmt=".10e")

########################### plot all #############################################
# plot corner plots
# breakpoint()
CORNER_KWARGS['truths'] = truths
overlaid_corner(list_chains, ['Scalar charge template','Vacuum template'], './figures/plot_bias', corn_kw=CORNER_KWARGS, title='Scalar charge injected $d=0.025$')
