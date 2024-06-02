import glob
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
import matplotlib as mpl
import re
import matplotlib.style as style
import seaborn as sns
# style.use('tableau-colorblind10')

default_width = 5.78853 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

import matplotlib.ticker as mticker

vals = [0.000001,0.00001,0.0001,0.01,1.0]

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

g = lambda x,pos : "${}$".format(f.set_scientific('%1.10e' % x))
fmt = mticker.FuncFormatter(g)

from scipy.constants import golden
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

def get_labels_chains(el):
    
    # get_repo name
    repo_name = el.split('_injected_pars.npy')[0]
    repo_name
    truths = np.load(el)
    toplot = np.load(repo_name + '/samples.npy')
    
    # Parse parameters from repo_name
    params = repo_name.split('_')[2:]
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
        mulab = params_dict.get('mu')
        if mulab == 10.:
            label += f", {int(mulab)}"
        else:
            label += f", {mulab}"
    label += f", {params_dict.get('a'):.2f}"
    label += f", {params_dict.get('e')}"
    
    label += f", {params_dict.get('T')}"
    label += ')'
    
    return label, toplot, truths

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

########################################################################
from scipy.stats import gaussian_kde

init_name = '../DataAnalysis/paper_runs/MCMC*'
datasets = [
# '../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+05_mu5.0_a0.95_p1.9e+01_e0.4_x1.0_charge0.0_SNR50.0_T1.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.0_SNR50.0_T0.5_seed2601_nw26_nt1.h5',
'../DataAnalysis/results/MCMC_noise0.0_M5e+05_mu3.6_a0.95_p9.2_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu5.0_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
# '../DataAnalysis/results/MCMC_noise0.0_M5e+04_mu3.6_a0.95_p2.1e+01_e0.4_x1.0_charge0.0_SNR11.0_T0.5_seed2601_nw26_nt1.h5',
# '../DataAnalysis/results/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.4_e0.2_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T4.0_seed2601_nw26_nt1.h5'
]

pars_inj =[
# '../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+05_mu5.0_a0.95_p1.9e+01_e0.4_x1.0_charge0.0_SNR50.0_T1.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+05_mu5.0_a0.95_p1.6e+01_e0.4_x1.0_charge0.0_SNR50.0_T0.5_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/results/MCMC_noise0.0_M5e+05_mu3.6_a0.95_p9.2_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu5.0_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
# '../DataAnalysis/results/MCMC_noise0.0_M5e+04_mu3.6_a0.95_p2.1e+01_e0.4_x1.0_charge0.0_SNR11.0_T0.5_seed2601_nw26_nt1_injected_pars.npy',
# '../DataAnalysis/results/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.4_e0.2_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',
'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p1e+01_e0.4_x1.0_charge0.0_SNR50.0_T4.0_seed2601_nw26_nt1_injected_pars.npy'
]

print("len names", len(datasets),len(pars_inj))
ls = ['-'  for i in range(len(datasets))]

colors = sns.color_palette('colorblind')
# cmap = plt.cm.get_cmap('tableau-colorblind10',)
# colors = ['#377eb8', '#ff7f00', '#4daf4a',
#                   '#f781bf', '#a65628', '#984ea3',
#                   '#999999', '#e41a1c', '#dede00']
# colors = [cmap(i) for i in range(len(datasets))]

fig, axs = plt.subplots(1, 2, figsize=(default_width*2, default_width * default_ratio))
plt.subplots_adjust(wspace=0.05)

ii = 0
for filename,el in zip(datasets, pars_inj):
    label, toplot, truths = get_labels_chains(el)

    Lambda = toplot[:,-1]
    mask = (Lambda>0.0)
    toplot = toplot[mask]
    Lambda = Lambda[mask]
    mu = np.exp(toplot[:,1])
    
    quantiles = [0.68, 0.95]
    
    # charge
    charge = np.sqrt(4 * Lambda)
    w_charge = 1 / np.sqrt(Lambda)
    ci_charge = weighted_quantile(charge, quantiles, sample_weight=w_charge)
    
    hist,bin_edges = np.histogram(charge, bins=np.linspace(0.0,0.04,num=30), weights=w_charge, density=True)
    hist = hist/hist.max()/2
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[0].bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], bottom=ii, color=colors[ii], alpha=0.8,label=label)
    axs[0].vlines(ci_charge[1], ii, ii+hist.max(), color=colors[ii], linestyle='--')
    axs[0].set_xlabel(r'Scalar charge $d$', size=22)

    # sqrt alpha 
    sqrt_alpha = 2*mu*MRSUN_SI/1e3*Lambda**(1/4)
    w_sqrta = mu * Lambda**(-3/4) * 0.5
    ci_sqrta = weighted_quantile(sqrt_alpha, quantiles, sample_weight=w_sqrta)
    
    hist,bin_edges = np.histogram(sqrt_alpha, bins=np.linspace(0.1,4.1,num=40), weights=w_sqrta, density=True)
    hist = hist/hist.max()/2
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[1].bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], bottom=ii, color=colors[ii], alpha=0.8)
    axs[1].vlines(ci_sqrta[1], ii, ii+hist.max(), color=colors[ii], linestyle='--')
    axs[1].set_xlabel(r'ESGB constant $\sqrt{\alpha}  [{\rm km}]$',size=22)
    # remove yticks
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    ii+=1

# add y ticks, centered at given values
axs[0].set_yticks(np.arange(0,ii,1)+0.2, np.arange(0,ii,1))

axs[1].axvline(0.26 * np.sqrt(16*np.pi**0.5), color='k',linestyle=':',label='GW230529')
axs[0].axvline(-0.01, color='k',linestyle=':',label='GW230529')
axs[0].set_xlim(0.0,0.035)
axs[1].set_xlim(0.0,4.1)
axs[0].legend(bbox_to_anchor=(0.05, 1.45), ncols=3, loc='upper left',
              title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0,T [{\rm yrs}])$',
              )
plt.savefig('figures/bound_charge_alpha.pdf', bbox_inches='tight')
