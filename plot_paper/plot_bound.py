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
        label += f", {int(params_dict.get('mu'))}"
    label += f", {params_dict.get('a'):.2f}"
    label += f", {params_dict.get('e')}"
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

cmap = plt.cm.get_cmap('Set1',)
colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


colors = sns.color_palette('colorblind')
########################################################################
from scipy.stats import gaussian_kde

init_name = '../DataAnalysis/paper_runs/*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))

print("len names", len(datasets),len(pars_inj))
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (1, 3))]

# Scalar plot
plt.figure()
for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
    label, toplot, truths = get_labels_chains(el)

    # # add to the following labels the log10 of the savage-dickey ratio
    # kde = gaussian_kde(toplot[:,-1], bw_method='scott')
    # med = np.median(toplot,axis=0)
    # if truths[-1]!=0.0:
    #     density_at_zero = kde.evaluate(toplot[:,-1].min())
    # else:
    #     density_at_zero = kde.evaluate(0.0)
    
    # prior = 1 / 0.2
    # savage_dickey = np.log10(prior / density_at_zero)[0]
    # print(f"log10 SD-Ratio: {savage_dickey}")
    Lambda = toplot[:,-1]
    Lambda = Lambda[Lambda>0.0]
    charge = np.sqrt(4 * Lambda)
    plt.hist(charge, weights=1/np.sqrt(Lambda), bins=40, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
    # plot kde of the last parameter
    # plt.plot(np.linspace(toplot[:,-1].min(),toplot[:,-1].max(),100),kde.evaluate(np.linspace(toplot[:,-1].min(),toplot[:,-1].max(),100)),color=cc,linestyle=ll,label=label, linewidth=3,)

plt.tight_layout()
plt.xlabel(r'$d$',size=22)
plt.ticklabel_format(style='sci')
plt.legend(title='\t \t'+r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$',loc='upper right')# bbox_to_anchor=(0.0, 0.5)
# plt.legend()
# plt.ylim(0.0,1970)
plt.savefig(f'./figures/bound_charge.pdf', bbox_inches='tight')

#####################################
labels = [r'$\Delta \ln (M/{\rm M}_\odot$)', r'$\Delta \ln (\mu / M_{\odot})$', r'$\Delta a$', r'$\Delta p_0 \, [M]$', r'$\Delta e_0$', 
            r'$\Delta D_L \, [{\rm Gpc}]$',
            r"$\Delta \cos \theta_S$",r"$\Delta \phi_S$",
            r"$\Delta \cos \theta_K$",r"$\Delta \phi_K$",
        r'$\Delta \Phi_{\varphi 0}$', r'$\Delta \Phi_{r 0}$',
            r"$\Lambda$",
        ]
for var in range(5):
    plt.figure()
    for filename,el in zip(datasets,pars_inj):
        label, toplot, truths = get_labels_chains(el)        
        # mask for positive Lambda
        Lambda = toplot[:,-1]
        mask = (Lambda>0.0)
        plt.hist(toplot[mask,var]-truths[var], bins=40, histtype='step', density=True, label=label, linewidth=3)

    plt.tight_layout()
    plt.xlabel(labels[var],size=22)
    plt.ticklabel_format(style='sci')
    plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
    # plt.legend()
    plt.savefig(f'./figures/bound_variable_{var}.pdf', bbox_inches='tight')
