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
import mpmath
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

    
    label += fr"{params_dict.get('M')/1e6}$\times 10^6$"
    if int(params_dict.get('mu'))==5:
        label += f", $\, \, \,${int(params_dict.get('mu'))}"
    else:
        label += f", {int(params_dict.get('mu'))}"
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

init_name = '../DataAnalysis/paper_runs/MCMC*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
    
print("len names", len(datasets),len(pars_inj))
cmap = plt.cm.get_cmap('Set1',)
colors = sns.color_palette('colorblind')
# colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
colors = [cmap(i) for i in range(len(datasets))]
# provide custom line styles
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (3, 5, 1, 5, 1)),'-']
ls = ['-'  for i in range(len(datasets))]
########################################################################
# Alpha plot
plt.figure()
for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
    label, toplot, truths = get_labels_chains(el)
    # # obtain the eigenvectors of the covariance matrix from the samples toplot
    # mean_samp = np.mean(toplot,axis=0)
    # zero_mean_samp = toplot - mean_samp
    # cov = np.cov(zero_mean_samp,rowvar=False)
    # eigenvalues, eigenvectors = mpmath.eig(mpmath.matrix(cov))

    # eigenvalues = np.array([float(e) for e in eigenvalues])
    # eigenvectors = np.asarray(eigenvectors,dtype=float).reshape(cov.shape)
    # print((cov @ eigenvectors[:,-1])/(eigenvalues[-1]*eigenvectors[:,-1]))
    
    # # Assuming `plane` is your plane and `toplot` are your samples
    # normal_vector = np.eye(toplot.shape[1])[-1] # eigenvectors[:, -1] #  U[-1] # 
    # sign_sides = np.sum(zero_mean_samp * normal_vector[None,:],axis=1)
    # positive = (sign_sides>0.0)
    # negative = (sign_sides<0.0)
    # # Compute the reflection matrix
    # reflection_matrix = np.eye(zero_mean_samp.shape[1]) - 2 * np.outer(normal_vector, normal_vector)
    # # Apply the reflection matrix to the samples
    
    # flipped_samp = np.dot(zero_mean_samp[negative], reflection_matrix.T).copy()
    # flipped_samp[:,-1] *= -1
    # zero_mean_samp[negative] = -flipped_samp.copy()
    # now we need to flip around the zer
    # produce corner of samples
    # figure = corner.corner(zero_mean_samp[-10000:]); figure.savefig('corner.png')
    # toplot = zero_mean_samp + mean_samp

    Lambda = toplot[:,-1]
    mask = (Lambda>0.0)
    toplot = toplot[mask]
    Lambda = Lambda[mask]
    mu = np.exp(toplot[:,1])
    # d = np.sqrt(4*toplot[:,-1])#np.abs(toplot[:,-1] - np.median(toplot[:,-1]))
    # mu = mu[d!=0.0]
    # d = d[d!=0.0]
    sqrt_alpha = 2*mu*MRSUN_SI/1e3*Lambda**(1/4)
    weights = mu * Lambda**(-3/4) * 0.5
    xlow = 0.0
    bins = np.linspace(xlow,5.0,num=30)#+ np.random.uniform(-0.05,-0.0001)
    # 
    plt.hist(sqrt_alpha, weights=weights, bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
    # create a function for the quantile of alpha and put in in summary
    # upp95 = weighted_quantile(np.log10(y),[0.95],sample_weight=w/y)
    # plt.axvline(np.quantile(np.log10(y),0.975),color=cc)

plt.tight_layout()
plt.xlabel(r'$\sqrt{\alpha}  [{\rm km}]$',size=22)
plt.ticklabel_format(style='sci')

# from Maselli, to be updated with Elise's paper

# vpos = 10**0.8
# # Create a bar
# ylev = 0.8
# plt.broken_barh([(xlow, vpos-xlow)], (ylev,0.1), edgecolor='black')#, facecolors='none')
# plt.text(vpos+0.1, ylev+0.05, 'LVK current\nconstraint', ha='left', va='center', fontsize=12)

# from figure 21 https://arxiv.org/pdf/2010.09010
Nsource_3g = 1e6
vpos = 0.37 * np.sqrt(16*np.pi**0.5)
ylev = 1.0
plt.broken_barh([(xlow, vpos-xlow)], (ylev,0.1), edgecolor='black')#, facecolors='none')
plt.text(vpos+0.1, ylev+0.05, r'LVK constraint', ha='left', va='center', fontsize=12)

# vpos = 5e-2 * np.sqrt(16*np.pi**0.5)
# ylev = 1.2
# plt.broken_barh([(xlow, vpos-xlow)], (ylev,0.1), edgecolor='black')#, facecolors='none')
# plt.text(vpos+0.1, ylev+0.05, 'Projected cumulative constraint\n LIGO Voyager 08/09', ha='left', va='center', fontsize=12)

plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0,T [{\rm yrs}])$',loc='upper right')#,ncol=2)

# plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# plt.legend()
plt.xlim(xlow,7.6)
plt.ylim(0.0,1.9)
plt.savefig(f'./figures/bound_alpha.pdf', bbox_inches='tight')
