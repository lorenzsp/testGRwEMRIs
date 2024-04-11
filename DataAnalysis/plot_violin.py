import glob
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
import matplotlib as mpl
import re
# import matplotlib.style as style
# style.use('tableau-colorblind10')
# get palette from tableau-colorblind10
# palette = plt.get_cmap('tab10')
# get the default color cycle
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

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

########################################################################
from scipy.stats import gaussian_kde

init_name = 'results_paper/mcmc_rndStart_M*_charge0.0*seed2601*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
# sort datasets by charge
datasets = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1.h5', ]
# create the list of injected parameters
pars_inj = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', ]

print("len names", len(datasets),len(pars_inj))
labels = []
samples = []
kde_list = []
for filename,el in zip(datasets,pars_inj):
    # get_repo name
    repo_name = el.split('_injected_pars.npy')[0]
    repo_name
    truths = np.load(el)
    toplot = np.load(repo_name + '/samples.npy')

    # add to the following labels the log10 of the savage-dickey ratio
    kde = gaussian_kde(toplot[:,-1], bw_method='scott')
    kde_list.append(kde)
    
    med = np.median(toplot,axis=0)
    if truths[-1]!=0.0:
        density_at_zero = kde.evaluate(toplot[:,-1].min())
    else:
        density_at_zero = kde.evaluate(0.0)
    
    prior = 1 / 0.2
    savage_dickey = np.log10(prior / density_at_zero)[0]
    print(f"log10 SD-Ratio: {savage_dickey}")

    # Parse parameters from repo_name
    params = repo_name.split('_')[3:]
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
    if truths[-1]!=0.0:
        label += fr", $>${savage_dickey:.2f}"
    else:
        label += f", {savage_dickey:.2f}"
    
    label += ')'
    labels.append(label)
    samples.append(toplot[:int(3e5),-1])


# # get the minimum of all the samples
# min_ = np.min([np.min(sample) for sample in samples[:-1]])
# # get the maximum of all the samples
# max_ = np.max([np.max(sample) for sample in samples[:-1]])
# # create an array of values from min to max
# x = np.linspace(min_,max_,100)
# # evaluate the kde at each value of x
# y = np.array([kde.evaluate(x) for kde in kde_list[:-1]])
# # multiply the different kde evaluations
# y = np.prod(y,axis=0)
# # normalize the kde
# y = y/np.sum(y*(x[1]-x[0]))
# plt.figure(figsize=(10, 6))
# # plot the combined kde
# plt.plot(x,y)
# # save the figure
# plt.savefig(f'./plot_paper/kde_charge_combined.pdf')

import seaborn as sns
import pandas as pd


# Assuming samples is a list of arrays, where each array is a run
data = pd.DataFrame({i: run for i, run in zip(labels, samples)})

# Create the violin plott
plt.figure(figsize=(10, 6))

# can you annotate on top of the violin plot the log10 of the savage-dickey ratio?
# plot the violin plot with a specific list of colors that I specify ith colors


sns.violinplot(data=samples, orient='h', bw_method='scott', palette=palette)# 'colorblind'
plt.xlabel(r"Scalar charge $d$",size=20)
# # Set the y-axis labels to the system parameters
plt.yticks(range(len(labels)), labels)

# shwo the y label horizontally)
plt.ylabel(r'$(M, \mu, a, e_0, \log_{10} {\rm BF})$', rotation=0,size=20)
plt.gca().yaxis.set_label_coords(-0.25,0.98)
plt.tight_layout()
plt.savefig(f'./plot_paper/violin_charge.pdf')
