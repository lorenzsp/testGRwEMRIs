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


init_name = 'results_paper/mcmc_rndStart_M*_charge0.0_*seed2601*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
# sort datasets by charge
datasets = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', ]
# create the list of injected parameters
pars_inj = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy',  ]

print("len names", len(datasets),len(pars_inj))
cmap = plt.cm.get_cmap('Set1',)
colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


colors = sns.color_palette('colorblind')
# colors = [cmap(i) for i in range(len(datasets))]#['black','red', 'royalblue']#
ls = ['-','--','-.',':',(0, (2, 2)),(0, (1, 2))]
# provide custom line styles
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (3, 5, 1, 5, 1))]
########################################################################
# Alpha plot
plt.figure()
for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
    # get_repo name
    repo_name = el.split('_injected_pars.npy')[0]
    repo_name
    truths = np.load(el)
    toplot = np.load(repo_name + '/samples.npy')
    
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
    label += ')'

    
    # alpha bound
    mu = np.exp(toplot[:,1] - np.median(toplot[:,1]) + truths[1])
    d = np.abs(toplot[:,-1] - np.median(toplot[:,-1]))
    mu = mu[d!=0.0]
    d = d[d!=0.0]
    w = mu / np.sqrt(d)
    y = np.sqrt(2*d)*mu*MRSUN_SI/1e3
    bins = np.linspace(-1.5,0.3,num=25)+ np.random.uniform(-0.05,-0.0001)
    
    plt.hist(np.log10(y), weights=w/y, bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
    # plt.axvline(np.quantile(np.log10(y),0.975),color=cc)

plt.tight_layout()
plt.xlabel(r'$\log_{10} [\sqrt{\alpha} / {\rm km} ]$',size=22)
vpos = 0.8
plt.ticklabel_format(style='sci')
# 
# plt.axvline(vpos,color='k',linestyle=':',label='Current bound', linewidth=2)
plt.annotate('Current bound \nfrom LVK', xy=(vpos, 0.0), xytext=(vpos, 0.35),
             arrowprops=dict(facecolor='black', shrink=0.001),
             fontsize=12, ha='center')

vpos = np.log10(0.4 * np.sqrt( 16 * np.pi**0.5 ))
# plt.axvline(vpos,color='r',linestyle='-.',label='Best bound from 3G', linewidth=2)

# Add arrow with text
plt.annotate('Best bound \nfrom 3G', xy=(vpos, 0.0), xytext=(vpos, 0.35),
             arrowprops=dict(facecolor='black', shrink=0.001),
             fontsize=12, ha='center')

plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# plt.legend()
plt.xlim(-1.5,1.1)
plt.ylim(0.0,1.3)
plt.savefig(f'./plot_paper/alpha_bound.pdf', bbox_inches='tight')

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
# cmap = plt.cm.get_cmap('Set1',)
# colors = [cmap(i) for i in range(len(datasets))]#['black','red', 'royalblue']#
ls = ['-','--','-.',':',(0, (3, 1, 1, 1, 3))]
# can you find a better way to plot the lines?
#
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (1, 3))]

# Scalar plot
plt.figure()
for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
    # get_repo name
    repo_name = el.split('_injected_pars.npy')[0]
    repo_name
    truths = np.load(el)
    toplot = np.load(repo_name + '/samples.npy')


    # label += f"{params_dict.get('T')}"
    
    # add to the following labels the log10 of the savage-dickey ratio
    kde = gaussian_kde(toplot[:,-1], bw_method='scott')
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

    plt.hist(toplot[:,-1], bins=40, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
    # plot kde of the last parameter
    # plt.plot(np.linspace(toplot[:,-1].min(),toplot[:,-1].max(),100),kde.evaluate(np.linspace(toplot[:,-1].min(),toplot[:,-1].max(),100)),color=cc,linestyle=ll,label=label, linewidth=3,)

plt.tight_layout()
plt.xlabel(r'$d$',size=22)
vpos = 0.8
plt.ticklabel_format(style='sci')
plt.legend(title='\t \t'+r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, \log_{10} {\rm BF})$',loc='upper right')# bbox_to_anchor=(0.0, 0.5)
# plt.legend()
plt.ylim(0.0,1970)
plt.savefig(f'./plot_paper/charge_bound_hist.pdf', bbox_inches='tight')

#####################################
labels = [r'$\Delta \ln (M/{\rm M}_\odot$)', r'$\Delta \ln (\mu / M_{\odot})$', r'$\Delta a$', r'$\Delta p_0 \, [M]$', r'$\Delta e_0$', 
            r'$\Delta D_L \, [{\rm Gpc}]$',
            r"$\Delta \cos \theta_S$",r"$\Delta \phi_S$",
            r"$\Delta \cos \theta_K$",r"$\Delta \phi_K$",
        r'$\Delta \Phi_{\varphi 0}$', r'$\Delta \Phi_{r 0}$',
            r"$d$",
        ]
for var in range(5):
    plt.figure()
    for filename,el in zip(datasets,pars_inj):
            # get_repo name
        repo_name = el.split('_injected_pars.npy')[0]
        repo_name
        truths = np.load(el)
        toplot = np.load(repo_name + '/samples.npy')
        
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
        label += ')'
        
        plt.hist(toplot[:,var]-truths[var], bins=40, histtype='step', density=True, label=label, linewidth=3)

    plt.tight_layout()
    plt.xlabel(labels[var],size=22)
    plt.ticklabel_format(style='sci')
    plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
    # plt.legend()
    plt.savefig(f'./plot_paper/variable_{var}.pdf', bbox_inches='tight')

#####################################


init_name = 'results_paper/mcmc_rndStart_M*_charge0.0_*seed2601*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
print("len names", len(datasets),len(pars_inj))
