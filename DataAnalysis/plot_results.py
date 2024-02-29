import glob
from eryn.backends import HDFBackend
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
import matplotlib as mpl
import re
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


init_name = 'results_paper/mcmc_*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
print("len names", len(datasets),len(pars_inj))

temp=0

samp_final = []
inj_pars = []


plt.figure()
for filename,el in zip(datasets,pars_inj):
    print('-------------------------------------')
    # get_repo name
    repo_name = el.split('_injected_pars.npy')[0]
    repo_name
    truths = np.load(el)
    toplot = np.load(repo_name + '/samples.npy')
    
    # Parse parameters from repo_name
    params = repo_name.split('_')[4:]
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
    mu = np.exp(toplot[:,1])
    d = np.abs(toplot[:,-1])
    w = mu / np.sqrt(d)
    y = np.sqrt(2*d)*mu*MRSUN_SI/1e3
    
    plt.hist(np.log10(y), weights=w/y, bins=np.linspace(-1.0,0.3,num=30)+ np.random.uniform(-0.05,0.0), histtype='step', density=True, label=label, linewidth=2)

plt.tight_layout()
plt.xlabel(r'$\log_{10} [\sqrt{\alpha} / {\rm km} ]$',size=22)
vpos = 0.8
plt.ticklabel_format(style='sci')
# 
plt.axvline(vpos,color='k',linestyle=':',label='Current bound', linewidth=2)
vpos = np.log10(0.4 * np.sqrt( 16 * np.pi**0.5 ))
plt.axvline(vpos,color='r',linestyle='-.',label='Best bound from 3G', linewidth=2)
plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# plt.legend()
plt.xlim(-1.0,1.0)
plt.savefig(f'./alpha_bound.png', bbox_inches='tight')

