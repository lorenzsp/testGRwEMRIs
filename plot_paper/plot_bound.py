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


for ii in range(len(datasets)):
    datasets[ii] = '../DataAnalysis/' +  datasets[ii]
    pars_inj[ii] = '../DataAnalysis/' +  pars_inj[ii]
    
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
plt.ticklabel_format(style='sci')

# vpos = 0.8
# plt.annotate('Current bound \nfrom LVK', xy=(vpos, 0.0), xytext=(vpos, 0.35),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center')

# vpos = np.log10(0.4 * np.sqrt( 16 * np.pi**0.5 ))
# plt.annotate('Forecasted bound \nfrom 3G', xy=(vpos, 0.0), xytext=(vpos, 0.35),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center')

# from Maselli, to be updated with Elise's paper
vpos = 0.8
xlow = -1.5
# Create a bar
ylev = 1.3
plt.broken_barh([(xlow, vpos-xlow)], (ylev,0.1), edgecolor='black')#, facecolors='none')
plt.text(vpos+0.1, ylev+0.05, 'LVK current\nconstraint', ha='left', va='center', fontsize=12)

# from figure 21 https://arxiv.org/pdf/2010.09010
Nsource_3g = 1e6
vpos = np.log10(0.4 * np.sqrt(16*np.pi**0.5))
ylev = 1.55
plt.broken_barh([(xlow, vpos-xlow)], (ylev,0.1), edgecolor='black')#, facecolors='none')
plt.text(vpos+0.1, ylev+0.05, 'Projected best event\nLIGO Voyager 08/09', ha='left', va='center', fontsize=12)

vpos = np.log10(5e-2 * np.sqrt(16*np.pi**0.5))
ylev = 1.8
plt.broken_barh([(xlow, vpos-xlow)], (ylev,0.1), edgecolor='black')#, facecolors='none')
plt.text(vpos+0.1, ylev+0.05, 'Projected cumulative constraint\n LIGO Voyager 08/09', ha='left', va='center', fontsize=12)

plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$',loc='lower right')

# plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# plt.legend()
plt.xlim(xlow,1.4)
plt.ylim(0.0,2.0)
plt.savefig(f'./figures/bound_alpha.pdf', bbox_inches='tight')
########################################################################
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_kerr_geo_constants_of_motion
# get Edot function
# trajELQ = EMRIInspiral(func="KerrEccentricEquatorial")
trajELQ = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
# from few.summation.interpolatedmodesum import CubicSplineInterpolant
#import cubic spline from scipy
from scipy.interpolate import CubicSpline

def get_dphi(M, mu, a, p0, e0, x0, charge):
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = trajELQ(M, mu, a, p0, e0, x0, charge, T=2.0, dt=10.0)
    t_0, p, e, x, Phi_phi_0, Phi_theta, Phi_r = trajELQ(M, mu, a, p0, e0, x0, 0.0, T=2.0, dt=10.0)
    interp = CubicSpline(t_0, Phi_phi_0)
    return np.max((1-Phi_phi/interp(t))[1:-1])

def get_Edot(M, mu, a, p0, e0, x0, charge):
    trajELQ.inspiral_generator.moves_check = 0
    kwargs = {
                "T":1.0,
            "dt":10.0,
            "err":1e-10,
            "DENSE_STEPPING":0,
            "max_init_len":int(1e4),
            "use_rk4":True,
            }
    trajELQ.inspiral_generator.initialize_integrator(**kwargs)

    # Compute the adimensionalized time steps and max time
    trajELQ.inspiral_generator.tmax_dimensionless = YRSID_SI / (M * MTSUN_SI)
    trajELQ.inspiral_generator.dt_dimensionless = 10.0 / (M * MTSUN_SI)
    trajELQ.inspiral_generator.Msec = MTSUN_SI * M
    trajELQ.inspiral_generator.a = a
    # define fixed variables
    trajELQ.inspiral_generator.integrator.add_parameters_to_holder(M, mu, a, np.asarray([charge]))
    y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)
    # y0 = np.array([y1, y2, y3])
    y0 = np.array([p0, e0, x0, 0.0, 0.0, 0.0])
    return trajELQ.inspiral_generator.integrator.get_derivatives(y0)[0] / (mu/M)

def get_delta_Edot(M, mu, a, p0, e0, x0, charge):
    # for defining B https://arxiv.org/pdf/1603.04075 and delta https://arxiv.org/pdf/1007.1995
    # https://arxiv.org/pdf/1007.1995
    # paper for pulsar constraints in ppE: https://arxiv.org/pdf/2002.02030
    
    Edot_Charge = get_Edot(M, mu, a, p0, e0, x0, charge)
    Edot_ZeroCharge = get_Edot(M, mu, a, p0, e0, x0, 0.0)
    delta_Edot = Edot_Charge - Edot_ZeroCharge
    B = (delta_Edot/Edot_ZeroCharge) / p0
    delta = delta_Edot/Edot_ZeroCharge
    return delta, B


Nsamp = int(1e5)

#------------------------- delta plot ---------------------------------
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
    newvar = toplot[:,:7].copy() - np.median(toplot[:,:7],axis=0) + truths[:7]
    newvar[:,0],newvar[:,1] = np.exp(newvar[:,0]), np.exp(newvar[:,1])
    newvar[:,5] = np.ones_like(newvar[:,6])
    newvar[:,6] = np.abs(toplot[:,-1] - np.median(toplot[:,-1])+ truths[-1])
    
    y = np.abs(np.asarray([get_delta_Edot(*newvar[ii])[0] for ii in range(Nsamp)]))
    bins = np.linspace(-11.5,-6.0,num=30) #+ np.random.uniform(-0.05,-0.0001)
    mask = (y!=0.0)
    plt.hist(np.log10(y[mask]),weights=1/newvar[:Nsamp,6][mask], bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
    # 
    # plt.axvline(np.quantile(np.log10(y),0.975),color=cc)

plt.tight_layout()
plt.xlabel(r'$\log_{10} \delta $',size=22)# {\dot{E}}
vpos = np.log10(6e-4) # binary J1141-6545 from https://arxiv.org/pdf/1603.04075
# approximately 60 000 orbits in  https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
vpos = np.log10(1-0.999963) # binary PSR J0737–3039 from https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
plt.ticklabel_format(style='sci')
plt.annotate('Bound from binary \npulsar J1141-6545', xy=(vpos, 0.0), xytext=(vpos, 0.1),
             arrowprops=dict(facecolor='black', shrink=0.001),
             fontsize=12, ha='center')

plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# plt.legend()
plt.xlim(-11.0,-2.0)
# plt.ylim(0.0,1.3)
plt.savefig(f'./figures/bound_delta.pdf', bbox_inches='tight')

# estimate period of the binary using the fundamental frequencies
# from few.utils.utility import get_fundamental_frequencies
# MTSUN_SI
# M,mu,a,p0,e0,x0=np.median(newvar,axis=0)[:6]
# frequency = get_fundamental_frequencies(a,p0,e0,x0)[0] / (2*np.pi*M*MTSUN_SI)
# period = 1/frequency
# breakpoint()
#------------------------- delta Edot ---------------------------------
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
    newvar = toplot[:,:7].copy() - np.median(toplot[:,:7],axis=0) + truths[:7]
    newvar[:,0],newvar[:,1] = np.exp(newvar[:,0]), np.exp(newvar[:,1])
    newvar[:,5] = np.ones_like(newvar[:,6])
    newvar[:,6] = np.abs(toplot[:,-1] - np.median(toplot[:,-1])+ truths[-1])
    
    y = np.abs(np.asarray([get_delta_Edot(*newvar[ii])[1] for ii in range(Nsamp)]))
    bins = np.linspace(-11.5,-6.0,num=30) #+ np.random.uniform(-0.05,-0.0001)
    mask = (y!=0.0)
    plt.hist(np.log10(y[mask]),weights=1/newvar[:Nsamp,6][mask], bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
    # 
    # plt.axvline(np.quantile(np.log10(y),0.975),color=cc)

plt.tight_layout()
plt.xlabel(r'$\log_{10} \delta \dot{E} $',size=22)# {\dot{E}}
# from fig 11 https://arxiv.org/pdf/2010.09010
vpos = np.log10(1e-8)
# plt.annotate('Projected\ncumulative\nconstraint\n3G', xy=(vpos, -0.25), xytext=(vpos, -0.25),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center',
#              xycoords='data', annotation_clip=False
#              )

xlow = -11.0
# Create a bar
plt.broken_barh([(xlow, vpos-xlow)], (0.65,0.1), edgecolor='black')#, facecolors='none')
plt.text(vpos+0.1, 0.7, 'SOBHs observed by \nfuture ground-based detectors', ha='left', va='center', fontsize=12)
# add the broken bar for the next valur of vpos

vpos = np.log10(1e-10)
plt.broken_barh([(xlow, vpos-xlow)], (0.8,0.1), edgecolor='black')#, facecolors='none')
plt.text(vpos+0.1, 0.85, 'SOBHs observed by \nLISA + future ground-based detectors (multiband)', ha='left', va='center', fontsize=12)


plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$',loc='lower right')
# plt.legend()
plt.xlim(-11.0,-4.0)
plt.ylim(0.0,0.95)
plt.savefig(f'./figures/bound_deltaEdot.pdf', bbox_inches='tight')
# breakpoint()
#----------------------------- delta phi ------------------------------------
# plt.figure()
# for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
#     # get_repo name
#     repo_name = el.split('_injected_pars.npy')[0]
#     repo_name
#     truths = np.load(el)
#     toplot = np.load(repo_name + '/samples.npy')
    
#     # Parse parameters from repo_name
#     params = repo_name.split('_')[3:]
#     params_dict = {}
    
#     for param in params:
#         name_to_split = re.match(r'([a-zA-Z]+)', param).groups()[0]
#         key, value = name_to_split, float(param.split(name_to_split)[1])
#         params_dict[key] = value

#     # labels
#     label = '('

#     # label += f"{params_dict.get('T')}"
#     label += fr"{params_dict.get('M')/1e6}$\times 10^6$"
#     if int(params_dict.get('mu'))==5:
#         label += f", $\, \, \,${int(params_dict.get('mu'))}"
#     else:
#         label += f", {int(params_dict.get('mu'))}"
#     label += f", {params_dict.get('a'):.2f}"
#     label += f", {params_dict.get('e')}"
#     label += ')'

    
#     # alpha bound
#     newvar = toplot[:,:7].copy() - np.median(toplot[:,:7],axis=0) + truths[:7]
#     newvar[:,0],newvar[:,1] = np.exp(newvar[:,0]), np.exp(newvar[:,1])
#     newvar[:,5] = np.ones_like(newvar[:,6])
#     newvar[:,6] = np.abs(toplot[:,-1] - np.median(toplot[:,-1])+ truths[-1])
    
#     y = np.abs(np.asarray([get_dphi(*newvar[ii]) for ii in range(Nsamp)]))
#     bins = np.linspace(-11.5,-6.0,num=30) #+ np.random.uniform(-0.05,-0.0001)
#     mask = (y!=0.0)
#     plt.hist(np.log10(y[mask]),weights=1/newvar[:Nsamp,6][mask], bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
#     # 
#     # plt.axvline(np.quantile(np.log10(y),0.975),color=cc)

# plt.tight_layout()
# plt.xlabel(r'$\log_{10} \delta \phi $',size=22)# {\dot{E}}
# vpos = np.log10(1e-7) # bound from https://arxiv.org/pdf/2002.02030 table IV
# plt.ticklabel_format(style='sci')
# # # 
# plt.annotate('Bound from LVK and binary pulsars', xy=(vpos, 0.0), xytext=(vpos, 0.1),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center')

# plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# # plt.legend()
# plt.xlim(-11.0,-6.0)
# # plt.ylim(0.0,1.3)
# plt.savefig(f'./figures/bound_deltaphi.pdf', bbox_inches='tight')

########################################################################
from scipy.stats import gaussian_kde

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
plt.savefig(f'./figures/bound_charge.pdf', bbox_inches='tight')

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
    plt.savefig(f'./figures/bound_variable_{var}.pdf', bbox_inches='tight')

#####################################


init_name = 'results_paper/mcmc_rndStart_M*_charge0.0_*seed2601*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
print("len names", len(datasets),len(pars_inj))
