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
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_kerr_geo_constants_of_motion, get_fundamental_frequencies

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
    params = repo_name.split('_')[2:-1]
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
# init_name = '../DataAnalysis/paper_runs/vacuum*'
# datasets = sorted(glob.glob(init_name + '.h5'))
# pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
   
print("len names", len(datasets),len(pars_inj))

colors = sns.color_palette('colorblind')
# provide custom line styles
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (3, 5, 1, 5, 1))]

# get Edot function
trajELQ = EMRIInspiral(func="KerrEccentricEquatorial")
from scipy.interpolate import CubicSpline
import multiprocessing

def get_Edot(M, mu, a, p0, e0, x0, Lambda):
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
    trajELQ.inspiral_generator.integrator.add_parameters_to_holder(M, mu, a, np.asarray([Lambda]))
    y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)
    y0 = np.array([y1, y2, y3])
    # y0 = np.array([p0, e0, x0, 0.0, 0.0, 0.0])
    return trajELQ.inspiral_generator.integrator.get_derivatives(y0)[0] * (mu/M)

def get_delta_Edot(M, mu, a, p0, e0, x0, charge):
    # for defining B https://arxiv.org/pdf/1603.04075 and delta https://arxiv.org/pdf/1007.1995
    # https://arxiv.org/pdf/1007.1995
    # paper for pulsar constraints in ppE: https://arxiv.org/pdf/2002.02030
    Lambda = charge**2/4
    Edot_tot = get_Edot(M, mu, a, p0, e0, x0, Lambda)
    Edot_grav = get_Edot(M, mu, a, p0, e0, x0, 0.0)
    Edot_scal = Edot_tot - Edot_grav
    omphi = get_fundamental_frequencies(a, p0, e0, x0)[0]
    B = (Edot_scal/Edot_grav) * (omphi)**(2/3)# / p0
    return B


# check that get_delta_Edot scales with charge
plt.figure()
charge = np.linspace(1e-5,1e-1,100)
plt.plot(charge,[get_delta_Edot(1e6, 10., 0.9, 10.0, 0.1, 1.0, ch) for ch in charge])
# check p0 scaling
pvec = np.linspace(4.0,20.0,100)
plt.loglog(pvec,[np.abs(get_delta_Edot(1e6, 10., 0.95, pp, 0.4, 1.0, 1e-2)) for pp in pvec])
plt.savefig(f'./figures/check_charge_scaling.pdf', bbox_inches='tight')

Nsamp = int(1e4)

# #------------------------- delta plot ---------------------------------
# plt.figure()
# for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
#     label, toplot, truths = get_labels_chains(el)
    
#     Lambda = toplot[:,-1]
#     mask = (Lambda>0.0)
#     toplot = toplot[mask]
#     Lambda = Lambda[mask]

#     mu = np.exp(toplot[:,1])
#     M = np.exp(toplot[:,0])
#     a = toplot[:,2]
#     p0 = toplot[:,3]
#     e0 = toplot[:,4]
#     x0 = np.ones_like(e0)

#     # charge
#     charge = np.sqrt(4 * Lambda)
#     w_charge = 1 / np.sqrt(Lambda)
    
#     y = np.abs(np.asarray([get_delta_Edot(M[ii], mu[ii], a[ii], p0[ii], e0[ii], x0[ii], charge[ii])[0] for ii in range(Nsamp)]))
#     bins = np.linspace(-11.5,-1.0,num=30) #+ np.random.uniform(-0.05,-0.0001)
#     mask = (y!=0.0)
#     plt.hist(np.log10(y[mask]),weights=1/newvar[:Nsamp,6][mask], bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)
#     # 
#     # plt.axvline(np.quantile(np.log10(y),0.975),color=cc)

# plt.tight_layout()
# plt.xlabel(r'$\log_{10} \delta $',size=22)# {\dot{E}}
# vpos = np.log10(6e-4) # binary J1141-6545 from https://arxiv.org/pdf/1603.04075
# # approximately 60 000 orbits in  https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
# vpos = np.log10(1-0.999963) # binary PSR J0737–3039 from https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
# plt.ticklabel_format(style='sci')
# plt.annotate('Bound from binary \npulsar J0737–3039', xy=(vpos, 0.0), xytext=(vpos, 0.1),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center')

# plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
# # plt.legend()
# plt.xlim(-11.0,-2.0)
# # plt.ylim(0.0,1.3)
# plt.savefig(f'./figures/bound_delta.pdf', bbox_inches='tight')

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
    label, toplot, truths = get_labels_chains(el)
    Lambda = toplot[:,-1]
    mask = (Lambda>0.0)
    toplot = toplot[mask]
    Lambda = Lambda[mask]

    mu = np.exp(toplot[:,1])
    M = np.exp(toplot[:,0])
    a = toplot[:,2]
    p0 = toplot[:,3]
    e0 = toplot[:,4]
    x0 = np.ones_like(e0)

    # charge
    charge = np.sqrt(4 * Lambda)
    w_charge = 1 / np.sqrt(Lambda)
    
    
    def calculate_delta_Edot(index):
        Lambda = charge**2/4
        Edot_tot = get_Edot(M[index], mu[index], a[index], p0[index], e0[index], x0[index], charge[index]**2 /4)
        Edot_grav = get_Edot(M[index], mu[index], a[index], p0[index], e0[index], x0[index], 0.0)
        Edot_scal = Edot_tot - Edot_grav
        omphi = get_fundamental_frequencies(a[index], p0[index], e0[index], x0[index])[0]
        B = (Edot_scal/Edot_grav) * (omphi)**(2/3)# / p0
        q = -1.0 # power in B (1/r)^q
        eta = mu[index] * M[index] / (mu[index] + M[index])**2 # symmetric mass ratio
        beta = -15/64 * 1/(4-q) * 1/(5-2*q) * B * eta**(-2*q/5) # https://arxiv.org/pdf/1204.2585 also eq 29 of https://arxiv.org/pdf/1603.08955
        # beta(phi_n) eq 10 of https://arxiv.org/pdf/1603.08955
        b = 2*q-5 # power ppE
        delta_phi = 128/3 * beta * eta**(-2/5) # eq 21 https://arxiv.org/pdf/2002.02030
        # zeta_edgb = (16*np.pi*alpha_edgb**2 / m**4)
        zeta_edgb = beta * 7168 / 5 * eta**(18/5)
        sqrt_alpha_edgb = (np.abs(zeta_edgb)/ (16*np.pi))**(1/4) * M[index]*MRSUN_SI/1e3
        # our factor
        sqrt_alpha_edgb *= np.sqrt(16*np.sqrt(np.pi))
        # mapping to specific theories https://arxiv.org/pdf/1603.08955
        En = get_kerr_geo_constants_of_motion(a[index], p0[index], e0[index], x0[index])[0] * mu[index]
        return Edot_tot, Edot_grav, Edot_scal, En, B, beta, delta_phi, sqrt_alpha_edgb

    with multiprocessing.Pool(16) as pool:
        y = np.abs(pool.map(calculate_delta_Edot, range(Nsamp)))
    # y = np.abs(np.asarray([calculate_delta_Edot(el) for el in range(Nsamp)]))

    Edot_tot, Edot_grav, Edot_scal, En, B, beta, delta_phi,sqrt_alpha_edgb = y.T
    
    # 95% upper bounds on B,beta,delta_phi
    print(np.quantile(B,0.95), np.quantile(beta,0.95), np.quantile(delta_phi,0.95), np.quantile(sqrt_alpha_edgb,0.95))

    bins = 30#np.linspace(-7.,-3.0,num=30) #+ np.random.uniform(-0.05,-0.0001)
    
    plt.hist(np.log10(sqrt_alpha_edgb), weights=1/Lambda[:Nsamp], bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)

plt.tight_layout()
plt.xlabel(r'$\log_{10} \delta \dot{E} $',size=22)# {\dot{E}}

# # from fig 11 https://arxiv.org/pdf/2010.09010
# vpos = np.log10(1e-8)
# plt.annotate('Projected\ncumulative\nconstraint\n3G', xy=(vpos, -0.25), xytext=(vpos, -0.25),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center',
#              xycoords='data', annotation_clip=False
#              )

# xlow = -11.0
# # Create a bar
# plt.broken_barh([(xlow, vpos-xlow)], (0.65,0.1), edgecolor='black')#, facecolors='none')
# plt.text(vpos+0.1, 0.7, 'SOBHs observed by \nfuture ground-based detectors', ha='left', va='center', fontsize=12)
# # add the broken bar for the next valur of vpos

# vpos = np.log10(1e-10)
# plt.broken_barh([(xlow, vpos-xlow)], (0.8,0.1), edgecolor='black')#, facecolors='none')
# plt.text(vpos+0.1, 0.85, 'SOBHs observed by \nLISA + future ground-based detectors (multiband)', ha='left', va='center', fontsize=12)


plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$',loc='lower right')
# plt.legend()
# plt.xlim(-11.0,-4.0)
# plt.ylim(0.0,0.95)
plt.savefig(f'./figures/bound_deltaEdot.pdf', bbox_inches='tight')

# https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
beta2 = 4e-6
Pdot_precision = 1.3e-4
boundB_prx_dpsr = Pdot_precision * beta2

boundB_here =  np.std(Edot_grav)/np.median(Edot_grav) / p0
