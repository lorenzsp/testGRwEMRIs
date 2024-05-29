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

def get_delta_Edot(M, mu, a, p0, e0, x0, Lambda):
    # for defining B https://arxiv.org/pdf/1603.04075 and delta https://arxiv.org/pdf/1007.1995
    # https://arxiv.org/pdf/1007.1995
    # paper for pulsar constraints in ppE: https://arxiv.org/pdf/2002.02030
    Edot_tot = get_Edot(M, mu, a, p0, e0, x0, Lambda)
    Edot_grav = get_Edot(M, mu, a, p0, e0, x0, 0.0)
    Edot_scal = Edot_tot - Edot_grav
    omphi = get_fundamental_frequencies(a, p0, e0, x0)[0]
    B = (Edot_scal/Edot_grav) * (omphi)**(2/3)# / p0
    return B


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
  'xtick.labelsize': 16,
  'ytick.labelsize': 16,
  'legend.title_fontsize' : 10,

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

def get_phi_n(pn_order, m1, m2, M, nu, chi_1, chi_2):
    # from appendix of https://arxiv.org/pdf/2203.13937.pdf
    gamma_E = 0.57721566490153286060  # Euler's constant
    delta = (m1 - m2) / M
    chi_S = (chi_1 + chi_2) / 2
    chi_A = (chi_1 - chi_2) / 2
    if int(2*pn_order) == 2:
        return 3715/756 + 55*nu/9
    elif int(2*pn_order) == 3:
        return -16*np.pi + 113*delta*chi_A/3 + (113/3 - 76*nu/3)*chi_S
    elif int(2*pn_order) == 4:
        return 15293365/508032 + 27145*nu/504 + 3085*nu**2/72 + (-405/8 + 200*nu)*chi_A**2 - 405*delta*chi_A*chi_S/4 + (-405/8 + 5*nu/2)*chi_S**2
    elif int(2*pn_order) == 5:
        return 38645*np.pi/756 - 65*np.pi*nu/9 + (-732985/2268 - 140*nu/9)*delta*chi_A + (-732985/2268 + 24260*nu/81 + 340*nu**2/9)*chi_S
    elif int(2*pn_order) == 6:
        return 11583231236531/4694215680 - 6848*np.log(4)/21 - 640*np.pi**2/3 + 6848*gamma_E/21 + (-15737765635/3048192 + 2255*np.pi**2/12)*nu + 76055*nu**2/1728 - 127825*nu**3/1296 + 2270*np.pi*delta*chi_A/3 + (2270*np.pi/3 - 520*np.pi*nu)*chi_S + (75515/288 - 547945*nu/504 - 8455*nu**2/24)*chi_A**2 + (75515/144 - 8225*nu/18)*delta*chi_A*chi_S + (75515/288 - 126935*nu/252 + 19235*nu**2/72)*chi_S**2
    elif int(2*pn_order) == 7:
        return 77096675*np.pi/254016 + 378515*np.pi*nu/1512 - 74045*np.pi*nu**2/756 + (-25150083775/3048192 + 26804935*nu/6048 - 1985*nu**2/48)*delta*chi_A + (-25150083775/3048192 + 10566655595*nu/762048 - 1042165*nu**2/3024 + 5345*nu**3/36)*chi_S
    else:
        return 1.0

def get_beta_dphi_from_B(B, pn_order, M, mu, a):
    # from eq 21 of https://arxiv.org/pdf/2002.02030
    eta = mu * M / (mu + M)**2 # symmetric mass ratio
    beta = -15/32 * 1/(4-pn_order) * 1/(5-2*pn_order) * B * eta**(-2*pn_order/5) # https://arxiv.org/pdf/1204.2585 also eq 29 of https://arxiv.org/pdf/1603.08955
    # beta(phi_n) eq 10 of https://arxiv.org/pdf/1603.08955
    b = 2*pn_order-5 # power ppE
    delta_phi = 128/3 * beta * eta**(2*pn_order/5) / get_phi_n(pn_order, M, mu, M+mu, eta, a, 0.0) # eq 21 https://arxiv.org/pdf/2002.02030
    return beta, delta_phi

def get_dphi_from_beta(beta, pn_order, M, mu, a):
    # from eq 21 of https://arxiv.org/pdf/2002.02030
    eta = mu * M / (mu + M)**2 # symmetric mass ratio
    # beta(phi_n) eq 10 of https://arxiv.org/pdf/1603.08955
    b = 2*pn_order-5 # power ppE
    delta_phi = 128/3 * beta * eta**(2*pn_order/5) / get_phi_n(pn_order, M, mu, M+mu, eta, a, 0.0) # eq 21 https://arxiv.org/pdf/2002.02030
    return delta_phi

def transform_Edot_to_constraints(pn_order, delta_Edot, Edot_grav, M, mu, a, p0, e0, x0):
    omphi = get_fundamental_frequencies(a, p0, e0, x0)[0]
    # eq 28 of https://arxiv.org/pdf/1603.08955
    # pn_order = q = -1.0 # power in B (1/r)^q or B * v^(2q)
    v = omphi**(1/3)
    B = (delta_Edot/Edot_grav) * (v)**(-2*pn_order)
    beta, delta_phi = get_beta_dphi_from_B(B, pn_order, M, mu, a)
    return B, beta, delta_phi

def like(inp,Edot_mean,Edot_sigma,pn_order=-1):
    M, mu, a, p0, e0, B = inp
    # return -inf if a is not in the right range
    if a < 0.0 or a > 1.0:
        return -np.inf
    if p0 < 0.0:
        return -np.inf
    if e0 < 0.0 or e0 > 1.0:
        return -np.inf
    x0 = 1.0
    omphi = get_fundamental_frequencies(a, p0, e0, x0)[0]
    v = omphi**(1/3)
    prediction = get_Edot(M, mu, a, p0, e0, x0, 0.0) * (1 + B * v**(2*pn_order))
    ll = -0.5*((Edot_mean - prediction)/Edot_sigma)**2
    return ll

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
datasets += ['../DataAnalysis/paper_runs/vacuumMCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_vacuum.h5']#,'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1.h5',] 
pars_inj += ['../DataAnalysis/paper_runs/vacuumMCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_vacuum_injected_pars.npy']#,'../DataAnalysis/paper_runs/MCMC_noise0.0_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw26_nt1_injected_pars.npy',]
   
print("len names", datasets,pars_inj)

colors = sns.color_palette('colorblind')
# provide custom line styles
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (3, 5, 1, 5, 1))]


# check that get_delta_Edot scales with charge
# plt.figure()
# charge = np.linspace(1e-5,1e-1,100)
# plt.plot(charge,[get_delta_Edot(1e6, 10., 0.9, 10.0, 0.1, 1.0, ch**2/4) for ch in charge])
# check p0 scaling
# pvec = np.linspace(4.0,20.0,100)
# plt.loglog(pvec,[np.abs(get_delta_Edot(1e6, 10., 0.95, pp, 0.4, 1.0, 1e-2)) for pp in pvec])
# plt.savefig(f'./figures/check_charge_scaling.pdf', bbox_inches='tight')

Nsamp = int(1e4)
#------------------------- delta phi ---------------------------------
pn_vec = [-1, 0, 0.5, 1, 1.5, 2]
# create list of B, beta, delta_phi
B_list = []
beta_list = []
delta_phi_list = []

for filename,el,cc in zip(datasets,pars_inj,colors):
    label, toplot, truths = get_labels_chains(el)
    Lambda = toplot[:,-1]
    mask = (Lambda>-100.0)
    toplot = toplot[mask]
    Lambda = Lambda[mask]

    mu = np.exp(toplot[:,1])
    M = np.exp(toplot[:,0])
    a = toplot[:,2]
    p0 = toplot[:,3]
    e0 = toplot[:,4]
    x0 = np.ones_like(e0)
    
    # charge
    # charge = np.sqrt(4 * Lambda)
    # w_charge = 1 / np.sqrt(Lambda)
    
    def calculate_delta_Edot(index):
        Edot_tot = get_Edot(M[index], mu[index], a[index], p0[index], e0[index], x0[index], Lambda[index])
        Edot_grav = get_Edot(M[index], mu[index], a[index], p0[index], e0[index], x0[index], 0.0)
        return Edot_tot, Edot_grav

    with multiprocessing.Pool(16) as pool:
        y = np.asarray(pool.map(calculate_delta_Edot, range(Nsamp)))

    
    if 'vacuum' in filename:
        Edot_tot, Edot_grav = y.T
        delta_Edot = np.median(Edot_grav)-Edot_grav
        
        # create an MCMC
        # define gaussian likelihood in Edot
        # use emcee to sample the likelihood
        # import emcee
        # ndim = 6
        # nwalkers = 12
        # nsteps = 10000
        # # initial guess
        # inp0 = np.asarray([M[0], mu[0], a[0], p0[0], e0[0], 0.0])
        
        # # create multivariate gaussian likelihood
        # def log_multi_gaussian(x,mean=0.0,invSigma=1):
        #     diff = x - mean
        #     return -0.5*diff.T @ invSigma @ diff
        
        # # define priors based on KDE of samples of M,mu,a,p0,e0
        # from scipy.stats import gaussian_kde
        # invC = np.linalg.inv(np.cov(np.asarray([M[:Nsamp],mu[:Nsamp],a[:Nsamp],p0[:Nsamp],e0[:Nsamp]])))
        # me = np.mean(np.asarray([M[:Nsamp],mu[:Nsamp],a[:Nsamp],p0[:Nsamp],e0[:Nsamp]]),axis=1)

        # def log_prior(inp):
        #     M, mu, a, p0, e0, B = inp
        #     return log_multi_gaussian(np.asarray([M, mu, a, p0, e0]),mean=me,invSigma=invC)
        
        # def log_prob(inp):
        #     # random choice among Edot_grav
        #     return log_prior(inp) + like(inp,np.mean(Edot_grav), np.std(Edot_grav),-1)
        
        # # create the sampler
        
        # # with multiprocessing.Pool(16) as pool:
        # pool=None
        # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        # # create corner plot with results
        # init_points = inp0 + 1e-8 * np.random.randn(nwalkers, ndim)
        # sampler.run_mcmc(init_points, nsteps, progress=True)
        # samples = sampler.get_chain(discard=1000, flat=True)
        # # add loglike to corner
        # ll = sampler.get_log_prob(discard=1000,flat=True)
        # samples = np.hstack((samples, ll[:,None]))
        # fig = corner.corner(samples, labels=[r"$M$", r"$\mu$", r"$a$", r"$p_0$", r"$e_0$", r"$B$", "loglike"],quantiles=[0.16, 0.5, 0.84],show_titles=True);plt.savefig(f'./figures/corner_vacuum.pdf', bbox_inches='tight')
        
        B, beta, delta_phi = np.asarray([[transform_Edot_to_constraints(pn_order, delta_Edot[index], Edot_grav[index], M[index], mu[index], a[index], p0[index], e0[index], x0[index]) for index in range(Nsamp)] for pn_order in pn_vec]).T
        # print(np.quantile(B,0.95,axis=0)[0],np.quantile(samples[:,-2],0.95))
    else:
        Edot_tot, Edot_grav = y.T
        delta_Edot = Edot_tot - Edot_grav
        B, beta, delta_phi = np.asarray([[transform_Edot_to_constraints(pn_order, delta_Edot[index], Edot_grav[index], M[index], mu[index], a[index], p0[index], e0[index], x0[index]) for index in range(Nsamp)] for pn_order in pn_vec]).T
    
    print(np.quantile(B,0.95,axis=0)[0],np.quantile(B,0.05,axis=0)[0])
    B_list.append(B)
    beta_list.append(beta)
    delta_phi_list.append(delta_phi)


from matplotlib.ticker import LogLocator
delta_phi_list = np.asarray(delta_phi_list)
# create two subplots one for the beta and one for PN order
fig, axs = plt.subplots(1, 1, figsize=(default_width, default_width * default_ratio*1.5))
axs.semilogy(["-1", "0", "0.5", "1", "1.5", "2"], np.quantile(delta_phi_list[-1],0.95,axis=0),'o',label='EMRI vacuum mapping',alpha=0.5,ms=10)
run_constraints = np.quantile(delta_phi_list[:-1,:,0],0.95,axis=1)
axs.semilogy(["-1"], run_constraints[4],'P',label='EMRI scalar charge mapping',alpha=0.5,ms=10)

axs.set_xlabel(r'PN order',fontsize=22)
axs.set_ylabel(r'$|\delta \varphi|$',fontsize=22)


axs.grid(axis='y')

# LVK bounds from Elise's paper
# axs.semilogy(["-1", "0", "0.5", "1", "1.5", "2"], dphi_lvk,'*',label='LVK')
axs.semilogy(["-1", "0", "0.5", "1", "1.5", "2"], [2e-5, 3e-1, 7e-2, 1e-1, 0.25, 3],'v',label='GW170817',alpha=0.5,ms=10)
axs.semilogy(["-1", "0", "0.5", "1", "1.5", "2"], [7e-3, 6e-1, 1.5e-1, 1e-1, 8e-1, 0.4],'D',label='GWTC-3',alpha=0.5,ms=10)
axs.semilogy(["-1", "0", "0.5", "1", "1.5", "2"], [8e-5, 5.0, 0.2, 0.2, 0.3, 3],'^',label='GW230529',alpha=0.5,ms=10)
axs.yaxis.set_major_locator(LogLocator(base=10.0,numticks=20))  # Set the number of y-axis ticks

# pulsar bounds # # https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
beta2 = 4e-6
Pdot_precision = 1.3e-4
boundB_prx_dpsr = 4e-10 # 95% quoted in the paper, 
# if we divide by two we obtain one sigma
B = np.random.normal(0,boundB_prx_dpsr / 2,size=10000) 
beta_dp, dphi_dp = get_beta_dphi_from_B(B, -1, 1.33818, 1.24886, 0.0)
axs.semilogy(["-1", "0", "0.5", "1"], [np.quantile(dphi_dp,0.95), 0.8e-4, 0.8, 10.0],'*',label='Double pulsar constraint',alpha=0.5,ms=20)
axs.set_ylim(0.5e-11,30)
plt.legend(loc='lower right')
plt.savefig(f'./figures/bound_delta_phi.pdf', bbox_inches='tight')  

# # create two subplots one for the beta and one for PN order
# fig, axs = plt.subplots(3, 1, sharex=True, figsize=(default_width, default_width * default_ratio*2))
# axs[0].semilogy(["-1", "0", "0.5", "1", "1.5", "2"], np.quantile(B_list[0],0.95,axis=0),'o')
# axs[1].semilogy(["-1", "0", "0.5", "1", "1.5", "2"], np.quantile(beta_list[0],0.95,axis=0),'o')
# axs[2].semilogy(["-1", "0", "0.5", "1", "1.5", "2"], np.quantile(delta_phi_list[0],0.95,axis=0),'o',label='Mapping a posteriori from vacuum')

# axs[0].semilogy(["-1"], np.quantile(np.abs(B_list[1]),0.95,axis=0)[0],'P')
# axs[1].semilogy(["-1"], np.quantile(np.abs(beta_list[1]),0.95,axis=0)[0],'P')
# axs[2].semilogy(["-1"], np.quantile(np.abs(delta_phi_list[1]),0.95,axis=0)[0],'P',label='Fitting for scalar charge')

# axs[2].set_xlabel('PN order',fontsize=22)
# axs[2].set_ylabel(r'$\delta \phi$',fontsize=22)
# axs[1].set_ylabel(r'$\beta$',fontsize=22)
# axs[0].set_ylabel(r'$B$',fontsize=22)

# axs[1].yaxis.set_major_locator(LogLocator(base=10.0,numticks=6))  # Set the number of y-axis ticks
# for el in axs:
#     el.grid(axis='y')

# # Terrestrial from https://arxiv.org/pdf/2010.09010 fig 7
# #           ["-1", "0",   "0.5", "1",  "1.5", "2"]
# beta_terr = [1e-10, 5e-7, 0.5e-6, 5e-6, 1e-4, 5e-3]
# from bilby.gw.conversion import *

# q = np.random.uniform(0,1,size=1000)
# Mc = 10**np.random.uniform(np.log10(5),2,size=1000)
# m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(Mc, q)

# [get_dphi_from_beta(bb, pno, m1, m2, 0.0) for bb,pno in zip(beta_terr,[-1, 0, 0.5, 1, 1.5, 2])]
# axs[1].semilogy(["-1", "0",   "0.5", "1",  "1.5", "2"], beta_terr,'X',label='Future ground based detectors')
# # bounds from papers

# # LVK bounds from Elise's paper
# # dphi_lvk = [2e-5, 6e-2, 7e-2, 1e-1, 7e-2, 0.4]
# # axs[2].semilogy(["-1", "0", "0.5", "1", "1.5", "2"], dphi_lvk,'*',label='LVK')
# axs[2].semilogy(["-1", "0", "0.5", "1", "1.5", "2"], [2e-5, 3e-1, 7e-2, 1e-1, 0.25, 3],'*',label='GW170817',alpha=0.5)
# axs[2].semilogy(["-1", "0", "0.5", "1", "1.5", "2"], [7e-3, 6e-1, 1.5e-1, 1e-1, 8e-1, 0.4],'D',label='GWTC-3',alpha=0.5)
# # pulsar bounds # # https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.041050
# beta2 = 4e-6
# Pdot_precision = 1.3e-4
# boundB_prx_dpsr = 4e-10 # 95% quoted in the paper, 
# # if we divide by two we obtain one sigma
# B = np.random.normal(0,boundB_prx_dpsr / 2,size=10000) 
# beta_dp, dphi_dp = get_beta_dphi_from_B(B, -1, 1.33818, 1.24886, 0.0)
# axs[2].semilogy(["-1", "0", "0.5", "1"], [np.quantile(dphi_dp,0.95), 0.8e-4, 0.8, 10.0],'^',label='Double pulsar constraint',alpha=0.5)

# plt.legend(loc='lower right')
# plt.savefig(f'./figures/bound_delta_phi.pdf', bbox_inches='tight')  

#------------------------- delta phi ---------------------------------
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
    
#     pn_vec = [-1, 0, 0.5, 1, 1.5, 2, 3, 3.5]
#     def calculate_delta_Edot(index):
#         Lambda = charge**2/4
#         Edot_tot = get_Edot(M[index], mu[index], a[index], p0[index], e0[index], x0[index], charge[index]**2 /4)
#         Edot_grav = get_Edot(M[index], mu[index], a[index], p0[index], e0[index], x0[index], 0.0)
#         delta_Edot = Edot_tot - Edot_grav
#         return np.asarray([transform_Edot_to_constraints(pn_order, delta_Edot, Edot_grav, M[index], mu[index], a[index], p0[index], e0[index], x0[index]) for pn_order in pn_vec])

#     with multiprocessing.Pool(16) as pool:
#         y = np.abs(pool.map(calculate_delta_Edot, range(Nsamp)))
#     # y = np.abs(np.asarray([calculate_delta_Edot(el) for el in range(Nsamp)]))

    
#     B, beta, delta_phi = y.T
#     breakpoint()
#     # 95% upper bounds on B,beta,delta_phi
#     print(np.quantile(B,0.95), np.quantile(beta,0.95), np.quantile(delta_phi,0.95))

#     bins = 30 # np.linspace(-7.,-3.0,num=30) #+ np.random.uniform(-0.05,-0.0001)
    
#     plt.hist(np.log10(B), weights=1/Lambda[:Nsamp], bins=bins, histtype='step', density=True, label=label, linewidth=3, ls=ll)#, color=cc)

# plt.tight_layout()
# plt.xlabel(r'$\log_{10} \delta \dot{E} $',size=22)# {\dot{E}}

# # # from fig 11 https://arxiv.org/pdf/2010.09010
# # vpos = np.log10(1e-8)
# # plt.annotate('Projected\ncumulative\nconstraint\n3G', xy=(vpos, -0.25), xytext=(vpos, -0.25),
# #              arrowprops=dict(facecolor='black', shrink=0.001),
# #              fontsize=12, ha='center',
# #              xycoords='data', annotation_clip=False
# #              )

# # xlow = -11.0
# # # Create a bar
# # plt.broken_barh([(xlow, vpos-xlow)], (0.65,0.1), edgecolor='black')#, facecolors='none')
# # plt.text(vpos+0.1, 0.7, 'SOBHs observed by \nfuture ground-based detectors', ha='left', va='center', fontsize=12)
# # # add the broken bar for the next valur of vpos

# # vpos = np.log10(1e-10)
# # plt.broken_barh([(xlow, vpos-xlow)], (0.8,0.1), edgecolor='black')#, facecolors='none')
# # plt.text(vpos+0.1, 0.85, 'SOBHs observed by \nLISA + future ground-based detectors (multiband)', ha='left', va='center', fontsize=12)


# plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$',loc='lower right')
# # plt.legend()
# # plt.xlim(-11.0,-4.0)
# # plt.ylim(0.0,0.95)
# plt.savefig(f'./figures/bound_deltaEdot.pdf', bbox_inches='tight')


# boundB_here =  np.std(Edot_grav)/np.median(Edot_grav) / p0
