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
import multiprocessing as mp
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

init_name = '../DataAnalysis/paper_runs/MCMC*T2.0*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))

    
print("len names", len(datasets),len(pars_inj))
cmap = plt.cm.get_cmap('Set1',)
colors = sns.color_palette('colorblind')
colors = [cmap(i) for i in range(len(datasets))]

ls = ['-','--','-.',':',(0, (2, 2)),(0, (1, 2))]
# provide custom line styles
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (3, 5, 1, 5, 1))]

# get Edot function
# trajELQ = EMRIInspiral(func="KerrEccentricEquatorial")
trajELQ = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
# from few.summation.interpolatedmodesum import CubicSplineInterpolant
#import cubic spline from scipy
from scipy.interpolate import CubicSpline

def get_dphi(params):
    lnM,lnmu,a,p0,e0,D_L,_,_,_,_,PhiP0,PhiR0,charge = params
    x0 = 1.0
    M = np.exp(lnM)
    mu = np.exp(lnmu)
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = trajELQ(M, mu, a, p0, e0, x0, charge, Phi_phi0=PhiP0, Phi_r0=PhiR0,  T=2.0, dt=10.0)
    v = get_fundamental_frequencies(a, p, e, x)[0]**(1/3) # which is approximately p0**(-0.5)
    interp = CubicSpline(v, Phi_phi)
    t_0, p, e, x, Phi_phi_0, Phi_theta, Phi_r = trajELQ(M, mu, a, p0, e0, x0, 0.0,  Phi_phi0=PhiP0, Phi_r0=PhiR0, T=2.0, dt=10.0)
    v_0 = get_fundamental_frequencies(a, p, e, x)[0]**(1/3) # which is approximately p0**(-0.5)
    interp_0 = CubicSpline(v_0, Phi_phi_0)
    sym_mass_ratio = mu*M / (mu+M)**2
    new_v = np.linspace(v[0],v[-1],100)
    # delta phi  * 3/(128 * sym_mass_ratio * v**(-7)) is the difference in phase
    factor = 3/(128 * sym_mass_ratio * v**(-7))
    return np.mean(np.abs(1-interp(new_v) / interp_0(new_v)) * new_v**(2))

def get_dphi_vacuum(params):
    lnM,lnmu,a,p0,e0,D_L,_,_,_,_,PhiP0,PhiR0 = params
    x0 = 1.0
    M = np.exp(lnM)
    mu = np.exp(lnmu)
    t_0, p, e, x, Phi_phi_0, Phi_theta, Phi_r = trajELQ(M, mu, a, p0, e0, x0, 0.0,  Phi_phi0=PhiP0, Phi_r0=PhiR0, T=2.0, dt=10.0)
    v_0 = get_fundamental_frequencies(a, p, e, x)[0]**(1/3) # which is approximately p0**(-0.5)
    interp_0 = CubicSpline(v_0, Phi_phi_0)
    sym_mass_ratio = mu*M / (mu+M)**2
    new_v = np.linspace(v[0],v[-1],100)
    # delta phi  * 3/(128 * sym_mass_ratio * v**(-7)) is the difference in phase
    factor = 3/(128 * sym_mass_ratio * v**(-7))
    return np.mean(np.abs(1-interp(new_v) / interp_0(new_v)) * new_v**(2))

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


Nsamp =int(5e4)

#----------------------------- delta phi ------------------------------------

plt.figure()
for filename,el,cc,ll in zip(datasets,pars_inj,colors,ls):
    label, toplot, truths = get_labels_chains(el)
    
    # alpha bound
    Lambda = toplot[:,-1]
    mask = (Lambda>0)
    # new samples
    toplot = toplot[mask]
    
    Lambda = toplot[:,-1]
    charge = np.sqrt(4*Lambda)
    newvar = toplot.copy()
    newvar[:,-1] = charge
    
    # use multiprocessing to obtain the results for dphi
    with mp.Pool(32) as pool:
        results = pool.map(get_dphi, newvar[:Nsamp])
    results = np.asarray(results)
    bins = np.linspace(0.0005, 0.003,40)
    plt.hist(results, weights=1/results, bins=bins, color=cc, label=label, histtype='step', linestyle=ll, density=True)


# plt.tight_layout()
# plt.xlabel(r'$\log_{10} \delta \phi $',size=22)# {\dot{E}}
# vpos = np.log10(1e-7) # bound from https://arxiv.org/pdf/2002.02030 table IV
# plt.ticklabel_format(style='sci')
# # # 
# plt.annotate('Bound from LVK and binary pulsars', xy=(vpos, 0.0), xytext=(vpos, 0.1),
#              arrowprops=dict(facecolor='black', shrink=0.001),
#              fontsize=12, ha='center')

plt.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0)$')
plt.savefig(f'./figures/bound_deltaphi.pdf', bbox_inches='tight')
# plt.legend()
# plt.xlim(-11.0,-6.0)
# plt.ylim(0.0,1.3)

