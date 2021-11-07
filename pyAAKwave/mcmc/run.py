#!/u/lsperi/conda-envs/scalar_few_env/bin/python
import numpy as np
import matplotlib.pyplot as plt

from lisatools.sensitivity import get_sensitivity
from lisatools.diagnostic import (
    inner_product,
    snr,
    fisher,
    covariance,
    mismatch_criterion,
    cutler_vallisneri_bias,
    scale_snr,
)
import time


from scipy.interpolate import interp1d

# few
from few.utils.utility import (get_fundamental_frequencies, get_separatrix,
                               get_mu_at_t,
                               get_p_at_t,
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase, GenerateEMRIWaveform
from few.utils.baseclasses import Pn5AAK,ParallelModuleBase

from lisatools.sampling.likelihood import Likelihood

import emcee

from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler
from lisatools.sampling.prior import *
from lisatools.utils.transform import TransformContainer

import warnings
warnings.filterwarnings("ignore")

traj = EMRIInspiral(func="KerrCircFlux")

#######################################
# class definition

from few.utils.baseclasses import Pn5AAK, ParallelModuleBase

class ScalarAAKWaveform(AAKWaveformBase, Pn5AAK, ParallelModuleBase):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None, return_list=True,
    ):

        AAKWaveformBase.__init__(
            self,
            EMRIInspiral,  # trajectory class
            AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
            return_list=return_list
        )
#######################################################

use_gpu = True

inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e4),  # all of the trajectories will be well under len = 1000
    "func": "KerrCircFlux"
}

wave_gen = ScalarAAKWaveform(
            sum_kwargs=dict(pad_output=True),
            inspiral_kwargs=inspiral_kwargs, 
            use_gpu=use_gpu,
            return_list=True,
            )

# define injection parameters
M = 1e6
mu = 10.0
p0 = 7.0
e0 = 0.0
Y0 = 1.0

# scala charge
scalar_charge = 0.05

Phi_phi0 = 0.0
Phi_theta0 = 0.0
Phi_r0 = 0.0
# define other parameters necessary for calculation
a = 0.9
qS = 0.5420879369091457
phiS = 5.3576560705195275
qK = 1.7348119514252445
phiK = 3.2004167279159637
dist = 1.0

# injection array
injection_params = np.array(
    [
        np.log(M),
        np.log(mu),
        a,
        p0,
        e0,
        Y0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        scalar_charge
    ]
)

# define other quantities
T = 1.  # years
dt = 15.0

#################################################

###################################################################################
snr_goal = 150.0

# for SNR and covariance calculation
inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd", use_gpu=use_gpu)

# transformation of arguments from sampling basis to waveform basis
transform_fn_in ={0: (lambda x: np.exp(x)),
                    1: (lambda x: np.exp(x)),
                    }


# use the special transform container
transform_fn = TransformContainer(transform_fn_in)

# copy parameters to check values
check_params = injection_params.copy()

# this transforms but also gives the transpose of the input dimensions due to inner sampler needs
check_params = transform_fn.transform_base_parameters(check_params).T

# INJECTION kwargs
waveform_kwargs = {"T": T, "dt": dt, "mich":False}#, "eps": 1e-2}

####################################
# TIME

check_sig = wave_gen(*check_params, **waveform_kwargs)


plt.figure()
h_real = check_sig[0].get()
freq = np.fft.rfftfreq(len(h_real), dt)
h_tilde = np.fft.rfft(h_real)
plt.loglog(freq, np.abs(h_tilde))
plt.savefig('waveform')

st = time.time()
for i in range(10):
    check_sig = wave_gen(*check_params, **waveform_kwargs)
print("timing of the waveform", (time.time()-st)/10)
####################################

# adjust distance for SNR goal
check_sig, snr_orig = scale_snr(snr_goal, check_sig, return_orig_snr=True,**inner_product_kwargs) #, dt=dt

print("orig_dist:", injection_params[6])
injection_params[6] *= snr_orig / snr_goal
print("new_dist:", injection_params[6])


################################################################
# define sampling quantities
nwalkers = 16  # per temperature
ntemps = 1

ndim_full = 15  # full dimensionality of inputs to waveform model

# which of the injection parameters you actually want to sample over
#                    M, mu, a, p0, dist, phi_phi0
test_inds = np.array([0, 1, 2, 3,                   14])#, 6, 7, 8, 9, 11, 10, 12, 13])

# ndim for sampler
ndim = len(test_inds)

# need to get values to fill arrays for quantities that we are not sampling
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = injection_params[fill_inds]

# to store in sampler file, get injection points we are sampling
injection_test_points = injection_params[test_inds]

# instantiate the likelihood class
nchannels = 2
like = Likelihood(
    wave_gen, nchannels, dt=dt, parameter_transforms=transform_fn, use_gpu=use_gpu,
)

# inject
like.inject_signal(
    params=injection_params.copy(),
    waveform_kwargs=waveform_kwargs,
    noise_fn=get_sensitivity,
    noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
    add_noise=False,
    
)

# for checks
check_params = injection_params.copy()

check_params = np.tile(check_params, (6, 1))

####################################################################
perc = 5e-4

# define priors, it really can only do uniform cube at the moment
priors_in = {0: uniform_dist(injection_params[test_inds[0]]*(1-perc), injection_params[test_inds[0]]*(1+perc)), 
             1: uniform_dist(injection_params[test_inds[1]]*(1-perc), injection_params[test_inds[1]]*(1+perc)),
             2: uniform_dist(injection_params[test_inds[2]]*(1-perc), injection_params[test_inds[2]]*(1+perc)),
             3: uniform_dist(injection_params[test_inds[3]]*(1-1e-3), injection_params[test_inds[3]]*(1+1e-3)),
             4: uniform_dist(0.0,scalar_charge*2.),
#             6: uniform_dist(0.1, 2.),
#             7: uniform_dist(0.0, np.pi),
#             8: uniform_dist(0.0, 2 * np.pi),
#             9: uniform_dist(0.0, np.pi),
#             10: uniform_dist(0.0, 2 * np.pi),
#             11: uniform_dist(0.0, 2 * np.pi),
#             12: uniform_dist(0.0, 2 * np.pi),
#             13: uniform_dist(0.0, 2 * np.pi),
             }

# set periodic parameters
periodic = {
    7: np.pi,
    8: 2 * np.pi,
    9: np.pi,
    10: 2 * np.pi,
    11: 2 * np.pi,
    12: 2 * np.pi,
    13: 2 * np.pi,
}  # the indexes correspond to the index within test_inds

priors = PriorContainer(priors_in)

# can add extra temperatures of 1 to have multiple temps accessing the target distribution
ntemps_target_extra = 0
# define max temperature (generally should be inf if you want to sample prior)
Tmax = np.inf

# not all walkers can fit in memory. subset says how many to do at one time
subset = 4 # it was 4 before

# set kwargs for the templates
waveform_kwargs_templates = waveform_kwargs.copy()
###########################################################

labels = [
    r"$\ln M$",
    r"$ln\mu$",
    r"$a$",
    r"$p_0$",
    r"$q$",
    ]
###############################################################


# sampler starting points around true point
factor = 1e-9

# random starts
np.random.seed(3000)
start_points = np.zeros((nwalkers * ntemps, ndim))
print('---------------------------')
print('Priors')
for i in range(ndim):
    #if i==4:
    #    start_points[:, i] = uniform_dist(0.0,standard_A*1.2 ).rvs(size=nwalkers * ntemps)
    #else:
    start_points[:, i] = injection_params[test_inds][i]*(1. + factor*np.random.normal(size=nwalkers * ntemps))#np.random.multivariate_normal(injection_params[test_inds], inv_gamma/1000,size=nwalkers * ntemps)[:,i] #priors_in[i].rvs(size=nwalkers * ntemps) ##
    print('variable ',i)
    print(start_points[:, i])
print('---------------------------')

# check the starting points
start_test = np.zeros((nwalkers * ntemps, ndim_full))

# need to set sampling and non-sampling quantities
start_test[:, test_inds] = start_points
start_test[:, fill_inds] = fill_values

split_inds = np.split(np.arange(len(start_test)), int(len(start_test) / subset))

start_ll = np.asarray(
    [
        like.get_ll(start_test[split], waveform_kwargs=waveform_kwargs)
        for split in split_inds
    ]
)

print('ll',start_ll)


###############################################################
corner_kwargs=dict(labels=labels)

# setup sampler
sampler = PTEmceeSampler(
    nwalkers,
    ndim,
    ndim_full,
    like,
    priors,
    subset=subset,
    lnlike_kwargs={"waveform_kwargs": waveform_kwargs_templates},
    test_inds=test_inds,
    fill_values=fill_values,
    ntemps=ntemps,
    autocorr_multiplier=100, # automatic stopper, be careful with this since the parallel tempering 
    autocorr_iter_count=100, # how often it checks the autocorrelation
    ntemps_target_extra=ntemps_target_extra,
    Tmax=Tmax,
    injection=injection_test_points,
    plot_iterations=100,
    plot_source="emri",
#    periodic=periodic,
    fp="scalar_AAK_snr_{:d}_no_noise_{}_{}_{}_{}_{}_T{}.h5".format(
        int(snr_goal), M, mu, a, p0, scalar_charge, T
    ),
    resume=True, # very important
    plot_kwargs=dict(corner_kwargs=corner_kwargs),
#    sampler_kwargs=sampler_kwargs
)

thin_by = 1
max_iter = int(5e5)
sampler.sample(
    start_points,
    iterations=max_iter,
    progress=True,
    skip_initial_state_check=False,
    thin_by=thin_by,
)
