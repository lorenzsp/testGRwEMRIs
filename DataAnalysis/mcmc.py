#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# python mcmc.py -Tobs 2 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0025 -dev 7 -nwalkers 26 -ntemps 1 -nsteps 1000 -outname test -vacuum 0
# python mcmc.py -Tobs 0.5 -Tplunge 0.5 -dt 0.7 -M 3.6e4 -mu 3.6 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0 -dev 7 -nwalkers 26 -ntemps 1 -nsteps 1000 -outname test -vacuum 0
# test with zero likelihood
# python mcmc.py -Tobs 0.01 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0 -dev 7 -nwalkers 16 -ntemps 1 -nsteps 5000 -outname test -zerolike 1


import argparse
import os
print("PID:",os.getpid())
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
import time
parser = argparse.ArgumentParser(description="MCMC of EMRI source")
parser.add_argument("-Tobs", "--Tobs", help="Observation Time in years", required=True, type=float)
parser.add_argument("-M", "--M", help="MBH Mass in solar masses", required=True, type=float)
parser.add_argument("-mu", "--mu", help="Compact Object Mass in solar masses", required=True, type=float)
parser.add_argument("-a", "--a", help="dimensionless spin", required=True, type=float)
parser.add_argument("-p0", "--p0", help="Semi-latus Rectum", required=True, type=float)
parser.add_argument("-e0", "--e0", help="Eccentricity", required=True, type=float)
parser.add_argument("-x0", "--x0", help="prograde orbits", default=1.0, required=False, type=float)
parser.add_argument("-charge", "--charge", help="Scalar Charge", required=True, type=float)
parser.add_argument("-dev", "--dev", help="Cuda Device", required=False, type=int, default=0)
parser.add_argument("-dt", "--dt", help="sampling interval delta t", required=False, type=float, default=10.0)
parser.add_argument("-nwalkers", "--nwalkers", help="number of MCMC walkers", required=True, type=int)
parser.add_argument("-ntemps", "--ntemps", help="number of MCMC temperatures", required=True, type=int)
parser.add_argument("-nsteps", "--nsteps", help="number of MCMC iterations", required=False, type=int, default=1000)
parser.add_argument("-SNR", "--SNR", help="SNR", required=False, type=float, default=50.0)
parser.add_argument("-outname", "--outname", help="output name", required=False, type=str, default="MCMC")
parser.add_argument("-zerolike", "--zerolike", help="zero likelihood test", required=False, type=int, default=0)
parser.add_argument("-noise", "--noise", help="noise injection on=1, off=0", required=False, type=float, default=0.0)
parser.add_argument("-vacuum", "--vacuum", help="mcmc in vacuum, vacuum=1 sampling with a vacuum template", required=False, type=int, default=0)
parser.add_argument("-Tplunge", "--Tplunge", help="Time to plunge in years to set p0", required=False, type=float, default=2.0)
args = vars(parser.parse_args())

# select the plunge time
Tplunge = args["Tplunge"]

os.system("CUDA_VISIBLE_DEVICES="+str(args['dev']))
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['dev'])
os.system("echo $CUDA_VISIBLE_DEVICES")
import sys
sys.path.append('/data/lsperi/lisa-on-gpu/')
sys.path.append('/data/lsperi/Eryn/')
import matplotlib.pyplot as plt


import numpy as np

from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend
import corner
from eryn.moves import GaussianMove, StretchMove

from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *

from lisatools.sensitivity import get_sensitivity

from eryn.utils import TransformContainer

from scipy.signal.windows import tukey

# from fastlisaresponse import ResponseWrapper
from powerlaw import powerlaw_dist
 
from few.waveform import AAKWaveformBase
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation,KerrAAKSummation
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix, get_fundamental_frequencies
import warnings
warnings.filterwarnings("ignore")

SEED = 2601

try:
    import cupy as xp
    # set GPU device
    gpu_available = True
    print("using gpus")

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using 
use_gpu = True

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

# define trajectory
func = "KerrEccentricEquatorialAPEX"
insp_kwargs = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    # "max_init_len": int(1e4),
    "use_rk4": False,
    "func": func,
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

from utils import *

# function call
def run_emri_pe(
    emri_injection_params, 
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    nsteps,
    emri_kwargs={},
    vacuum=False,
    source_SNR=50.0,
    zero_like=False,
    noise=1.0,
    verbose=False
):
    """
    Run the parameter estimation for an extreme mass-ratio inspiral (EMRI) event.

    Args:
        emri_injection_params (array-like): The parameters of the EMRI waveform to be injected.
        Tobs (float): The observation time in seconds.
        dt (float): The time step size in seconds.
        fp (str): File path.
        ntemps (int): The number of temperatures for the parallel tempering algorithm.
        nwalkers (int): The number of walkers for the MCMC algorithm.
        nsteps (int): The number of steps for the MCMC algorithm.
        emri_kwargs (dict, optional): Additional keyword arguments for the EMRI waveform generation. Defaults to {}.
        vacuum (bool, optional): Whether to run a vacuum MCMC. Defaults to False.
        source_SNR (float, optional): The desired signal-to-noise ratio (SNR) of the injected waveform. Defaults to 50.0.
        zero_like (bool, optional): Whether to set the likelihood to zero. Defaults to False.
        noise (float, optional): The noise level to be added to the waveform. Defaults to 1.0.

    Returns:
        None
    """
    # fix seed for reproducibility and noise injection
    np.random.seed(SEED)
    xp.random.seed(SEED)
    
    # FEW waveform with specified AAK summation and inspiral
    few_gen = GenerateEMRIWaveform(
    AAKWaveformBase, 
    EMRIInspiral,
    KerrAAKSummation,
    # when using intrinsic only , we return a list
    return_list=True,
    inspiral_kwargs=insp_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    frame=None
    )
    
    # sets the proper number of points and what not
    print("use gpus, use vacuum", use_gpu, vacuum)
    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    t_arr = xp.arange(N_obs) * dt

    # inner product
    inner_kw = dict(dt=dt,PSD="lisasens",PSD_args=(),PSD_kwargs={},use_gpu=use_gpu,)
    
    # orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5" 
    orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen ="1st generation"# "2nd generation"#

    order = 20  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 10000.0  # throw away on both ends when our orbital information is weird
   
    # resp_gen = ResponseWrapper(
    #     few_gen,
    #     Tobs,
    #     dt,
    #     index_lambda,
    #     index_beta,
    #     t0=t0,
    #     flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
    #     use_gpu=use_gpu,
    #     is_ecliptic_latitude=False,  # False if using polar angle (theta)
    #     remove_garbage=True,#"zero",  # removes the beginning of the signal that has bad information
    #     **tdi_kwargs_esa,
    # )
    
    h_plus = few_gen(*emri_injection_params,**emri_kwargs)[0]

    len_tot = len(h_plus)
    window = xp.asarray(tukey(len_tot,alpha=0.005))
    def wave_gen(*args, **kwargs):
        temp_data_channels = few_gen(*args, **kwargs)
        return [el*window for el in temp_data_channels]

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 15,
       "fill_values": emri_injection_params[np.array([ 5, 12])], # inclination and Phi_theta
       "fill_inds": np.array([ 5, 12]),
    }

    (
        M,  
        mu,
        a, 
        p0, 
        e0, 
        x0,
        dist, 
        qS,
        phiS,
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0,
        Lambda
    ) = emri_injection_params
    injection_values = np.asarray([M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, Lambda])
    
    # emri_injection_params are the sampling parameters
    # get the sampling parameters
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[1] = np.log(emri_injection_params[1])
    
    # do conversion only when sampling over all parameters
    emri_injection_params[7] = np.cos(emri_injection_params[7]) 
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9]) 
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)
    
    if vacuum:
        # in the vacuum case we do not sample over Lambda and emri_injection_params is fixed to zero, which means a vacuum signal
        emri_injection_params[-1] = 0.0
        fill_dict = {
            "ndim_full": 15,
            "fill_values": emri_injection_params[np.array([ 5, 12, 14])], # inclination and Phi_theta and charge
            "fill_inds": np.array([ 5, 12, 14]),
            }
    else:
        prior_charge = uniform_dist(-0.6, 0.6)

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    if vacuum:
        parameter_transforms = {
            0: np.exp,  # M 
            1: np.exp,  # mu
            7: np.arccos, # qS
            9: np.arccos,  # qK
        }
    else:
        parameter_transforms = {
            0: np.exp,  # M 
            1: np.exp,  # mu
            7: np.arccos, # qS
            9: np.arccos,  # qK
        }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )
    
    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    
    # get SNR
    temp_emri_kwargs = emri_kwargs.copy()
    temp_emri_kwargs['T'] = Tplunge
    temp_data_channels = few_gen(*injection_values, **temp_emri_kwargs)
    temp_data_channels = [el*xp.asarray(tukey(len(temp_data_channels[0]),alpha=0.005)) for el in temp_data_channels]

    ############################## distance based on SNR ########################################################
    check_snr = snr([temp_data_channels[0], temp_data_channels[1]],**inner_kw)

    dist_factor = check_snr.get() / source_SNR
    emri_injection_params[6] *= dist_factor
    injection_values[6] *= dist_factor
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    print("new distance based on SNR", emri_injection_params[6])
    ##########################################################################################    
    print("difference between injected and transform", injection_values - transform_fn.both_transforms(emri_injection_params_in))
    # get injection
    data_channels = wave_gen(*injection_values, **emri_kwargs)
    tic = time.perf_counter()
    [wave_gen(*injection_values, **emri_kwargs) for _ in range(10)]
    toc = time.perf_counter()
    print("timing",(toc-tic)/10, "len vec", len(data_channels[0]))
    
    
    check_snr = snr([data_channels[0], data_channels[1]],**inner_kw)
    
    print("SNR",check_snr)
    ############################## priors ########################################################
    
    delta = 0.01
    
    # priors
    
    if vacuum:
        priors_in = {
                0: uniform_dist(emri_injection_params_in[0]*(1-delta), emri_injection_params_in[0]*(1+delta)),  # ln M
                1: uniform_dist(emri_injection_params_in[1]*(1-delta), emri_injection_params_in[1]*(1+delta)),  # ln mu
                2: uniform_dist(emri_injection_params_in[2]*(1-delta), emri_injection_params_in[2]*(1+delta)),  # a
                3: uniform_dist(emri_injection_params_in[3]*(1-delta), emri_injection_params_in[3]*(1+delta)),  # p0
                4: uniform_dist(emri_injection_params_in[4]*(1-delta), emri_injection_params_in[4]*(1+delta)),  # e0
                5: powerlaw_dist(0.01,10.0),  # dist in Gpc
                6: uniform_dist(-0.99999, 0.99999),  # qS
                7: uniform_dist(0.0, 2 * np.pi),  # phiS
                8: uniform_dist(-0.99999, 0.99999),  # qK
                9: uniform_dist(0.0, 2 * np.pi),  # phiK
                10: uniform_dist(0.0, np.pi),  # Phi_phi0
                11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
    else:
        priors_in = {
                0: uniform_dist(emri_injection_params_in[0]*(1-delta), emri_injection_params_in[0]*(1+delta)),  # ln M
                1: uniform_dist(emri_injection_params_in[1]*(1-delta), emri_injection_params_in[1]*(1+delta)),  # ln mu
                2: uniform_dist(emri_injection_params_in[2]*(1-delta), emri_injection_params_in[2]*(1+delta)),  # a
                3: uniform_dist(emri_injection_params_in[3]*(1-delta), emri_injection_params_in[3]*(1+delta)),  # p0
                4: uniform_dist(emri_injection_params_in[4]*(1-delta), emri_injection_params_in[4]*(1+delta)),  # e0
                5: powerlaw_dist(0.01,10.0),  # dist in Gpc
                6: uniform_dist(-0.99999, 0.99999),  # qS
                7: uniform_dist(0.0, 2 * np.pi),  # phiS
                8: uniform_dist(-0.99999, 0.99999),  # qK
                9: uniform_dist(0.0, 2 * np.pi),  # phiK
                10: uniform_dist(0.0, np.pi),  # Phi_phi0
                11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
                12: prior_charge,  # charge
            }
    
    priors = {
        "emri": ProbDistContainer(priors_in) 
    }

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "emri": {7: 2 * np.pi, 9: 2 * np.pi, 10: np.pi, 11: 2 * np.pi}
    }
    
    ############################## likelihood ########################################################
    # this is a parent likelihood class that manages the parameter transforms
    like = Likelihood(
        wave_gen,
        2,  # channels (A,E)
        dt=dt,
        parameter_transforms={"emri": transform_fn},
        use_gpu=use_gpu,
        vectorized=False,
        transpose_params=False,
        subset=3,  # may need this subset
    )

    def get_noise_injection(N, dt, sens_fn="lisasens",sym=False):
        freqs = xp.fft.fftfreq(N, dt)
        df_full = xp.diff(freqs)[0]
        freqs[0] = freqs[1]
        psd = [get_sensitivity(freqs,sens_fn=sens_fn), get_sensitivity(freqs,sens_fn=sens_fn)]
        # psd = [get_sensitivity_stas(freqs.get(),sens_fn=sens_fn), get_sensitivity_stas(freqs.get(),sens_fn=sens_fn)]
        # psd = [xp.asarray(psd_temp) for psd_temp in psd]
        
        # normalize by the factors:
        # 1/dt because when you take the FFT of the noise in time domain
        # 1/sqrt(4 df) because of the noise is sqrt(S / 4 df)
        noise_to_add_FD = [xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0]))+ 1j * xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0])) for psd_temp in psd]
        if sym:
            for ii in range(2):
                noise_to_add_FD[ii][freqs<0.0] = noise_to_add_FD[ii][freqs>0.0][::-1].conj()
        # noise_to_add = [xp.fft.ifft(xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0]))+ 1j * xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0])) ).real for psd_temp in psd]
        noise_to_add = [xp.fft.ifft(el).real for el in noise_to_add_FD]
        return [noise/(dt*np.sqrt(2*df_full)) * noise_to_add[0], noise/(dt*np.sqrt(2*df_full)) * noise_to_add[1]]

    full_noise = get_noise_injection(len_tot,dt)
    print("check nosie value",full_noise[0][0],full_noise[1][0])
    print("noise check ", inner_product(full_noise,full_noise, **inner_kw)/len(data_channels[0]) )
    print("matched SNR ", inner_product(full_noise[0]+data_channels[0],data_channels[0], **inner_kw)/inner_product(data_channels[0],data_channels[0], **inner_kw)**0.5 ) 
    
    nchannels = 2
    
    like.inject_signal(
        data_stream=[data_channels[0]+full_noise[0][:len(data_channels[0])], data_channels[1]+full_noise[1][:len(data_channels[0])]],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "lisasens"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndim = 13
    if vacuum:
        ndim = 12
    print('Sampling in ',ndim,' dimensions')
    ############################## plots ########################################################
    if use_gpu:
        get_spectrogram(data_channels[0],dt,fp[:-3] + "_spectrogram.pdf")
    
        plt.figure()
        
        ffth = xp.fft.rfft(data_channels[0]+full_noise[0][:len(data_channels[0])])*dt
        fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)
        plt.plot(fft_freq.get(), (xp.abs(ffth)**2).get())
        
        ffth = xp.fft.rfft(data_channels[0])*dt
        fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)
        plt.plot(fft_freq.get(), (xp.abs(ffth)**2).get())
        
        for el in ["lisasens"]:
            PSD_arr = get_sensitivity(fft_freq+1e-8, sens_fn=el)/ (4 * xp.diff(fft_freq)[0])
            plt.loglog(fft_freq.get()+1e-8, PSD_arr.get(),label=el,alpha=0.5)
        plt.legend()
        plt.savefig(fp[:-3] + "injection_fd.pdf")

        plt.figure()
        plt.plot(np.arange(len(data_channels[0].get()))*dt,  (data_channels[0]+full_noise[0][:len(data_channels[0])]).get())
        plt.savefig(fp[:-3] + "injection_td.pdf")
        
        plt.figure()
        for cc in 10**np.linspace(-5,-2,num=20):
            injection_temp = injection_values.copy()
            injection_temp[-1] = cc**2/4
            data_temp = wave_gen(*injection_temp, **emri_kwargs)
            
            Overlap = inner_product([data_channels[0], data_channels[1]],[data_temp[0], data_temp[1]],normalize=True,**inner_kw)
            plt.loglog(cc, 1-Overlap.get(),'ko')
        plt.ylabel('Mismatch')
        plt.xlabel('Charge')
        plt.savefig(fp[:-3] + "mismatch_evolution.pdf")
        # plt.savefig("mismatch_evolution.pdf")
    
    else:
        plt.figure()
        plt.plot(data_channels[0])
        plt.show()
        plt.savefig(fp[:-3] + "injection.pdf")

    #####################################################################
    # generate starting points
    try:
        file  = HDFBackend(fp)
        burn = int(file.iteration*0.25)
        thin = 1
        
        # get samples
        toplot = file.get_chain(discard=burn, thin=thin)['emri'][:,0][file.get_inds(discard=burn, thin=thin)['emri'][:,0]] # np.load(fp.split('.h5')[0] + '/samples.npy') # 
        
        cov = np.load(fp[:-3] + "_covariance.npy")
        tmp = toplot[:nwalkers*ntemps]
        tmp[0] = emri_injection_params_in.copy()
        toplot = np.load(fp[:-3] + "_samples.npy") # 
        print("covariance imported")
    except:
        print("find starting points")
        # precision of 1e-5
        cov = np.load("covariance.npy") * 2.38**2 /ndim /1000
        if vacuum:
            cov = cov[:-1,:-1]
        # increase the size of the covariance only along the last direction
        tmp = draw_initial_points(emri_injection_params_in, cov/10., nwalkers*ntemps)
        # set one to the true value
        tmp[0] = emri_injection_params_in.copy()

        cov = (np.cov(tmp,rowvar=False) +1e-20*np.eye(ndim))* 2.38**2 / ndim        
    #########################################################################
    # save parameters
    np.save(fp[:-3] + "_injected_pars",emri_injection_params_in)
    
    logp = priors["emri"].logpdf(tmp)
    print("logprior",logp)
    # draw again for infinit prior
    if int(np.sum(np.isinf(logp)))>0:
        tmp[np.isinf(logp)] =  priors["emri"].rvs(int(np.sum(np.isinf(logp))))
    tic = time.time()
    start_like = like(tmp, **emri_kwargs)
    toc = time.time()
    timelike = (toc-tic)/np.prod(tmp.shape)
    start_params = tmp.copy()
    print("start like",start_like, "in ", timelike," seconds")
    start_prior = priors["emri"].logpdf(start_params)
    # true likelihood
    true_like = like(emri_injection_params_in[None,:], **emri_kwargs)
    print("true log like",true_like)

    # start state
    start_state = State(
        {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
        log_like=start_like.reshape(ntemps, nwalkers), 
        log_prior=start_prior.reshape(ntemps, nwalkers)
    )

    # gibbs sampling
    update_all = np.repeat(True,ndim)
    update_none = np.repeat(False,ndim)
    indx_list = []
    
    def get_True_vec(ind_in):
        out = update_none.copy()
        out[ind_in] = update_all[ind_in]
        return out
    
    # gibbs variables
    # import itertools
    # stuff = np.arange(ndim)
    # list_comb = []
    # for subset in itertools.combinations(stuff, 2):
    #     list_comb.append(subset)
    # [indx_list.append(get_True_vec([el[0],el[1]])) for el in list_comb]
    # [indx_list.append(get_True_vec(np.arange(ndim))) for el in list_comb]
    
    sky_periodic = [("emri",el[None,:] ) for el in [get_True_vec([6,7]), get_True_vec([8,9])]]

    # # MCMC moves (move, percentage of draws)
    indx_list.append(get_True_vec([5,6,7,8,9,10,11]))
    if vacuum:
        indx_list.append(get_True_vec([0,1,2,3,4]))
    else:
        indx_list.append(get_True_vec([0,1,2,3,4,12]))
    indx_list.append(get_True_vec(np.arange(ndim)))

    gibbs_setup_start = [("emri",el[None,:] ) for el in indx_list]
    
    moves = [
        (StretchMove(use_gpu=use_gpu, live_dangerously=True),0.5),
        (GaussianMove({"emri": cov}, mode="AM", sky_periodic=sky_periodic, factor=100.0, indx_list=gibbs_setup_start),0.5),
    ]

    def stopping_fn(i, res, samp):
        current_it = samp.iteration
        discard = int(current_it*0.5)
        check_it = 1000
        update_it = 1000
        max_it_update = int(1e4)
        
        if (current_it>=check_it)and(current_it % check_it == 0):
            # check acceptance and max loglike
            print("max last loglike", samp.get_log_like()[-1])
            print("acceptance", samp.acceptance_fraction )
            print("Temperatures", 1/samp.temperature_control.betas)
            current_acceptance_rate = np.mean(samp.acceptance_fraction)
            print("current acceptance rate", current_acceptance_rate)
            # plot
            if (not zero_like)and(current_it<max_it_update)and(verbose):
                # get samples
                samples = sampler.get_chain(discard=discard, thin=1)["emri"][:, 0].reshape(-1, ndim)
                ll = samp.get_log_like(discard=discard, thin=1)[:,0].flatten()
                
                fig = corner.corner(np.hstack((samples,ll[:,None])),truths=np.append(emri_injection_params_in,true_like)); fig.savefig(fp[:-3] + "_corner.png", dpi=150)
                
                if (current_it<max_it_update):
                    num = 10
                    # beginning
                    start_ind = int(N_obs/4)
                    end_ind = start_ind + 500
                    plt.figure()
                    for i in range(num):
                        h_temp = wave_gen(*transform_fn.both_transforms(samples[i][None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                        plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),alpha=0.5,label=f'{ll[i]}')
                    h_temp = wave_gen(*transform_fn.both_transforms(emri_injection_params_in[None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                    plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),'k',alpha=0.3)
                    plt.legend()
                    plt.savefig(fp[:-3] + "_td_1over4.pdf")
                    
                    # end
                    start_ind = int(3*N_obs/4)
                    end_ind = start_ind + 500
                    plt.figure()
                    for i in range(num):
                        h_temp = wave_gen(*transform_fn.both_transforms(samples[i][None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                        plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),alpha=0.5,label=f'{ll[i]}')
                    h_temp = wave_gen(*transform_fn.both_transforms(emri_injection_params_in[None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                    plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),'k',alpha=0.3)
                    plt.legend()
                    plt.savefig(fp[:-3] + "_td_3over4.pdf")
                    

        if (current_it<max_it_update)and(current_it>=update_it)and(current_it % update_it == 0):            
            # update moves from chain
            chain = samp.get_chain(discard=discard)['emri'][:,:]
            inds = samp.get_inds(discard=discard)['emri'][:,:]
            to_cov = chain[inds]
            # update DE chain
            samp.moves[1].chain = to_cov.copy()
            # update cov and svd
            samp_cov = np.cov(to_cov,rowvar=False) * 2.38**2 / ndim
            svd = np.linalg.svd(samp_cov)
            samp.moves[1].all_proposal['emri'].svd = svd
            samp.moves[1].all_proposal['emri'].scale = samp_cov
            # save cov
            np.save(fp[:-3] + "_covariance", samp_cov)
            np.save(fp[:-3] + "_samples", to_cov)

        if (i==0)and(current_it>1):
            print("resuming run calculate covariance from chain")            
            samp_cov = np.load(fp[:-3] + "_covariance.npy")
            svd = np.linalg.svd(samp_cov)
            samp.moves[1].all_proposal['emri'].svd = svd
            samp.moves[1].all_proposal['emri'].scale = samp_cov.copy()
            # get DE chain
            samp.moves[1].chain = np.load(fp[:-3] + "_samples.npy").copy()
            
        plt.close()
        return False
    

    # check for previous runs
    try:
        file_samp = HDFBackend(fp)
        last_state = file_samp.get_last_sample()
        inds = last_state.branches_inds.copy()
        new_coords = last_state.branches_coords.copy()
        coords = new_coords.copy()
        resume = True
        print("resuming")
    except:
        resume = False
        print("file not found")

    def new_like(params, **kargs):
        if zero_like:
            return np.zeros(len(params))
        # to avoid nans
        like_val = np.zeros(len(params))-1e300
        like_val = like(params,**kargs)
        like_val[np.isnan(like_val)] = np.zeros(like_val[np.isnan(like_val)].shape)-1e300
        return like_val
    
    # prepare sampler
    sampler = EnsembleSampler(
        nwalkers,
        [ndim],  # assumes ndim_max
        new_like,
        priors,
        # SNR is decreased by a factor of 1/sqrt(T)
        tempering_kwargs={"ntemps": ntemps, "adaptive": True, "Tmax": 50**2/20**2},
        moves=moves,
        kwargs=emri_kwargs,
        backend=fp,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        stopping_fn=stopping_fn,
        stopping_iterations=1,
        branch_names=["emri"],
        track_moves=True,
    )
    
    if resume:
        log_prior = sampler.compute_log_prior(coords, inds=inds)
        log_like = sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]
        print("initial loglike", log_like)
        start_state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

    if zero_like:
        # start state
        start_state = State(
            {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
            log_like=start_like.reshape(ntemps, nwalkers)*0.0, 
            log_prior=start_prior.reshape(ntemps, nwalkers)
        )

    out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=1, burn=0)

    # get samples
    samples = sampler.get_chain(discard=int(sampler.iteration*0.25), thin=1)["emri"][:, 0].reshape(-1, ndim)
    
    # plot
    fig = corner.corner(samples,truths=emri_injection_params_in, levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2))
    fig.savefig(fp[:-3] + "_corner.png", dpi=150)
    return

if __name__ == "__main__":
    # set parameters
    M = args["M"]  # 1e6
    a = args["a"]
    mu = args["mu"]  # 10.0
    p0 = args["p0"]  # 12.0
    e0 = args["e0"]  # 0.35
    x0 = args["x0"]  # will be ignored in Schwarzschild waveform
    qK = np.pi/4  # polar spin angle
    phiK = np.pi/3 # azimuthal viewing angle
    qS = np.pi/2 # polar sky angle
    phiS = np.pi # azimuthal viewing angle
    get_plot_sky_location(qK,phiK,qS,phiS)
    dist = 3.0  # distance
    Phi_phi0 = np.pi/3
    Phi_r0 = np.pi
    Phi_theta0 = Phi_r0
    # LVK bound from paper sqrt(alpha) = 1.1 km 
    # bound in our scaling sqrt(alpha) = 1.1*np.sqrt(16*np.pi**0.5)
    # 0.4 extremal bound from Fig 21 https://arxiv.org/pdf/2010.09010.pdf
    # sqrt_alpha = 0.3 * np.sqrt( 16 * np.pi**0.5 )
    # d = (sqrt_alpha/(mu*MRSUN_SI/1e3))**2 / 2
    # d = (0.4 * np.sqrt( 16 * np.pi**0.5 )/(mu*MRSUN_SI/1e3))**2 / 2
    Lambda = args['charge']**2/4
    charge = args['charge']
    # flag for vacuum runs
    vacuum = bool(args["vacuum"])
    # observation span
    Tobs = args["Tobs"]  # years
    # sampling interval
    dt = args["dt"]  # seconds
    source_SNR = args["SNR"]

    ntemps = args["ntemps"]
    nwalkers = args["nwalkers"]

    traj = EMRIInspiral(func=func)
    # fix p0 given T
    # p0 = get_p_at_t(
    #     traj,
    #     Tplunge * 0.999,
    #     [M, mu, a, e0, x0, 0.0],
    #     bounds=[get_separatrix(a,e0,x0)+0.1, 100.0],
    #     traj_kwargs={"dt":dt,"err":insp_kwargs['err']},
        
    # )
    v = np.abs(get_fundamental_frequencies(a,p0, e0, x0)[0])**(1/3)
    print("new p0 fixed by Tobs, p0=", p0)
    print("new v fixed by Tobs, v=", v)
    
    tic = time.time()
    tvec = traj(M, mu, a, p0, e0, x0, charge*charge/4.,T=10.0,err=insp_kwargs['err'])[0]/YRSID_SI
    print("finalt ",tvec[-1],len(tvec))
    toc = time.time()
    print("traj timing",toc - tic)
    
    # Load the grid data
    grid = np.loadtxt("../mathematica_notebooks_fluxes_to_Cpp/grav_Edot_Ldot/data_total.dat")

    def find_closest_value_indices(array, target_value):
        """
        Find the indices where the values in the array are closest to the target value.

        Parameters:
        - array: NumPy array
        - target_value: The value to which you want to find the closest indices

        Returns:
        - indices: Indices where the values are closest to the target value
        """
        # Calculate the absolute differences between array values and the target value
        absolute_diff = np.abs(array - target_value)

        # Find the index with the minimum absolute difference
        closest_index = np.argmin(absolute_diff)

        return closest_index

    # Set the parameters for the trajectory

    # Find the closest value indices in the grid
    ind = find_closest_value_indices(grid[:,0], x0*a)
    mask = (grid[:,0] == grid[ind,0])
    print("max", np.max(grid[:,1][mask]))
    # breakpoint()
    
    # name of the folder to store the plots
    folder = "./results/"
    
    if bool(args['zerolike']):
        folder + "zerolike_"
    
    if vacuum:
        fp = folder + args["outname"] + f"_noise{args['noise']}_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}_vacuum.h5"
    else:
        fp = folder + args["outname"] + f"_noise{args['noise']}_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}.h5"

    tvec, p_tmp, e_tmp, x_tmp, Phi_phi_tmp, Phi_theta_tmp, Phi_r_tmp = traj(M, mu, a, p0, e0, x0, charge*charge/4,T=10.0,err=insp_kwargs['err'],use_rk4=insp_kwargs['use_rk4'])
    print("len", len(tvec))
    fig, axes = plt.subplots(2, 3)
    plt.subplots_adjust(wspace=0.3)
    fig.set_size_inches(14, 8)
    axes = axes.ravel()
    ylabels = [r'$e$', r'$p$', r'$e$', r'$\Phi_\phi$', r'$\Phi_r$', r'Flux']
    xlabels = [r'$p$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$']
    ys = [e_tmp, p_tmp, e_tmp, Phi_phi_tmp, Phi_r_tmp]
    xs = [p_tmp, tvec, tvec, tvec, tvec]
    for i, (ax, x, y, xlab, ylab) in enumerate(zip(axes, xs, ys, xlabels, ylabels)):
        ax.plot(x, y)
        ax.set_xlabel(xlab, fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
    plt.savefig(fp[:-3] + "_trajectory.pdf")
    # print number of cycles
    freq_sep = get_fundamental_frequencies(a,p_tmp[-1], e_tmp[-1],1.0)[0]/(2*np.pi*MTSUN_SI*M)
    print("number of cycles", Phi_phi_tmp[-1]/(2*np.pi))
    print("frequency separatrix", freq_sep)

    emri_injection_params = np.array([
        M,  
        mu, 
        a,
        p0, 
        e0, 
        x0, 
        dist, 
        qS, 
        phiS, 
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0,
        Lambda
    ])

    waveform_kwargs = {
        "T": Tobs,
        "dt": dt,
        "mich": True
    }

    
    run_emri_pe(
        emri_injection_params, 
        Tobs,
        dt,
        fp,
        ntemps,
        nwalkers,
        args['nsteps'],
        emri_kwargs=waveform_kwargs,
        vacuum=vacuum,
        source_SNR=source_SNR,
        zero_like=bool(args['zerolike']),
        noise=args['noise']
    )
