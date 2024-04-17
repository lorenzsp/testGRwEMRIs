#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# python bias.py -Tobs 2 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0 -dev 7 -nwalkers 8 -ntemps 1 -nsteps 10 -outname test
# test with zero likelihood
# python bias.py -Tobs 0.01 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0 -dev 6 -nwalkers 16 -ntemps 1 -nsteps 5000 -outname test -zerolike 1
# select the plunge time
Tplunge = 2.0

import argparse
import os
os.environ["OMP_NUM_THREADS"] = str(2)
os.system("OMP_NUM_THREADS=2")
print("PID:",os.getpid())
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
parser.add_argument("-noise", "--noise", help="noise injection on=1, off=0", required=False, type=float, default=1.0)

args = vars(parser.parse_args())

os.system("CUDA_VISIBLE_DEVICES="+str(args['dev']))
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['dev'])
os.system("echo $CUDA_VISIBLE_DEVICES")
import sys
sys.path.append('/data/lsperi/lisa-on-gpu/')
sys.path.append('/data/lsperi/Eryn/')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import numpy as np

from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend
import corner
from lisatools.utils.utility import AET
from eryn.moves import StretchMove, GaussianMove, DIMEMove

from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *

from lisatools.sensitivity import get_sensitivity

from eryn.utils import TransformContainer
from eryn.moves import DistributionGenerate

from scipy.signal.windows import tukey
from scipy import signal

from fastlisaresponse import ResponseWrapper
from eryn.moves.gaussian import reflect_cosines_array
from scipy.stats import special_ortho_group
from powerlaw import powerlaw_dist, SklearnGaussianMixtureModel
 
from few.waveform import AAKWaveformBase, Pn5AAKWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase
import warnings
warnings.filterwarnings("ignore")

SEED = 2601#1996

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
    log_prior=False,
    source_SNR=50.0,
    zero_like=False,
    noise=1.0
):

    # fix seed for reproducibility and noise injection
    np.random.seed(SEED)
    xp.random.seed(SEED)
    
    # FEW waveform with specified AAK summation and inspiral
    few_gen = GenerateEMRIWaveform(
    AAKWaveformBase, 
    EMRIInspiral,
    AAKSummation,
    # when using intrinsic only , we return a list
    return_list=True,
    inspiral_kwargs=insp_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    frame=None
    )
    
    # sets the proper number of points and what not
    print("use gpus, use logprior", use_gpu, log_prior)
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
   
    resp_gen = ResponseWrapper(
        few_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=True,#"zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )
    
    h_plus = few_gen(*emri_injection_params,**emri_kwargs)[0]

    len_tot = len(h_plus)
    window = xp.asarray(tukey(len_tot,alpha=0.005))
    def wave_gen(*args, **kwargs):
        temp_data_channels = few_gen(*args, **kwargs)
        return [el*window for el in temp_data_channels]

    # parameters
    (
        M,  
        mu,
        a, 
        p0, 
        e0, 
        x0,
        dist, 
        qS, # 7
        phiS,
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0,
        charge
    ) = emri_injection_params
    
    emri_injection_params[-1] = 0.0
    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 15,
       "fill_values": emri_injection_params[np.array([ 5, 12, 14])], # spin and inclination and Phi_theta
       "fill_inds": np.array([ 5, 12, 14]),
    }
    

    # get the sampling parameters
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[1] = np.log(emri_injection_params[1])
    
    # do conversion only when sampling over all parameters

    emri_injection_params[7] = np.cos(emri_injection_params[7]) 
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9]) 
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)


    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    if log_prior:
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
            # 14: np.exp
        }
    
    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )
    
    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]
    injection_in[-1] = charge
    # get AE
    temp_emri_kwargs = emri_kwargs.copy()
    temp_emri_kwargs['T'] = Tplunge
    temp_data_channels = few_gen(*injection_in, **temp_emri_kwargs)
    temp_data_channels = [el*xp.asarray(tukey(len(temp_data_channels[0]),alpha=0.005)) for el in temp_data_channels]

    ############################## distance based on SNR ########################################################
    check_snr = snr([temp_data_channels[0], temp_data_channels[1]],**inner_kw)

    dist_factor = check_snr.get() / source_SNR
    emri_injection_params[6] *= dist_factor
    
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    print("new distance based on SNR", emri_injection_params[6])

    # inject parameters
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]
    
    # get AE
    injection_in[-1] = charge
    data_channels = wave_gen(*injection_in, **emri_kwargs)
    tic = time.perf_counter()
    [wave_gen(*injection_in, **emri_kwargs) for _ in range(10)]
    toc = time.perf_counter()
    print("timing",(toc-tic)/10, "len vec", len(data_channels[0]))
    
    check_snr = snr([data_channels[0], data_channels[1]],**inner_kw)
    
    print("SNR",check_snr)
    ############################## priors ########################################################
    
    delta = 0.01
    
    # priors
    priors = {
        "emri": ProbDistContainer(
            {
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
        ) 
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
        subset=6,  # may need this subset
    )

    def get_noise_injection(N, dt, sens_fn="lisasens",sym=True):
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


    # get injected parameters after transformation
    nocharge_channels = wave_gen(*transform_fn.both_transforms(emri_injection_params_in[None, :])[0], **emri_kwargs)
    
    full_noise = get_noise_injection(len_tot,dt)
    print("check nosie value",full_noise[0][0],full_noise[1][0])
    print("noise check ", inner_product(full_noise,full_noise, **inner_kw)/len(data_channels[0]) )
    print("matched SNR ", inner_product(full_noise[0]+data_channels[0],nocharge_channels[0], **inner_kw)/inner_product(nocharge_channels[0],nocharge_channels[0], **inner_kw)**0.5 ) 
    
    nchannels = 2
    
    like.inject_signal(
        data_stream=[data_channels[0]+full_noise[0][:len(data_channels[0])], data_channels[1]+full_noise[1][:len(data_channels[0])]],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "lisasens"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndim = 12

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
            injection_temp = injection_in.copy()
            injection_temp[-1] = cc
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
        
        # create move
        toplot = np.load(fp[:-3] + "_samples.npy") # )
        print("covariance imported")
    except:
        print("find starting points")
        # precision of 1e-5
        cov = np.load("covariance.npy") # np.cov(np.load("samples.npy"),rowvar=False) * 2.38**2 / ndim

        tmp = draw_initial_points(emri_injection_params_in, cov[:-1,:-1], nwalkers*ntemps)

        # set one to the true value
        tmp[0] = emri_injection_params_in.copy()
        # new_tmp = emri_injection_params_in.copy()
        # for mean_i in range(4):
        #     new_tmp = emri_injection_params_in.copy()
        #     # intial periodic points
        #     new_tmp[8:11] = gmm_means[mean_i]
        #     tmp[mean_i] = new_tmp

        # new_tmp = emri_injection_params_in.copy()
        # new_tmp[9] += np.pi
        # tmp[2] = new_tmp.copy()
        
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
    # stuff = np.asarray([0,1,2,3,4,12])
    # list_comb = []
    # for subset in itertools.combinations(stuff, 2):
    #     list_comb.append(subset)
    # [indx_list.append(get_True_vec([el[0],el[1]])) for el in list_comb]
    
    sky_periodic = [("emri",el[None,:] ) for el in [get_True_vec([6,7]), get_True_vec([8,9])]]
    
    # MCMC moves (move, percentage of draws)
    indx_list.append(get_True_vec(np.arange(ndim)))
    indx_list.append(get_True_vec([5,6,7,8,9,10,11]))
    indx_list.append(get_True_vec([0,1,2,3,4]))
    
    # shift values move
    to_shift = [("emri",el[None,:] ) for el in [get_True_vec([10]), get_True_vec([9]), get_True_vec([8])]]
    # prob, index par to shift, value
    shift_value = [0.3, to_shift, np.pi]
    
    
    moves = [
        (GaussianMove({"emri": cov}, mode="AM", sky_periodic=sky_periodic, shift_value=shift_value),0.5),
        # (move_gmm,1e-5),
        (GaussianMove({"emri": cov}, mode="DE", sky_periodic=sky_periodic),0.5),
    ]

    def stopping_fn(i, res, samp):
        current_it = samp.iteration
        discard = int(current_it*0.25)
        check_it = 2000
        update_it = 200
        max_it_update = 2000

        if current_it<1000: 
            # optimization active
            samp.moves[-1].use_current_state = False
        else:
            # optimization inactive
            samp.moves[-1].use_current_state = True
        
        if (current_it>=check_it)and(current_it % check_it == 0):
            # check acceptance and max loglike
            print("max last loglike", samp.get_log_like()[-1])
            print("acceptance", samp.acceptance_fraction )
            print("Temperatures", 1/samp.temperature_control.betas)
            current_acceptance_rate = np.mean(samp.acceptance_fraction)
            
            # get samples
            samples = sampler.get_chain(discard=discard, thin=1)["emri"][:, 0].reshape(-1, ndim)
            ll = samp.get_log_like(discard=discard, thin=1)[:,0].flatten()
            
            # plot
            # if not zero_like:
            #     fig = corner.corner(np.hstack((samples,ll[:,None])),truths=np.append(emri_injection_params_in,true_like)); fig.savefig(fp[:-3] + "_corner.png", dpi=150)
                
                # if (current_it<max_it_update):
                #     num = 10
                #     # beginning
                #     start_ind = int(N_obs/4)
                #     end_ind = start_ind + 500
                #     plt.figure()
                #     for i in range(num):
                #         h_temp = wave_gen(*transform_fn.both_transforms(samples[i][None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                #         plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),alpha=0.5,label=f'{ll[i]}')
                #     h_temp = wave_gen(*transform_fn.both_transforms(emri_injection_params_in[None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                #     plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),'k',alpha=0.3)
                #     plt.legend()
                #     plt.savefig(fp[:-3] + "_td_1over4.pdf")
                    
                #     # end
                #     start_ind = int(3*N_obs/4)
                #     end_ind = start_ind + 500
                #     plt.figure()
                #     for i in range(num):
                #         h_temp = wave_gen(*transform_fn.both_transforms(samples[i][None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                #         plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),alpha=0.5,label=f'{ll[i]}')
                #     h_temp = wave_gen(*transform_fn.both_transforms(emri_injection_params_in[None, :])[0], **emri_kwargs)[0][start_ind:end_ind]
                #     plt.plot(np.arange(len(h_temp.get()))*dt,  h_temp.get(),'k',alpha=0.3)
                #     plt.legend()
                #     plt.savefig(fp[:-3] + "_td_3over4.pdf")
                    

        if (current_it<max_it_update)and(current_it>=update_it)and(current_it % update_it == 0):
            start_ind = current_it - update_it # this ensures new samples
            end_ind = current_it - 20 # this avoids using the same samples
            # update moves from chain
            chain = samp.get_chain()['emri'][start_ind:end_ind]
            inds = samp.get_inds()['emri'][start_ind:end_ind]
            to_cov = chain[inds]
            # update DE chain
            samp.moves[-1].chain = to_cov.copy()
            # update cov and svd
            samp_cov = np.cov(to_cov,rowvar=False) * 2.38**2 / ndim
            svd = np.linalg.svd(samp_cov)
            samp.moves[0].all_proposal['emri'].svd = svd
            # update Dist
            # pdc = samp.moves[1].generate_dist["emri"]
            # pdc.priors_in[(0,1,2,3,4,5,6,7,8,9,10,11,12)].fit(samples)
            # save cov
            np.save(fp[:-3] + "_covariance",samp_cov)
            np.save(fp[:-3] + "_samples", to_cov)   

        if (i==0)and(current_it>1):
            print("resuming run calculate covariance from chain")            
            samp_cov = np.load(fp[:-3] + "_covariance.npy")
            svd = np.linalg.svd(samp_cov)
            samp.moves[0].all_proposal['emri'].svd = svd
            # get DE chain
            samp.moves[-1].chain = np.load(fp[:-3] + "_samples.npy").copy()
            # chain = samp.get_chain(discard=discard)['emri']
            # inds = samp.get_inds(discard=discard)['emri']
            # to_cov = chain[inds]
            
            # samp.weights[0]=0.0
            # samp.weights[1]=0.0
            # samp.weights[2]=0.5
            # samp.weights[3]=0.5
            
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
        tempering_kwargs={"ntemps": ntemps, "adaptive": True, "Tmax": 30.0},
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
    # sqrt_alpha = 0.4 * np.sqrt( 16 * np.pi**0.5 )
    # d = (sqrt_alpha/(mu*MRSUN_SI/1e3))**2 / 2
    # d = (0.4 * np.sqrt( 16 * np.pi**0.5 )/(mu*MRSUN_SI/1e3))**2 / 2
    charge = args['charge']

    Tobs = args["Tobs"]  # years
    dt = args["dt"]  # seconds
    source_SNR = args["SNR"]

    ntemps = args["ntemps"]
    nwalkers = args["nwalkers"]

    traj = EMRIInspiral(func=func)
    # fix p0 given T
    p0 = get_p_at_t(
        traj,
        Tplunge * 0.999,
        [M, mu, a, e0, x0, 0.0],
        bounds=[get_separatrix(a,e0,x0)+0.1, 30.0],
        traj_kwargs={"dt":dt},
        
    )
    print("new p0 fixed by Tobs, p0=", p0)
    tic = time.time()
    tvec = traj(M, mu, a, p0, e0, x0, charge,T=10.0)[0]/YRSID_SI
    print("finalt ",tvec[-1],len(tvec))
    toc = time.time()
    print("traj timing",toc - tic)

    logprior = False
    folder = "./results/"
    
    if bool(args['zerolike']):
        folder + "zerolike_"
    
    if logprior:
        fp = folder + args["outname"] + f"_bias_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}_logprior.h5"
    else:
        fp = folder + args["outname"] + f"_bias_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}.h5"

    tvec, p_tmp, e_tmp, x_tmp, Phi_phi_tmp, Phi_theta_tmp, Phi_r_tmp = traj(M, mu, a, p0, e0, x0, charge,T=10.0,err=insp_kwargs['err'],use_rk4=insp_kwargs['use_rk4'])
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
        charge
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
        log_prior=logprior,
        source_SNR=source_SNR,
        zero_like=bool(args['zerolike']),
        noise=args['noise']
    )
