# clean up this code and comment
#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# python search.py -delta 1e-3 -Tobs 2 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -dev 6 -nwalkers 8 -ntemps 1 -nsteps 10 -outname yo  -hdf_data search_results/test_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_delta0.1_SNR50.0_T0.1_seed2601_nw8_nt1.h5 
# nohup python search.py -delta 1e-1 -Tobs 0.1 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -dev 5 -nwalkers 16 -ntemps 1 -nsteps 500000 -outname search2 > out2.out &
# test with zero likelihood
# python search.py -delta 1e-3  -Tobs 0.01 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -dev 6 -nwalkers 16 -ntemps 1 -nsteps 5000 -outname test -zerolike 1
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
parser.add_argument("-delta", "--delta", help="delta prior", required=True, type=float)
parser.add_argument("-dev", "--dev", help="Cuda Device", required=False, type=int, default=0)
parser.add_argument("-dt", "--dt", help="sampling interval delta t", required=False, type=float, default=10.0)
parser.add_argument("-nwalkers", "--nwalkers", help="number of MCMC walkers", required=True, type=int)
parser.add_argument("-ntemps", "--ntemps", help="number of MCMC temperatures", required=True, type=int)
parser.add_argument("-nsteps", "--nsteps", help="number of MCMC iterations", required=False, type=int, default=1000)
parser.add_argument("-SNR", "--SNR", help="SNR", required=False, type=float, default=50.0)
parser.add_argument("-outname", "--outname", help="output name", required=False, type=str, default="MCMC")
parser.add_argument("-zerolike", "--zerolike", help="zero likelihood test", required=False, type=int, default=0)
parser.add_argument("-noise", "--noise", help="noise injection on=1, off=0", required=False, type=float, default=1.0)
parser.add_argument("-hdf_data", "--hdf_data", help="hdf_data", required=False, type=str, default=None)

args = vars(parser.parse_args())
args['hdf_data']
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
from eryn.moves.gaussian import propose_DE
from scipy.signal.windows import tukey
from scipy import signal
# get short fourier transform for cuda
from cupyx.scipy.signal import stft

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
from sklearn.neighbors import KernelDensity
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
from few.utils.utility import get_fundamental_frequencies, get_separatrix
from scipy.optimize import minimize

traj = EMRIInspiral(func=func)

def trace_track_freq(M, a, p0, e0, m=2, n=0):
    omega_phi,omega_theta,omega_r = get_fundamental_frequencies(a, p0, e0, 1.0)
    out = m*omega_phi + n*omega_r
    return out/(M*MTSUN_SI*np.pi*2)

def objective_function(inputs, true_freqs, a):
    p, e = inputs
    if p < 6 + 2*e:
        return np.inf
    else:
        frs = get_fundamental_frequencies(a, p, e, 1.0)
        return (frs[0] - true_freqs[0])**2 + (frs[2] - true_freqs[1])**2

def reverse_rootfind(true_freqs, a):
    result = minimize(
        objective_function, 
        x0 = [10., 0.3], 
        bounds=([6.0, 16.],[0.1, 0.45]), 
        args = (true_freqs, a),
        method="Nelder-Mead", 
        options=dict(xatol=1e-8),  # xatol specifies tolerance on p,e
    ) 
    return result.x

def get_p_e_from_freq(M, a, freqs):
    return reverse_rootfind(freqs*(M*MTSUN_SI*np.pi*2), a)

def get_fm_fn_from_p_e(M, a, p, e):
    f_m = trace_track_freq(M, a, p, e, m=1, n=0)
    f_n = trace_track_freq(M, a, p, e, m=0, n=1)
    return f_m,f_n

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
    delta,
    emri_kwargs={},
    log_prior=False,
    source_SNR=50.0,
    intrinsic_only=False,
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

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 15,
       "fill_values": emri_injection_params[np.array([ 5, 12, 14])], # spin and inclination and Phi_theta
       "fill_inds": np.array([ 5, 12, 14]),
    }
    

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

    # get the sampling parameters
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[1] = np.log(emri_injection_params[1])
    
    reparam = False
    if reparam:
        emri_injection_params[3],emri_injection_params[4] = get_fm_fn_from_p_e(M, a, p0, e0)
    
    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    emri_injection_params[7] = np.cos(emri_injection_params[7]) 
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9]) 
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)


    def transf_func(logM, logmu, ahere, fm, fn):
        if reparam:
            p_e = np.asarray([get_p_e_from_freq(np.exp(MM), aa, np.asarray([f1,f2])) for MM,aa,f1,f2 in zip(logM, ahere, fm,fn)])
            return [np.exp(logM), np.exp(logmu), ahere, p_e[:,0], p_e[:,1],]
        else:
            return [np.exp(logM), np.exp(logmu), ahere, fm, fn]

    parameter_transforms = {
        # 0: np.exp,  # M 
        # 1: np.exp,  # mu
        (0,1,2,3,4): transf_func,
        7: np.arccos, # qS
        9: np.arccos,  # qK
    }
    
    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )
    
    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]
    
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

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]
    
    # get AE
    data_channels = wave_gen(*injection_in, **emri_kwargs)
    tic = time.perf_counter()
    [wave_gen(*injection_in, **emri_kwargs) for _ in range(10)]
    toc = time.perf_counter()
    print("timing",(toc-tic)/10, "len vec", len(data_channels[0]))
    
    check_snr = snr([data_channels[0], data_channels[1]],**inner_kw)
    
    print("SNR",check_snr)
    ############################## priors ########################################################
        
    # priors
    if reparam:
        dist3 = uniform_dist(emri_injection_params_in[3]-1e-4, emri_injection_params_in[3]+1e-4)
        dist4 = uniform_dist(emri_injection_params_in[4]-1e-4, emri_injection_params_in[4]+1e-4)
    else:
        dist3 = uniform_dist(p0 - delta, p0 + delta)
        dist4 = uniform_dist(np.max([e0 - delta,0.0]), np.min([e0 + delta, 0.45]))
        
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(emri_injection_params_in[0] - delta, emri_injection_params_in[0] + delta),  # ln M
                1: uniform_dist(emri_injection_params_in[1] - delta, emri_injection_params_in[1] + delta),  # ln mu
                2: uniform_dist(emri_injection_params_in[2] - delta, np.min([emri_injection_params_in[2]+delta, 0.98])),  # a
                3: dist3,
                4: dist4,
                5: powerlaw_dist(0.01,10.0),  # dist in Gpc
                6: uniform_dist(-0.99999, 0.99999),  # qS
                7: uniform_dist(0.0, 2 * np.pi),  # phiS
                8: uniform_dist(-0.99999, 0.99999),  # qK
                9: uniform_dist(0.0, 2 * np.pi),  # phiK
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "emri": {7: 2 * np.pi, 9: 2 * np.pi, 10: 2 * np.pi, 11: 2 * np.pi}
    }
    
    ndim = 12
    # Define the ranges for each parameter
    bounds = np.asarray([[priors["emri"].priors[i][1].min_val, priors["emri"].priors[i][1].max_val]  for i in range(ndim)])
    
    # Number of samples
    num_samples = nwalkers*ntemps

    # Generate Sobol sequence
    import sobol_seq
    from pyDOE2 import lhs


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

    def get_noise_injection(N, dt, sens_fn="lisasens"):
        freqs = xp.fft.fftfreq(N, dt)
        df_full = xp.diff(freqs)[0]
        freqs[0] = freqs[1]
        psd = [get_sensitivity(freqs,sens_fn=sens_fn), get_sensitivity(freqs,sens_fn=sens_fn)]
        # psd = [get_sensitivity_stas(freqs.get(),sens_fn=sens_fn), get_sensitivity_stas(freqs.get(),sens_fn=sens_fn)]
        # psd = [xp.asarray(psd_temp) for psd_temp in psd]
        
        # normalize by the factors:
        # 1/dt because when you take the FFT of the noise in time domain
        # 1/sqrt(4 df) because of the noise is sqrt(S / 4 df)
        noise_to_add = [xp.fft.ifft(xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0]))+ 1j * xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0])) ).real for psd_temp in psd]
        return [noise/(dt*np.sqrt(2*df_full)) * noise_to_add[0], noise/(dt*np.sqrt(2*df_full)) * noise_to_add[1]]

    full_noise = get_noise_injection(len_tot,dt)
    print("check nosie value",full_noise[0][0],full_noise[1][0])
    print("noise check ", inner_product(full_noise,full_noise, **inner_kw)/len(data_channels[0]) )
    print("matched SNR ", inner_product(full_noise[0]+data_channels[0],data_channels[0], **inner_kw)/inner_product(data_channels[0],data_channels[0], **inner_kw)**0.5 ) 
    
    data_stream = [data_channels[0]+full_noise[0][:len(data_channels[0])], data_channels[1]+full_noise[1][:len(data_channels[0])]]
    # ---------------------------------------------------------------------------------
    f_stft, t_stft, Zxx = stft(data_stream[0], fs=1/dt, nperseg=7*86400/dt, window='boxcar')
    stft_dd = xp.asarray([stft(el, fs=1/dt, nperseg=7*86400/dt, window='boxcar')[2] for el in data_stream])
    stft_nn = xp.asarray([stft(el, fs=1/dt, nperseg=7*86400/dt, window='boxcar')[2] for el in full_noise])
    test = 4 * xp.sum( xp.abs(stft_nn[0])**2 / get_sensitivity(f_stft)[None, :, None] /(7*86400*7*86400/dt)) 
    # get inner product from stft
    def TF_inner(params):
        wavehere = wave_gen(*transform_fn.both_transforms(params[None,:])[0], **emri_kwargs)
        stft_wave = xp.asarray([stft(el, fs=1/dt, nperseg=7*86400/dt, window='boxcar')[2] * dt for el in wavehere])
        out = xp.abs( xp.sum(4 * stft_wave * stft_dd.conj() / get_sensitivity(f_stft)[None, :, None], axis=1) )
        
        return xp.sum(out)
    
    breakpoint()
    TF_inner(emri_injection_params_in)

    freqs = xp.fft.fftfreq(len_tot, dt)
    Sn = get_sensitivity(xp.abs(freqs))
    Sn[0] = Sn[1]
    ifft_data = xp.fft.ifft(get_fft(xp.asarray(data_stream),dt)/ Sn ,axis=1) / dt
    d_d = inner_product(data_stream,data_stream, **inner_kw)
    
    check = xp.sqrt(2*xp.real(xp.sum(xp.diff(freqs)[0] * get_fft(data_channels,dt) * get_fft(xp.asarray(data_channels),dt).conj()/ Sn)))
    check2 = xp.sqrt(2*xp.real(xp.sum(xp.asarray(data_channels)*ifft_data*dt)))
    
    def get_matched_SNR(el):
        wavehere = wave_gen(*transform_fn.both_transforms(el[None,:])[0], **emri_kwargs)
        snr_here = inner_product(wavehere,wavehere, **inner_kw)**0.5
        return inner_product(data_stream,wavehere, **inner_kw)/snr_here
    
    def get_new_inner(el, verbose=False):
        inp_par = transform_fn.both_transforms(el[None,:])[0]
        wavehere = wave_gen(*inp_par, **emri_kwargs)
        snr_here = inner_product(wavehere,wavehere, **inner_kw)**0.5
        new_inner = xp.abs( xp.sum(2. * xp.fft.fft(xp.asarray(wavehere) * ifft_data, axis=1) * dt, axis=0) )
        fphi,fr = get_fm_fn_from_p_e(inp_par[0],inp_par[2],inp_par[3],inp_par[4])
        maxf = 2*fphi
        new_inner = new_inner[xp.abs(freqs)<maxf]
        # d_m_h = inner_product([data_stream[0]-wavehere[0],data_stream[1]-wavehere[1]],[data_stream[0]-wavehere[0],data_stream[1]-wavehere[1]], **inner_kw)
        # mSNR = inner_product(data_stream,wavehere, **inner_kw)/snr_here
        if verbose:
            best_ = xp.argsort(new_inner/snr_here)[-1]
            
            print("delta_f",freqs[xp.abs(freqs)<maxf][best_], xp.sort(new_inner/snr_here)[-1])
            # print("factor", mSNR)
            plt.figure(); plt.loglog(freqs[xp.abs(freqs)<maxf].get(), np.abs(new_inner.get())); plt.xlim(-maxf,maxf); plt.savefig('newinner')
            # get_p_e_from_freq(inp_par[0],inp_par[2],np.asarray([fphi - freqs[xp.argmax(new_inner/snr_here)],fr]))
            
            wavehere = wave_gen(*inp_par, **emri_kwargs)
            argexp = 2*np.pi*1j*freqs[xp.abs(freqs)<maxf][best_]*xp.arange(len(wavehere[0]))*dt
            plt.figure(); 
            to_plot = get_fft(xp.asarray(wavehere)*xp.exp(-argexp) ,dt)[0]
            plt.loglog(freqs.get(), np.abs(to_plot.get()),label='shifted')
            to_plot = get_fft(xp.asarray(wavehere),dt)[0]
            plt.loglog(freqs.get(), np.abs(to_plot.get()),label='tested')
            to_plot = get_fft(xp.asarray(data_channels),dt)[0]
            plt.loglog(freqs.get(), np.abs(to_plot.get()),'--',label='true')
            plt.legend()
            plt.xlim(1e-4,1e-2)
            plt.savefig('spectrum_freq')
        
        return -0.25 *( d_d - 2 * new_inner.max() + snr_here**2 ).get()
    
    print("check maximization over frequency", get_new_inner(emri_injection_params_in))
    yo = emri_injection_params_in.copy()
    yo[0] += 1e-4
    print("check maximization over frequency", get_new_inner(yo))
    
    nchannels = 2
    
    like.inject_signal(
        data_stream=data_stream,
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "lisasens"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

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
        plt.plot(np.arange(len(data_channels[0].get()))*dt,  (data_channels[0]).get())
        plt.savefig(fp[:-3] + "injection_td.pdf")

    
    else:
        plt.figure()
        plt.plot(data_channels[0])
        plt.show()
        plt.savefig(fp[:-3] + "injection.pdf")

    #####################################################################
    # generate starting points
        
    def get_sequence(num_samples,name='sob'):
        if name == 'sob':
            sequence = sobol_seq.i4_sobol_generate(ndim, num_samples)
        else:
            sequence = lhs(ndim, samples=num_samples, criterion='maximin')

        # Rescale the samples to the specified ranges
        for i, (min_val, max_val) in enumerate(bounds):
            sequence[:, i] = min_val + sequence[:, i] * (max_val - min_val)
        return sequence
    
    try:
        if args['hdf_data']:
            file  = HDFBackend(args['hdf_data'])
        else:
            file  = HDFBackend(fp)
        burn = int(file.iteration*0.25)
        thin = 1
        
        # get samples
        toplot = file.get_chain(discard=burn, thin=thin)['emri'][:,0][file.get_inds(discard=burn, thin=thin)['emri'][:,0]] # np.load(fp.split('.h5')[0] + '/samples.npy') # 
        # bounds = np.quantile(toplot,[0.025,0.975],axis=0).T
        cov = np.cov(toplot,rowvar=False) # np.load(fp[:-3] + "_covariance.npy")
        tmp = toplot[:nwalkers*ntemps]
        
        # tmp[0] = emri_injection_params_in.copy()
        
        # create move
        # toplot = np.load(fp[:-3] + "_samples.npy") # 
        # sklearn_gmm = SklearnGaussianMixtureModel(n_components=4)  # You can adjust the number of components as needed
        # sklearn_gmm.fit(toplot)
        # pdc_gmm = ProbDistContainer({(0,1,2,3,4,5,6,7,8,9,10,11,12): sklearn_gmm})
        # move_gmm = DistributionGenerate({"emri":pdc_gmm})
        
        # get samples
        
        # freq = [get_fm_fn_from_p_e(*el) for el in  np.asarray(transf_func(*toplot[:,:5].T))[[0,2,3,4]].T]
        # omphi,omr=np.asarray(freq).T
        # logl = file.get_log_like(discard=burn, thin=thin)[:,0].flatten()
        # plt.figure(); plt.plot(omr-omphi,logl,'.'); plt.savefig('omr-omphi_ll.png')
        # plt.figure(); plt.plot(omr,omphi,'.'); plt.savefig('omr,omphi_ll.png')
        # plt.figure(); plt.plot(toplot[:,4],omr-omphi,'.'); plt.savefig('e_omr-omphi_ll.png')
        print("covariance imported")
    except:
        print("find starting points")
        # precision of 1e-5
        cov = np.load("covariance.npy")/1000 # np.cov(np.load("samples.npy"),rowvar=False) * 2.38**2 / ndim
        if intrinsic_only:
            filtered_matrix = np.delete(cov, [5, 6, 7, 8, 9], axis=0)
            cov = np.delete(filtered_matrix, [5, 6, 7, 8, 9], axis=1)

        tmp = get_sequence(5000) # priors['emri'].rvs(nwalkers*ntemps) #draw_initial_points(emri_injection_params_in, cov, nwalkers*ntemps, intrinsic_only=intrinsic_only)
        
        
        stll = like(tmp, **emri_kwargs)
        tmp = tmp[np.argsort(stll)[-nwalkers*ntemps:]]
        # set one to the true value        
        cov = (np.cov(tmp,rowvar=False) +1e-20*np.eye(ndim))* 2.38**2 / ndim        

    
    print("check matched SNR true",get_matched_SNR(emri_injection_params_in))
    #########################################################################
    # save parameters
    np.save(fp[:-3] + "_injected_pars",emri_injection_params_in)
    
    logp = priors["emri"].logpdf(tmp)
    print("logprior",logp)
    tic = time.time()
    start_like = like(tmp, **emri_kwargs)
    toc = time.time()
    timelike = (toc-tic)/np.prod(tmp.shape)
    start_params = tmp.copy()
    print("start like",start_like, "in ", timelike," seconds")
    start_prior = priors["emri"].logpdf(start_params)
    # true likelihood
    true_like = like(emri_injection_params_in[None,:], **emri_kwargs)
    # true_like = get_new_inner(emri_injection_params_in)
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
    indx_list_extr = []
    indx_list_intr = []
    
    def get_True_vec(ind_in):
        out = update_none.copy()
        out[ind_in] = update_all[ind_in]
        return out
    
    # gibbs variables
    # import itertools
    # stuff = np.arange(5,ndim)
    # list_comb = []
    # for subset in itertools.combinations(stuff, 2):
    #     list_comb.append(subset)
    # [indx_list_extr.append(get_True_vec([el[0],el[1]])) for el in list_comb]
    
    # stuff = np.arange(0,5)
    # list_comb = []
    # for subset in itertools.combinations(stuff, 2):
    #     list_comb.append(subset)
    # [indx_list_intr.append(get_True_vec([el[0],el[1]])) for el in list_comb]
    
    if intrinsic_only:
        sky_periodic = None
    else:
        sky_periodic = [("emri",el[None,:] ) for el in [get_True_vec([6,7]), get_True_vec([8,9])]]
    
    # MCMC moves (move, percentage of draws)
    # indx_list_extr.append(get_True_vec(np.arange(ndim)))
    indx_list_extr.append(get_True_vec([5,6,7,8,9,10,11]))
    # for ii in range(5,ndim):
    #     indx_list_intr.append(get_True_vec([ii]))
    setup_extr = [("emri",el[None,:] ) for el in indx_list_extr]
    
    indx_list_intr.append(get_True_vec([0,1,2,3,4]))
    # for ii in range(5):
    #     indx_list_intr.append(get_True_vec([ii]))
        
    setup_intr = [("emri",el[None,:] ) for el in indx_list_intr]
    # shift values move
    to_shift = [("emri",el[None,:] ) for el in [get_True_vec([10]), get_True_vec([9]), get_True_vec([8])]]
    # prob, index par to shift, value
    shift_value = [0.3, to_shift, np.pi]
    
    # define a proposal to convert into frequencies make the proposal and go back to the space
    def propose_transform(x0):
        samples = x0.copy()
        freq = [get_fm_fn_from_p_e(*el) for el in  np.asarray(transf_func(*x0[:,:5].T))[[0,2,3,4]].T]
        samples[:,[3,4]] = np.asarray(freq)
        backsamp = propose_DE(samples, samples,F=0.1, crossover=True, use_current_state=False)
        p_e = np.asarray([get_p_e_from_freq(np.exp(backsamp[ii,0]),backsamp[ii,2],backsamp[ii,3:5]) for ii in range(backsamp.shape[0])])
        samples = backsamp.copy()
        samples[:,3] = p_e[:,0]
        samples[:,4] = p_e[:,1]
        return samples

    moves = [
        # (GaussianMove({"emri": cov+1e-6*np.eye(ndim)}, mode="DE", factor=10.0, sky_periodic=sky_periodic, indx_list=setup_extr),0.5),
        # (GaussianMove({"emri": cov+1e-6*np.eye(ndim)}, mode="DE", factor=1e4, sky_periodic=sky_periodic, indx_list=setup_intr, prop=propose_transform),0.5),#
        (GaussianMove({"emri": cov+1e-6*np.eye(ndim)}, mode="DE", factor=1e4, sky_periodic=sky_periodic),0.5),#
    ]

    def stopping_fn(i, res, samp):
        current_it = samp.iteration
        discard = int(current_it*0.25)
        check_it = 100
        max_it_update = 1000
        
        # if (current_it // 100) % 2 == 0:
        #     # intrinsic
        #     samp.weights[1]=1.0
        #     samp.weights[0]=0.0
        # else:
        #     # extrinsic
        #     samp.weights[1]=0.0
        #     samp.weights[0]=1.0
            
        # rn = np.random.uniform(0,1)
        # if rn>0.3:
        # else:
        for el in samp.moves:
            el.use_current_state = True
            el.crossover = True
        
        if (current_it>=check_it)and(current_it % check_it == 0):
            # check acceptance and max loglike
            if (current_it // 100) % 2 == 0 :
                print("intrinsic")
            else:
                print("extrinsic")
            print("true - max last loglike", true_like - samp.get_log_like()[-1])
            print("median, min", np.median(true_like - samp.get_log_like()[-1]), np.min(true_like - samp.get_log_like()[-1]))
            print("acceptance", np.mean(samp.acceptance_fraction) )
            print("Temperatures", 1/samp.temperature_control.betas)
            current_acceptance_rate = np.mean(samp.acceptance_fraction)
            ll = samp.get_log_like(discard=discard, thin=1).flatten()
            # # # select the best half
            mask=(ll>np.quantile(ll,0.95))
            # mask=(ll<0.0)
            chain = samp.get_chain(discard=discard)['emri']
            inds = samp.get_inds(discard=discard)['emri']
            to_cov = chain[inds][mask]
            # print("len",to_cov.shape)
            # freq = [get_fm_fn_from_p_e(*el) for el in  np.asarray(transf_func(*to_cov[:,:5].T))[[0,2,3,4]].T]
            # omphi,omr=np.asarray(freq).T
            # plt.figure(); plt.semilogy(np.log10(omphi),np.abs(ll[mask]-true_like),'.'); plt.axvline(np.log10(get_fm_fn_from_p_e(M, a, p0, e0)[0]),color='k'); plt.tight_layout(); plt.savefig('omphi_ll.png')
            # plt.figure(); plt.hist(np.log10(omphi),bins=30); plt.axvline(np.log10(get_fm_fn_from_p_e(M, a, p0, e0)[0]),color='k'); plt.tight_layout(); plt.savefig('hist_omphi.png')
            # plt.figure(); plt.semilogy(to_cov[:,0],np.abs(ll[mask]-true_like),'.'); plt.axvline(np.log(M),color='k');plt.savefig('M_ll.png')

            # else:
            samp.moves[0].chain = None # np.vstack((to_cov,get_sequence(100,name='sob')))
            # samp.moves[1].chain = None # propose_transform(to_cov) # np.vstack((to_cov,get_sequence(100,name='sob')))
            #     print('cov_sob',samp.moves[1].chain.shape)
            # else:
            #     print('none')
            
            # matched SNR
            print("matched SNR ", get_matched_SNR(chain[inds][np.argmax(ll)]) ) 
            print("new matched SNR ", get_new_inner(chain[inds][np.argmax(ll)],verbose=True) ) 
            
            # if np.any((true_like - samp.get_log_like()[-1])<0.0):
            print("best fit",samp.get_chain()["emri"][-1,0][np.argmax(samp.get_log_like()[-1,0])])
            print("true",emri_injection_params_in)
            print("diff fit",samp.get_chain()["emri"][-1,0][np.argmax(samp.get_log_like()[-1,0])]-emri_injection_params_in)
            
            # plot
            if (not zero_like)and(current_it % check_it == 0):
                samples = samp.get_chain(discard=discard, thin=1)["emri"][:, 0].reshape(-1, ndim)
                diff = ll[-200*nwalkers*ntemps:] - true_like
                # diff = get_matched_SNR(chain[inds][-200*nwalkers*ntemps:])
                plt.figure(); plt.plot(diff,'.'); plt.savefig(fp[:-3] + "_diff_ll.png", dpi=150)

                # plt.figure(); plt.plot(omr,omphi,'.'); plt.savefig('omr,omphi_ll.png')
                
                # fig = corner.corner(np.hstack((samples,ll[:,None])),truths=np.append(emri_injection_params_in,true_like)); fig.savefig(fp[:-3] + "_corner.png", dpi=150)
                
            # if (current_it<max_it_update):
            #     # update moves from chain
                # update DE chain
            #     samp.moves[-1].chain = to_cov.copy()
            #     # update cov and svd
            #     samp_cov = np.cov(to_cov,rowvar=False) * 2.38**2 / ndim
            #     svd = np.linalg.svd(samp_cov)
            #     samp.moves[0].all_proposal['emri'].scale = samp_cov
            #     # update Dist
            #     # pdc = samp.moves[1].generate_dist["emri"]
            #     # pdc.priors_in[(0,1,2,3,4,5,6,7,8,9,10,11,12)].fit(samples)
            #     # save cov
            #     np.save(fp[:-3] + "_covariance",samp_cov)
            #     np.save(fp[:-3] + "_samples", samples)   
            # else:
            #     # adapt covariance depending on the acceptance rate
            #     target_acceptance_rate = 0.23  # Target acceptance rate
            #     if current_it<max_it_update+601:
            #         if current_acceptance_rate > 0.4:
            #             samp.moves[0].all_proposal['emri'].scale *= 1.1  # Increase covariance for higher acceptance rate
            #         elif current_acceptance_rate < 0.2:
            #             samp.moves[0].all_proposal['emri'].scale *= 0.9  # Decrease covariance for lower acceptance rate
            #     # samp.weights[0]=0.3
            #     # samp.weights[1]=0.3
            #     # samp.weights[2]=0.3
            
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
        like_val = like(params,**kargs) # np.asarray([get_new_inner(el) for el in params]) # 
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
    charge = 0.0

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
    folder = "./search_results/"
    
    if bool(args['zerolike']):
        folder + "zerolike_"
    
    fp = folder + args["outname"] + f"_rndStart_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_delta{args['delta']}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}.h5"

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
        args['delta'],
        emri_kwargs=waveform_kwargs,
        log_prior=logprior,
        source_SNR=source_SNR,
        intrinsic_only=False,
        zero_like=bool(args['zerolike']),
        noise=args['noise']
    )
