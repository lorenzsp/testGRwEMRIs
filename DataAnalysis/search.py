# clean up this code and comment
#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# python search.py -delta 1e-1 -Tobs 0.5 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -dev 6 -nwalkers 8 -nsteps 500000 -outname test -SNR 30 -noise 1.0 -window_duration 86400
# nohup python search.py -delta 1e-1 -Tobs 0.5 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -dev 6 -nwalkers 8 -nsteps 500000 -outname search -SNR 30 -noise 1.0 -window_duration 86400 > out.out &
# select the plunge time
Tplunge = 0.5

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
parser.add_argument("-nsteps", "--nsteps", help="number of MCMC iterations", required=False, type=int, default=1000)
parser.add_argument("-SNR", "--SNR", help="SNR", required=False, type=float, default=50.0)
parser.add_argument("-outname", "--outname", help="output name", required=False, type=str, default="MCMC")
parser.add_argument("-noise", "--noise", help="noise injection on=1, off=0", required=False, type=float, default=1.0)
parser.add_argument("-hdf_data", "--hdf_data", help="hdf_data", required=False, type=str, default=None)
parser.add_argument("-window_duration", "--window_duration", help="window_duration", required=False, type=float, default=86400)

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
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.optimize import differential_evolution
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

# from fastlisaresponse import ResponseWrapper
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
def run_emri_search(
    emri_injection_params, 
    Tobs,
    dt,
    fp,
    nwalkers,
    nsteps,
    delta,
    seed,
    duration_window = 86400, # 24 hours
    emri_kwargs={},
    log_prior=False,
    source_SNR=50.0,
    noise=1.0
):

    # fix seed for reproducibility and noise injection
    np.random.seed(seed)
    xp.random.seed(seed)
    
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
    
    h_plus = few_gen(*emri_injection_params,**emri_kwargs)[0]

    len_tot = len(h_plus)
    window = xp.asarray(tukey(len_tot,alpha=0.005))
    
    def wave_gen(*args, **kwargs):
        temp_data_channels = few_gen(*args, **kwargs)
        return [el for el in temp_data_channels]
    
        wave_gen = few_gen
    
    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 15,
       "fill_values": emri_injection_params[np.array([ 5, 6, 12, 14])], # spin and inclination and Phi_theta
       "fill_inds": np.array([ 5, 6, 12, 14]),
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
    temp_data_channels = [el for el in temp_data_channels]

    ############################## distance based on SNR ########################################################
    check_snr = inner_product(temp_data_channels, temp_data_channels, **inner_kw)**0.5

    dist_factor = check_snr.get() / source_SNR
    print("old distance based on SNR", emri_injection_params[6])
    emri_injection_params[6] *= dist_factor    
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    print("new distance based on SNR", emri_injection_params[6])
    fill_dict["fill_values"] = emri_injection_params[np.array([ 5, 6, 12, 14])]

    def get_injected_waveform(emri_injection_params_in, Tobs):
        
        emri_kwargs["T"] = Tobs
        # get injected parameters after transformation
        injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]
        # get AE
        injected_h = wave_gen(*injection_in, **emri_kwargs)
        tic = time.perf_counter()
        [wave_gen(*injection_in, **emri_kwargs) for _ in range(10)]
        toc = time.perf_counter()
        print("timing",(toc-tic)/10, "len vec", len(injected_h[0]))
        print("optimal SNR ", inner_product(injected_h,injected_h, **inner_kw)**0.5 ) 
        return injected_h

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
                5: uniform_dist(-0.99999, 0.99999),  # qS
                6: uniform_dist(0.0, 2 * np.pi),  # phiS
                7: uniform_dist(-0.99999, 0.99999),  # qK
                8: uniform_dist(0.0, 2 * np.pi),  # phiK
                9: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    
    ndim = 11
    # Define the ranges for each parameter
    bounds = np.asarray([[priors["emri"].priors[i][1].min_val, priors["emri"].priors[i][1].max_val]  for i in range(ndim)])
    
    ############################## noise injection ########################################################
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

    injected_h = get_injected_waveform(emri_injection_params_in, Tobs)
    # test noise injection
    full_noise = get_noise_injection(len_tot,dt)
    print("check nosie value",full_noise[0][0],full_noise[1][0])
    print("noise check ", inner_product(full_noise,full_noise, **inner_kw)/len(injected_h[0]) )
    print("optimal SNR ", inner_product(injected_h,injected_h, **inner_kw)**0.5 ) 
    
    data_stream = [injected_h[0]+full_noise[0][:len(injected_h[0])], injected_h[1]+full_noise[1][:len(injected_h[0])]]
    print("matched SNR ", inner_product(data_stream,injected_h, **inner_kw)/inner_product(injected_h,injected_h, **inner_kw)**0.5 ) 
    ############################## filter low freq ########################################################
    from scipy.signal import butter, filtfilt

    # Design a high-pass filter
    def high_pass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Apply the high-pass filter
    cutoff_frequency = 0.5e-3  # Cutoff frequency in Hz
    data_stream = xp.asarray([high_pass_filter(el.get(), cutoff_frequency, 1/dt) for el in data_stream])
    full_noise = xp.asarray([high_pass_filter(el.get(), cutoff_frequency, 1/dt) for el in full_noise])
    injected_h = xp.asarray([high_pass_filter(el.get(), cutoff_frequency, 1/dt) for el in injected_h])
        
    ############################## STFT prep ########################################################
    # define class for STFT inner produce
    class STFTInnerProduct():
        def __init__(self, data_stream, duration_window, dt):
            self.data_stream = data_stream
            self.dt = dt

            # nperseg is the number of samples in each window
            nperseg = int(duration_window/dt)
            # stft kwargs
            self.stf_kw = dict(fs=1/dt, nperseg=nperseg, window=('hann',))#, nfft = int(nperseg*30))#, noverlap=nperseg//2)#, nfft=nperseg*2)# boundary='zeros', padded=True, window='hann', scaling='psd')
            # stft info
            self.f_stft, t_stft, Zxx = stft(data_stream[0], **self.stf_kw)
            self.active_windows = xp.arange(len(t_stft))
            # data
            self.stft_data_stream = xp.asarray([stft(el, **self.stf_kw)[2] for el in data_stream])
            df_stft = self.f_stft[1] - self.f_stft[0]
            self.noise_fact = 1 / get_sensitivity(self.f_stft) / df_stft * 4/3
        
        def TF_inner(self,sig1, sig2):
            # return xp.abs(xp.sum(sig1.conj() * sig2 * self.noise_fact[None,:,None],axis=1)).sum()
            return xp.sum(xp.abs(sig1.conj()) * xp.abs(sig2) * self.noise_fact[None,:,None])
        
        def __call__(self, params):
            wavehere = wave_gen(*transform_fn.both_transforms(params[None,:])[0], **emri_kwargs)
            stft_wave = xp.asarray([stft(el, **self.stf_kw)[2] for el in wavehere])[:,:,self.active_windows]
            den = self.TF_inner(stft_wave, stft_wave)**0.5
            num = self.TF_inner(stft_wave, self.stft_data_stream[:,:,self.active_windows])
            return -(num / den).get()
    
    # fft of the noise in one window to compare to what the stft does
    # ft_nn = xp.asarray([xp.fft.rfft(el[nperseg:2*nperseg]*hann)  for el in full_noise])
    # freq = xp.fft.rfftfreq(nperseg,dt)
    # plt.figure()
    # plt.plot(freq.get(), (xp.abs(ft_nn[0])**2).get())
    # plt.semilogy(f_stft.get(), (xp.abs(stft_nn[0])[:,3]**2).get() , '--')
    # ft_hh = xp.asarray([xp.fft.rfft(el[nperseg:2*nperseg]*hann)  for el in injected_h])
    # plt.plot(freq.get(), (xp.abs(ft_hh[0])**2).get())
    # plt.loglog(f_stft.get(), (xp.abs(stft_hh[0])[:,5]**2).get() , '--')
    # plt.plot(f_stft.get(), (1/noise_fact).get())
    # plt.savefig(fp + '_spectrum.pdf')
    
    
    ############################## Inner Product TF ########################################################
    # frequency mask
    TFinner = STFTInnerProduct(data_stream, duration_window, dt)
    # time the inner product
    true_tf = TFinner(emri_injection_params_in)
    print("matched SNR from STFT", true_tf)
    
    stft_noise = xp.asarray([stft(el, **TFinner.stf_kw)[2] for el in full_noise])
    TFinner.TF_inner(stft_noise, stft_noise)
    # breakpoint()
    
    # create differential evolution search
    result = differential_evolution(TFinner, bounds, maxiter=1, popsize=nwalkers, tol=0.01, mutation=(0.1, 1.99), recombination=0.9, seed=seed, callback=None, disp=True, polish=False, init='latinhypercube', atol=0)
    # result = DifferentialEvolutionSolver(TF_inner_snr, bounds, args = (stft_dd,), maxiter=nsteps, popsize=nwalkers, tol=0.01, mutation=(0.1, 1.99), recombination=0.9, seed=None, callback=None, disp=True, polish=True, init='latinhypercube', atol=0)
    x_best = result.x # priors["emri"].rvs()
    all_active_windows = TFinner.active_windows.copy()
    # print("number of windows", TFinner.active_windows)
    print(result.x - emri_injection_params_in)
    ii=0
    with open(fp + "_best_values.txt", "a") as f:
        f.write(f"iteration, wind_duration, injTFstat, inj_params\n")
        f.write(f"{ii}, {duration_window}, {true_tf}, {emri_injection_params_in.tolist()}\n")
    
    # The code is assigning a slice of the `all_active_windows` list to the
    # `TFinner.active_windows` variable. The slice starts from the beginning of the list and goes
    # up to (but does not include) the index `ii`.
    for ii in range(nsteps):# np.asarray([len(all_active_windows)*0.1, len(all_active_windows)*0.5, len(all_active_windows)*0.9, len(all_active_windows)],dtype=int):
        # TFinner.active_windows = all_active_windows[:ii] 
        # random recomb
        recombination = 0.9#np.random.uniform(0.5,0.9)
        mutation = 10**np.random.uniform(-4.0, np.log10(1.999))
        # randomly choose strategy from 
        strategy = "best1bin" # np.random.choice(['best1bin','best1exp','rand1bin','rand1exp','rand2bin','rand2exp','randtobest1bin','randtobest1exp','currenttobest1bin','currenttobest1exp','best2exp','best2bin'])
        
        print('---------------------------------')
        print("strategy",strategy, "recombination",recombination, "mutation",mutation)  
        init = 'latinhypercube'#result.population
        
        for dwind in [duration_window, duration_window*2, duration_window*3, duration_window*4, duration_window*5, duration_window*6, duration_window*7, duration_window*15]:
            print("duration_window [s]", dwind)
            TFinner = STFTInnerProduct(data_stream, dwind, dt)
            true_tf = TFinner(emri_injection_params_in)
            print("TF true", true_tf)
            result = differential_evolution(TFinner, bounds, x0=x_best, strategy=strategy, maxiter=10, popsize=nwalkers, tol=0.01, mutation=mutation, recombination=recombination, seed=seed, callback=None, disp=True, polish=False, init=init, atol=0)
            
            x_best = result.x
            f_best = result.fun
            
            print(ii, f_best)
            print("diff of ln M, ln mu, p0, e0",(x_best-emri_injection_params_in)[:5])
            with open(fp + "_best_values.txt", "a") as f:
                f.write(f"{ii}, {dwind}, {f_best}, {x_best.tolist()}\n")
    
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
    charge = 0.0

    Tobs = args["Tobs"]  # years
    dt = args["dt"]  # seconds
    source_SNR = args["SNR"]

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
    
    
    fp = folder + args["outname"] + f"_rndStart_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_delta{args['delta']}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_wind{args['window_duration']}"

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
    
    run_emri_search(
        emri_injection_params, 
        Tobs,
        dt,
        fp,
        nwalkers,
        args['nsteps'],
        args['delta'],
        SEED,
        duration_window=args['window_duration'],
        emri_kwargs=waveform_kwargs,
        log_prior=logprior,
        source_SNR=source_SNR,
        noise=args['noise']
    )
