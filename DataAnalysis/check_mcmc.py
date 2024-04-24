#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# python check_mcmc.py -Tobs 2 -dt 10.0 -M 1e6 -mu 5.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.005 -dev 6 -nwalkers 8 -ntemps 1 -nsteps 10 -outname yo
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
from scipy.signal.windows import tukey
from scipy import signal

from fastlisaresponse import ResponseWrapper
from eryn.moves.gaussian import reflect_cosines_array
from scipy.stats import special_ortho_group
from powerlaw import powerlaw_dist
 

from few.waveform import AAKWaveformBase, Pn5AAKWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase

SEED = 26011996

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

from scipy.interpolate import CubicSpline
S_git = np.genfromtxt('./LISA_Alloc_Sh.txt')
Sh_X = CubicSpline(S_git[:,0], S_git[:,1])

def get_sensitivity_stas(f, **kwargs):
    """
    Calculate the LISA Sensitivity curve as defined in https://arxiv.org/abs/2108.01167.
    
    arguments:
        f (double scalar or 1D np.ndarray): Frequency array in Hz

    returns:
        1D array or scalar: S(f) with dimensions of seconds.

    """
    return Sh_X(np.abs(f))
    
import warnings

warnings.filterwarnings("ignore")


# if use_gpu is True:
#     xp = np

def draw_initial_points(mu, cov, size, intrinsic_only=False):    
    
    tmp = np.random.multivariate_normal(mu, cov, size=size)
    
    # for ii in range(tmp.shape[0]):
    #     Rot = special_ortho_group.rvs(tmp.shape[1])
    #     tmp[ii] = np.random.multivariate_normal(mu, (Rot.T @ cov @ Rot))
    
    if intrinsic_only:
        for el in [-2,-3]:
            tmp[:,el] = tmp[:,el]%(2*np.pi)
    else:
        # ensure prior
        for el in [10,11]:
            tmp[:,el] = tmp[:,el]%(2*np.pi)
        
        tmp[:,6],tmp[:,7] = reflect_cosines_array(tmp[:,6],tmp[:,7])
        tmp[:,8],tmp[:,9] = reflect_cosines_array(tmp[:,8],tmp[:,9])
        
        # tmp[:,6] = np.cos(np.random.uniform(0.,2*np.pi,size=len(tmp[:,6])))
        # tmp[:,7] = np.random.uniform(0.,2*np.pi,size=len(tmp[:,7]))
        # tmp[:,8] = np.cos(np.random.uniform(0.,2*np.pi,size=len(tmp[:,8])))
        # tmp[:,9] = np.random.uniform(0.,2*np.pi,size=len(tmp[:,9]))
    
    return tmp

def spectrogram(x, window_size=256, step_size=128, fs=1/10):
    # Calculate number of time steps
    n_timesteps = (len(x) - window_size) // step_size + 1
    
    # Initialize spectrogram array
    spectrogram = xp.zeros((window_size // 2 + 1, n_timesteps))
    
    # Compute spectrogram
    for t in range(n_timesteps):
        # Extract windowed segment
        segment = x[t * step_size:t * step_size + window_size]
        
        # Apply window function (Hann window)
        windowed_segment = segment * xp.hanning(window_size)
        
        # Compute FFT
        fft_result = xp.fft.fft(windowed_segment)
        
        # Store magnitude spectrum
        spectrogram[:, t] = xp.abs(fft_result[:window_size // 2 + 1])
    
    return spectrogram.get()  # Transfer data from GPU to CPU

def get_spectrogram(h,dt,name):
    # Compute spectrogram
    spec = spectrogram(h,fs=1/dt)

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(spec), aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    plt.savefig(name)

func = "KerrEccentricEquatorial"
insp_kwargs = {
    "err": 1e-12,
    "DENSE_STEPPING": 0,
    # "max_init_len": int(1e4),
    "use_rk4": True,
    "func": func,
    }

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

def pad_to_next_power_of_2(arr):
    original_length = len(arr)
    next_power_of_2 = int(2 ** xp.ceil(np.log2(original_length)))

    # Calculate the amount of padding needed
    pad_length = next_power_of_2 - original_length

    # Pad the array with zeros
    padded_arr = xp.pad(arr, (0, pad_length), mode='constant')

    return padded_arr

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def get_plot_sky_location(qK,phiK,qS,phiS,name=None):
    # draw the SSB frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->')

    a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='k')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='k')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='k',label='SSB')
    ax.add_artist(a)

    ax.text(1.1, 0, 0, r'$x$')
    ax.text(0, 1.1, 0, r'$y$')
    ax.text(0, 0, 1.1, r'$z$')

    # sky direction
    th, ph, lab = qS, phiS, 'Sky location'
    x_ = np.sin(th) * np.cos(ph)
    y_ = np.sin(th) * np.sin(ph)
    z_ = np.cos(th)
    a = Arrow3D([0, x_], [0, y_], [0, z_], **arrow_prop_dict, color='blue', label=lab)
    ax.add_artist(a)
    ax.scatter(x_,y_,z_,s=40,label='source')

    # sky spin
    th, ph, lab = qK, phiK, 'MBH Spin'
    x_s = np.sin(th) * np.cos(ph)
    y_s = np.sin(th) * np.sin(ph)
    z_s = np.cos(th)
    a = Arrow3D([x_, x_+x_s], [y_, y_+y_s], [z_, z_+z_s], **arrow_prop_dict, color='red', label=lab)
    ax.add_artist(a)

    ax.view_init(azim=-70, elev=20)
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.5,1.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.legend()
    if name is None:
        plt.savefig('skylocalization.pdf')
    else:
        plt.savefig(name)

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
    intrinsic_only=False,
    zero_like=False
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

    # frequencies
    freqs = xp.fft.rfftfreq(N_obs, dt)

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
    few_gen_list = GenerateEMRIWaveform(
    AAKWaveformBase, 
    EMRIInspiral,
    AAKSummation,
    # when using intrinsic only , we return a list
    return_list=False,
    inspiral_kwargs=insp_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    frame=None
    )
    resp_gen = ResponseWrapper(
        few_gen_list,
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
    
    h_plus_aak = few_gen(*emri_injection_params,**emri_kwargs)[0]
    emri_kwargs['mich'] = False
    h_plus_fresp = resp_gen(*emri_injection_params,**emri_kwargs)[0]
    # create spectrum of the signals using fft
    fft_h_plus = xp.fft.rfft(h_plus_aak)
    fft_h_plus_fresp = xp.fft.rfft(h_plus_fresp)
    # get the frequencies
    # make loglog plot
    plt.figure()
    freq = xp.fft.rfftfreq(len(h_plus_aak), dt)
    plt.loglog(freq.get(), xp.abs(fft_h_plus).get()*dt, label='AAK')
    freq = xp.fft.rfftfreq(len(h_plus_fresp), dt)
    plt.loglog(freq.get(), xp.abs(fft_h_plus_fresp).get()*dt, label='AAK + Fresp')
    for el in ["lisasens", "cornish_lisa_psd", "noisepsd_AE"]:
        PSD_arr = get_sensitivity(freq, sens_fn=el)/ (4 * xp.diff(freq)[0])
        plt.loglog(freq.get(), xp.sqrt(PSD_arr).get(),label=el,alpha=0.5)
    
    plt.legend()
    plt.savefig('spectrum.pdf')
    

    len_tot = len(h_plus_aak)
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
       "fill_values": emri_injection_params[np.array([ 5, 12])], # spin and inclination and Phi_theta
       "fill_inds": np.array([ 5, 12]),
    }
    
    if intrinsic_only:        
        fill_dict = {
       "ndim_full": 15,
       "fill_values": emri_injection_params[np.array([5,6,7,8,9,10,12])], # spin and inclination and Phi_theta
       "fill_inds": np.array([5,6,7,8,9,10,12]),
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
    
    # do conversion only when sampling over all parameters
    if not intrinsic_only:
        emri_injection_params[7] = np.cos(emri_injection_params[7]) 
        emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
        emri_injection_params[9] = np.cos(emri_injection_params[9]) 
        emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)
    
    if log_prior:
        if emri_injection_params[-1] == 0.0:
            emri_injection_params[-1] = np.log(1.001e-7)
        else:
            emri_injection_params[-1] = np.log(emri_injection_params[-1])
        
        prior_charge = uniform_dist(np.log(1e-7) , np.log(0.5))
    else:
        prior_charge = uniform_dist(-0.1, 0.1)

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    if log_prior:
        parameter_transforms = {
            0: np.exp,  # M 
            1: np.exp,  # mu
            7: np.arccos, # qS
            9: np.arccos,  # qK
            14: np.exp
        }
    else:
        parameter_transforms = {
            0: np.exp,  # M 
            1: np.exp,  # mu
            7: np.arccos, # qS
            9: np.arccos,  # qK
            # 14: np.exp
        }
    
    if intrinsic_only:
        del parameter_transforms[7]
        del parameter_transforms[9]

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
    check_snr = snr([temp_data_channels[0], temp_data_channels[1]],
        dt=dt,
        PSD="lisasens",
        PSD_args=(),
        PSD_kwargs={},
        use_gpu=use_gpu,
        )

    dist_factor = check_snr.get() / source_SNR
    emri_injection_params[6] *= dist_factor
    
    if intrinsic_only:
        fill_dict = {
       "ndim_full": 15,
       "fill_values": emri_injection_params[np.array([5,6,7,8,9,10,12])], # spin and inclination and Phi_theta
       "fill_inds": np.array([5,6,7,8,9,10,12]),
        }
        transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
        )
    
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
    
    check_snr = snr([data_channels[0], data_channels[1]],
        dt=dt,
        PSD="lisasens",
        PSD_args=(),
        PSD_kwargs={},
        use_gpu=use_gpu,
        )
    
    print("SNR",check_snr)
    ############################## priors ########################################################
    
    delta = 0.05
    
    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(emri_injection_params_in[0] - delta, emri_injection_params_in[0] + delta),  # ln M
                1: uniform_dist(emri_injection_params_in[1] - delta, emri_injection_params_in[1] + delta),  # ln mu
                2: uniform_dist(emri_injection_params_in[2] - delta, 0.98),  # a
                3: uniform_dist(emri_injection_params_in[3] - delta, emri_injection_params_in[3] + delta),  # p0
                4: uniform_dist(emri_injection_params_in[4] - delta, emri_injection_params_in[4] + delta),  # e0
                5: powerlaw_dist(0.01,10.0),  # dist in Gpc
                6: uniform_dist(-0.99999, 0.99999),  # qS
                7: uniform_dist(0.0, 2 * np.pi),  # phiS
                8: uniform_dist(-0.99999, 0.99999),  # qK
                9: uniform_dist(0.0, 2 * np.pi),  # phiK
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
                12: prior_charge,  # charge
            }
        ) 
    }
    
    if intrinsic_only:
        priors = {
            "emri": ProbDistContainer(
                {
                    0: uniform_dist(emri_injection_params_in[0] - delta, emri_injection_params_in[0] + delta),  # ln M
                    1: uniform_dist(emri_injection_params_in[1] - delta, emri_injection_params_in[1] + delta),  # ln mu
                    2: uniform_dist(emri_injection_params_in[2] - delta, 0.98),  # a
                    3: uniform_dist(emri_injection_params_in[3] - delta, emri_injection_params_in[3] + delta),  # p0
                    4: uniform_dist(emri_injection_params_in[4] - delta, emri_injection_params_in[4] + delta),  # e0
                    5: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                    6: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
                    7: prior_charge,  # charge
                }
            ) 
        }

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "emri": {7: 2 * np.pi, 9: 2 * np.pi, 10: 2 * np.pi, 11: 2 * np.pi}
    }
    
    if intrinsic_only:
        periodic = {"emri": {5: 2 * np.pi, 6: 2 * np.pi}}
    
    ############################## likelihood ########################################################
    # this is a parent likelihood class that manages the parameter transforms
    def get_noise_injection(N, dt,sens_fn="lisasens"):
        freqs = xp.fft.fftfreq(N, dt)
        df_full = xp.diff(freqs)[0]
        freqs[0] = freqs[1]
        psd = [get_sensitivity_stas(freqs.get(),sens_fn=sens_fn), get_sensitivity_stas(freqs.get(),sens_fn=sens_fn)]
        psd = [xp.asarray(psd_temp) for psd_temp in psd]
        # normalize by the factors:
        # 1/dt because when you take the FFT of the noise in time domain
        # 1/sqrt(4 df) because of the noise is sqrt(S / 4 df)
        noise_to_add = [xp.fft.ifft(xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0]))+ 1j * xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0])) ).real for psd_temp in psd]
        return [1/(dt*np.sqrt(2*df_full)) * noise_to_add[0],1/(dt*np.sqrt(2*df_full)) * noise_to_add[1]]

    for _ in range(10):
        full_noise = get_noise_injection(len(data_channels[0]),dt,sens_fn="lisasens")
        # print("check nosie value",full_noise[0][0],full_noise[1][0])
        inner_kw = dict(dt=dt,PSD="lisasens",PSD_args=(),PSD_kwargs={},use_gpu=True)
        freqs = np.fft.rfftfreq(len(data_channels[0]), dt)[1:]
        inner_kw = dict(dt=dt,PSD=xp.asarray(get_sensitivity_stas(freqs)),use_gpu=True)
        print("noise check ", inner_product(full_noise,full_noise, **inner_kw)/len(data_channels[0]) )
    print("matched SNR ", inner_product(full_noise[0]+data_channels[0],data_channels[0], **inner_kw)/inner_product(data_channels[0],data_channels[0], **inner_kw)**0.5 ) 
    
    nchannels = 2
    like_noise = Likelihood(
        wave_gen,
        nchannels,  # channels (A,E)
        dt=dt,
        parameter_transforms={"emri": transform_fn},
        use_gpu=use_gpu,
        vectorized=False,
        transpose_params=False,
        subset=6,  # may need this subset
    )
    
    like_noise.inject_signal(
        data_stream=[data_channels[0]+full_noise[0][:len(data_channels[0])], data_channels[1]+full_noise[1][:len(data_channels[0])]],
        noise_fn=get_sensitivity_stas,
        noise_kwargs=[{"sens_fn": "lisasens"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )
    
    like = Likelihood(
        wave_gen,
        nchannels,  # channels (A,E)
        dt=dt,
        parameter_transforms={"emri": transform_fn},
        use_gpu=use_gpu,
        vectorized=False,
        transpose_params=False,
        subset=6,  # may need this subset
    )
    
    like.inject_signal(
        data_stream=[data_channels[0], data_channels[1]],
        noise_fn=get_sensitivity_stas,
        noise_kwargs=[{"sens_fn": "lisasens"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )
    


    # plt.figure()
    
    # ffth = xp.fft.rfft(data_channels[0]+full_noise[0][:len(data_channels[0])])*dt
    # fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)
    # plt.plot(fft_freq.get(), (xp.abs(ffth)**2).get())
    
    # ffth = xp.fft.rfft(data_channels[0])*dt
    # fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)
    # plt.plot(fft_freq.get(), (xp.abs(ffth)**2).get())
    
    
    # for el in ["lisasens"]:
    #     PSD_arr = get_sensitivity(fft_freq, sens_fn=el)/ (4 * xp.diff(fft_freq)[0])
    #     plt.loglog(fft_freq.get(), PSD_arr.get(),label=el,alpha=0.5)
        
    # plt.loglog(fft_freq.get(), Sh_X(fft_freq.get())/ (4 * xp.diff(fft_freq)[0].get()) ,'--',  label='Stas',alpha=0.5)
    # plt.legend()
    # plt.savefig("injection_fd.pdf")
    
    # plt.figure(); plt.plot((data_channels[0]+full_noise[0][:len(data_channels[0])]).get()); plt.savefig('injection_td.pdf')
    # breakpoint()
    ###################################################
    def get_wave(pars):
        # get injected parameters after transformation
        injection_in = transform_fn.both_transforms(pars[None, :])[0]
        
        # get AE
        data_channels = wave_gen(*injection_in, **emri_kwargs)
        return data_channels

    # like_noise(chain[:1])
    # chain = np.load(repo_name + '/samples.npy')
    # bf = np.median(chain,axis=0)
    
    ind = 11
    
    def get_ll_plot(ind, likefun, var_vec, name='ll_plot.png', ):
        # first
        tmp = np.zeros((len(var_vec),len(emri_injection_params_in)))
        tmp[:,:] = emri_injection_params_in[None,:].copy()
        tmp[:,ind] = var_vec.copy()
        ll_n_at_true = likefun(tmp, **emri_kwargs)        
        plt.figure(); 
        plt.plot(var_vec,ll_n_at_true,'-',label='at true'); 
        plt.axvline(emri_injection_params_in[ind],label='true',color='k'); 
        plt.legend(); 
        plt.savefig(name)
    
    # for vv in [7,9,10,11]:
    #     var_vec = np.linspace(0.0, 2*np.pi, num=20)
    #     get_ll_plot(vv, like_noise, var_vec, name = f'll_{vv}.png', )
    # var_vec = np.cos(np.linspace(0.0, np.pi, num=20))
    # get_ll_plot(8, like_noise, var_vec, name = 'll_thethaK.png', )
    # var_vec = np.cos(np.linspace(0.0, np.pi, num=20))
    # get_ll_plot(6, like_noise, var_vec, name = 'll_thethaS.png', )
    var_vec = np.linspace(emri_injection_params_in[-1]-0.005,emri_injection_params_in[-1]+0.005, num=20)
    get_ll_plot(-1, like, var_vec, name = 'll_d.png', )

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
    phiK = 2*np.pi/4 # azimuthal viewing angle
    qS = 3*np.pi/4 # polar sky angle
    phiS = np.pi # azimuthal viewing angle
    get_plot_sky_location(qK,phiK,qS,phiS)
    dist = 3.0  # distance
    Phi_phi0 = np.pi/2 
    Phi_r0 = np.pi
    Phi_theta0 = Phi_r0 # it must be always be the same
    
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
    folder = "./"
    
    if bool(args['zerolike']):
        folder + "zerolike_"
    
    if logprior:
        fp = folder + args["outname"] + f"_rndStart_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}_logprior.h5"
    else:
        fp = folder + args["outname"] + f"_rndStart_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}.h5"

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
    # plt.savefig(fp[:-3] + "_trajectory.pdf")

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
    
    repo_name = (fp[:-3] + "_trajectory.pdf").split('_trajectory.pdf')[0]

    run_emri_pe(
        emri_injection_params, 
        Tobs,
        dt,
        repo_name,
        ntemps,
        nwalkers,
        args['nsteps'],
        emri_kwargs=waveform_kwargs,
        log_prior=logprior,
        source_SNR=source_SNR,
        intrinsic_only=False,
        zero_like=bool(args['zerolike'])
    )
