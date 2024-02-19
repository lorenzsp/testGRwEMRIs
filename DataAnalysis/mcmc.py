# python mcmc.py -Tobs 2 -dt 10.0 -M 5e5 -mu 5.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0 -dev 3 -nwalkers 8 -ntemps 1 -nsteps 10 -outname yo
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

args = vars(parser.parse_args())

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
 
SEED = 26011996
np.random.seed(SEED)

try:
    import cupy as xp
    # set GPU device
    gpu_available = True
    print("using gpus")

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

import warnings

warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = True

# if use_gpu is True:
#     xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")


insp_kwargs = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "use_rk4": False,

    "func":"KerrEccentricEquatorial",
    }

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

from few.waveform import AAKWaveformBase, Pn5AAKWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix

def pad_to_next_power_of_2(arr):
    original_length = len(arr)
    next_power_of_2 = int(2 ** xp.ceil(np.log2(original_length)))

    # Calculate the amount of padding needed
    pad_length = next_power_of_2 - original_length

    # Pad the array with zeros
    padded_arr = xp.pad(arr, (0, pad_length), mode='constant')

    return padded_arr

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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
):

    few_gen = GenerateEMRIWaveform(
    AAKWaveformBase, 
    EMRIInspiral,
    AAKSummation,
    # when using intrinsic only , we return a list
    return_list=False,
    inspiral_kwargs=insp_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    frame="detector"
    )
    
    # sets the proper number of points and what not
    print("use gpus, use logprior", use_gpu, log_prior)
    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    t_arr = xp.arange(N_obs) * dt

    # frequencies
    freqs = xp.fft.rfftfreq(N_obs, dt)

    orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5" 
    # orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
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
    
    len_tot = len(resp_gen(*emri_injection_params)[0])
    window = xp.asarray(tukey(len_tot,alpha=0.005))
    def wave_gen(*args, **kwargs):
        temp_data_channels = resp_gen(*args, **kwargs)
        return [el*window for el in temp_data_channels]

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 15,
       "fill_values": np.array([ emri_injection_params[5], 0.0]), # spin and inclination and Phi_theta
       "fill_inds": np.array([ 5, 12]),
    }
    
    if intrinsic_only:        
        fill_dict = {
       "ndim_full": 15,
       "fill_values": np.append(emri_injection_params[np.array([5,6,7,8,9,10,12])],0.0), # spin and inclination and Phi_theta
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
        prior_charge = uniform_dist(-0.05, 0.05)

    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

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
    
    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]
    
    # get AE
    data_channels = wave_gen(*injection_in, **emri_kwargs)

    ############################## distance based on SNR ########################################################
    check_snr = snr([data_channels[0], data_channels[1]],
        dt=dt,
        PSD="noisepsd_AE",
        PSD_args=(),
        PSD_kwargs={},
        use_gpu=use_gpu,
        )

    dist_factor = check_snr.get() / source_SNR
    emri_injection_params[6] *= dist_factor
    
    if intrinsic_only:
        fill_dict = {
       "ndim_full": 15,
       "fill_values": np.append(emri_injection_params[np.array([5,6,7,8,9,10,12])],0.0), # spin and inclination and Phi_theta
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
    data_channels = wave_gen(*injection_in, **emri_kwargs)
    toc = time.perf_counter()
    print("timing",toc-tic, "len vec", len(data_channels[0]))
    
    check_snr = snr([data_channels[0], data_channels[1]],
        dt=dt,
        PSD="noisepsd_AE",
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
                5: uniform_dist(0.01,10.0),  # dist in Gpc
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
    
    ############################## plots ########################################################
    if use_gpu:
        get_spectrogram(data_channels[0],dt,fp[:-3] + "_spectrogram.pdf")

        ffth = xp.fft.rfft(data_channels[0])*dt
        fft_freq = xp.fft.rfftfreq(len(data_channels[0]),dt)
        PSD_arr = get_sensitivity(fft_freq, sens_fn="noisepsd_AE")

        plt.figure()
        plt.plot(fft_freq.get(), (xp.abs(ffth)**2).get())
        plt.loglog(fft_freq.get(), PSD_arr.get())
        plt.savefig(fp[:-3] + "injection_fd.pdf")
        # plt.savefig("injection_fd.pdf")

        plt.figure()
        plt.plot(np.arange(len(data_channels[0].get()))*dt,  data_channels[0].get())
        plt.savefig(fp[:-3] + "injection_td.pdf")

        plt.figure()
        for cc in 10**np.linspace(-5,-2):
            injection_temp = injection_in.copy()
            injection_temp[-1] = cc
            data_temp = wave_gen(*injection_temp, **emri_kwargs)
            
            Overlap = inner_product([data_channels[0], data_channels[1]],[data_temp[0], data_temp[1]],
                dt=dt,
                PSD="noisepsd_AE",
                PSD_args=(),
                PSD_kwargs={},
                use_gpu=use_gpu,
                normalize=True
                )
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

    def get_noise_injection(N, dt,sens_fn="noisepsd_AE"):
        freqs = xp.fft.rfftfreq(N+1, dt)
        df_full = xp.diff(freqs)[0]
        freqs[0] = freqs[1]
        psd = [get_sensitivity(freqs,sens_fn=sens_fn), get_sensitivity(freqs,sens_fn=sens_fn)]
        # normalize by the factors:
        # 1/dt because when you take the FFT of the noise in time domain
        # 1/sqrt(4 df) because of the noise is sqrt(S / 4 df)
        noise_to_add = [xp.fft.irfft(xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0]))+ 1j * xp.random.normal(0, psd_temp ** (1 / 2), len(psd[0])) ) for psd_temp in psd]

        return [1/(dt*np.sqrt(4*df_full)) * noise_to_add[0],1/(dt*np.sqrt(4*df_full)) * noise_to_add[1]]

    full_noise = get_noise_injection(len(data_channels[0]),dt,sens_fn="noisepsd_AE")
    print("noise check ", inner_product(full_noise,full_noise, dt=dt,PSD="noisepsd_AE",PSD_args=(),PSD_kwargs={},use_gpu=True)/len(data_channels[0]) )
    nchannels = 2
    like.inject_signal(
        data_stream=[data_channels[0]+full_noise[0][:len(data_channels[0])], data_channels[1]+full_noise[1][:len(data_channels[0])]],
        noise_fn=get_sensitivity,
        noise_kwargs=[{"sens_fn": "noisepsd_AE"} for _ in range(nchannels)],
        noise_args=[[] for _ in range(nchannels)],
    )

    ndim = 13
    if intrinsic_only:
        ndim = 8
    
    #####################################################################
    # generate starting points
    try:
        # file  = HDFBackend(fp)
        # burn = int(file.iteration*0.25)
        # thin = 1
        
        # # get samples
        toplot = np.load(fp.split('.h5')[0] + '/samples.npy')
        # file.get_chain(discard=burn, thin=thin)['emri'][:,0][file.get_inds(discard=burn, thin=thin)['emri'][:,0]]
        cov = np.cov(toplot,rowvar=False) * 2.38**2 / ndim   
        tmp = toplot[:nwalkers*ntemps]
        print("covariance imported")
    except:
        print("find starting points")
        # precision of 1e-5
        cov = np.cov(np.load("samples.npy"),rowvar=False) * 2.38**2 / ndim
        if intrinsic_only:
            filtered_matrix = np.delete(cov, [5, 6, 7, 8, 9], axis=0)
            cov = np.delete(filtered_matrix, [5, 6, 7, 8, 9], axis=1)

        tmp = draw_initial_points(emri_injection_params_in, cov, nwalkers*ntemps, intrinsic_only=intrinsic_only)
        
        # # draw
        # fact = 1.0
        # iter_check = 0
        # max_iter = 50
        # start_like = np.zeros((nwalkers * ntemps))-1e30

        # while np.min(start_like+N_obs) < -2e3:

        #     logp = np.full_like(start_like, -np.inf)
        #     tmp = np.zeros((ntemps * nwalkers, ndim))
        #     fix = np.ones((ntemps * nwalkers), dtype=bool)
        #     while np.any(fix):
        #         tmp[fix] = draw_initial_points(emri_injection_params_in, cov*fact, nwalkers*ntemps, intrinsic_only=intrinsic_only)[fix]
        #         if charge == 0.0:
        #             if logprior:
        #                 tmp[fix,-1] = np.random.uniform(prior_charge.min_val, np.log(1e-5),nwalkers*ntemps)[fix]
        #             # else:
        #             #     tmp[fix,-1] = np.random.uniform(prior_charge.min_val, 1e-5,nwalkers*ntemps)[fix]
        #         logp = priors["emri"].logpdf(tmp)
        #         fix = np.isinf(logp)

        #     start_like = like(tmp, **emri_kwargs)
        
        #     iter_check += 1
        #     fact /= 10.0

        #     print("min starting likelihood",np.min(start_like+N_obs))
        #     print("std",np.std(tmp,axis=0))

        #     if iter_check > max_iter:
        #         print("Unable to find starting parameters.")
        #         break

        # set one to the true value
        tmp[0] = emri_injection_params_in.copy()
        
        cov = (np.cov(tmp,rowvar=False) +1e-20*np.eye(ndim))* 2.38**2 / ndim        
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
    
    if intrinsic_only:
        sky_periodic = None
        gibbs_setup = None
        indx_list.append(get_True_vec([0,2]))
        indx_list.append(get_True_vec([1,7]))
        indx_list.append(get_True_vec([3,4]))
        indx_list.append(get_True_vec([5,6]))
    else:
        indx_list.append(get_True_vec([0,1,2,3,4,12]))
        indx_list.append(get_True_vec([2,12]))
        indx_list.append(get_True_vec([1,12]))
        indx_list.append(get_True_vec([0,12]))
        indx_list.append(get_True_vec([5,6,7]))
        indx_list.append(get_True_vec([8,9]))
        indx_list.append(get_True_vec([10,11]))
        indx_list.append(get_True_vec([5,6,7,8,9,10,11]))

        gibbs_setup = [("emri",el[None,:] ) for el in indx_list]
        sky_periodic = [("emri",el[None,:] ) for el in [get_True_vec([6,7]), get_True_vec([8,9])]]
    
    # MCMC moves (move, percentage of draws)
    moves = [
        (GaussianMove({"emri": cov}, mode="AM", factor=100, sky_periodic=sky_periodic),0.95),
        (GaussianMove({"emri": cov}, mode="AM", factor=100, indx_list=gibbs_setup, sky_periodic=sky_periodic),0.05),
    ]

    def stopping_fn(i, res, samp):
        discard = int(samp.iteration*0.8)
        current_it = samp.iteration
        check_it = 500
        
        if (current_it>check_it)and(current_it % check_it == 0):
            print("max last loglike", samp.get_log_like()[-1])
            print("acceptance", samp.acceptance_fraction )
            print("Temperatures", 1/samp.temperature_control.betas)
            # get samples
            samples = sampler.get_chain(discard=discard, thin=1)["emri"][:, 0].reshape(-1, ndim)
            ll = samp.get_log_like(discard=discard, thin=1)[:,0].flatten()
            
            # plot
            fig = corner.corner(np.hstack((samples,ll[:,None])),truths=np.append(emri_injection_params_in,true_like)); fig.savefig(fp[:-3] + "_corner.png", dpi=150)
            
            if (current_it<2000):
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
                
                # update cov
                chain = samp.get_chain(discard=discard)['emri']
                inds = samp.get_inds(discard=discard)['emri']
                to_cov = chain[inds]
                
                samp_cov = np.cov(to_cov,rowvar=False) * 2.38**2 / ndim
                # prev_cov = samp.moves[1].all_proposal['emri'].scale.copy() 
                # learning_reate = (1e3 - i)/1e3 # it goes from 1 to zero
                svd = np.linalg.svd(samp_cov)
                samp.moves[1].all_proposal['emri'].svd = svd
                samp.moves[0].all_proposal['emri'].svd = svd
        
        # if (i==0)and(current_it>1) we are starting the mcmc again
        if (current_it==2000)or((i==0)and(current_it>1)):
            print("resuming run calculate covariance from chain")
            chain = samp.get_chain(discard=discard)['emri'][:,0]
            inds = samp.get_inds(discard=discard)['emri'][:,0]
            to_cov = chain[inds]
            samp_cov = np.cov(to_cov,rowvar=False) * 2.38**2 / ndim
            svd = np.linalg.svd(samp_cov)
            samp.moves[1].all_proposal['emri'].svd = svd
            samp.moves[0].all_proposal['emri'].svd = svd
        
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
        tempering_kwargs={"ntemps": ntemps, "adaptive": True},
        moves=moves,
        kwargs=emri_kwargs,
        backend=fp,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        stopping_fn=stopping_fn,
        stopping_iterations=1,
        branch_names=["emri"],
    )
    
    if resume:
        log_prior = sampler.compute_log_prior(coords, inds=inds)
        log_like = sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]
        print("initial loglike", log_like)
        start_state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)


    out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=1, burn=0)

    # get samples
    samples = sampler.get_chain(discard=0, thin=1)["emri"][:, 0].reshape(-1, ndim)
    
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
    qK = np.pi/6  # polar spin angle
    phiK = np.pi/5  # azimuthal viewing angle
    qS = np.pi/4 # polar sky angle
    phiS = np.pi/3 # azimuthal viewing angle
    get_plot_sky_location(qK,phiK,qS,phiS)
    dist = 3.0  # distance
    Phi_phi0 = np.pi/3 # changed
    Phi_theta0 = 0.0
    Phi_r0 = 3*np.pi/4  # changed
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

    traj = EMRIInspiral(func="KerrEccentricEquatorial")
    # fix p0 given T
    p0 = get_p_at_t(
        traj,
        Tobs * 0.999,
        [M, mu, a, e0, x0, 0.0],
        bounds=[get_separatrix(a,e0,x0)+0.1, 30.0],
        traj_kwargs={"dt":dt}
    )
    print("new p0 fixed by Tobs, p0=", p0)
    
    logprior = False
    folder = "./results_paper/"
    if logprior:
        fp = folder + args["outname"] + f"_rndStart_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}_logprior.h5"
    else:
        fp = folder + args["outname"] + f"_rndStart_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}.h5"

    tic = time.time()
    tvec, p_tmp, e_tmp, x_tmp, Phi_phi_tmp, Phi_theta_tmp, Phi_r_tmp = traj(M, mu, a, p0, e0, x0, charge,T=10.0)
    print("finalt ",tvec[-1]/YRSID_SI,len(tvec))
    toc = time.time()
    print("traj timing",toc - tic)
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
        "mich": False
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
        intrinsic_only=False,
    )
