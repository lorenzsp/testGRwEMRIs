#!/data/lsperi/miniconda3/envs/bgr_env/bin/python
# python resampling.py -Tobs 2 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -charge 0.0 -dev 3 -nwalkers 26 -ntemps 1 -nsteps 1000 -outname MCMC -vacuum 0


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

from fastlisaresponse import ResponseWrapper
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
        subset=32,  # may need this subset
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

    #####################################################################
    file  = HDFBackend(fp)
    burn = int(file.iteration*0.25)
    thin = 1
    
    # get samples
    toplot = file.get_chain(discard=burn, thin=thin)['emri'][:,0][file.get_inds(discard=burn, thin=thin)['emri'][:,0]]
    ll = file.get_log_like(discard=burn, thin=thin)[:,0].flatten()
    Nresamp = 10000
    weights = np.exp(like(toplot[:Nresamp],**emri_kwargs)-ll[:Nresamp])
    
    CORNER_KWARGS = dict(
        # labels=labels,
        bins=40,
        truths=emri_injection_params_in,
        label_kwargs=dict(fontsize=35),
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=False,
        show_titles=False,
        max_n_ticks=3,
        truth_color='k',
        labelpad=0.3,
    )
    overlaid_corner([toplot, toplot[:Nresamp]],['no resamp', 'resamp'], weights=[np.ones(len(toplot)), weights], name_save=fp[:-3] + '_resampling', corn_kw=CORNER_KWARGS)
    np.save(fp[:-3] + '_resampling_weights.npy', weights)
    np.save(fp[:-3] + '_resampling_toplot.npy', toplot[:Nresamp])
    np.save(fp[:-3] + '_resampling_ll.npy', ll[:Nresamp])
    
    
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
    p0 = get_p_at_t(
        traj,
        Tplunge * 0.999,
        [M, mu, a, e0, x0, 0.0],
        bounds=[get_separatrix(a,e0,x0)+0.1, 30.0],
        traj_kwargs={"dt":dt},
        
    )
    print("new p0 fixed by Tobs, p0=", p0)
    tic = time.time()
    tvec = traj(M, mu, a, p0, e0, x0, charge*charge/4.,T=10.0)[0]/YRSID_SI
    print("finalt ",tvec[-1],len(tvec))
    toc = time.time()
    print("traj timing",toc - tic)

    
    # name of the folder to store the plots
    folder = "./paper_runs/"
    
    if bool(args['zerolike']):
        folder + "zerolike_"
    
    if vacuum:
        fp = folder + args["outname"] + f"_noise{args['noise']}_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}_vacuum.h5"
    else:
        fp = folder + args["outname"] + f"_noise{args['noise']}_M{M:.2}_mu{mu:.2}_a{a:.2}_p{p0:.2}_e{e0:.2}_x{x0:.2}_charge{charge}_SNR{source_SNR}_T{Tobs}_seed{SEED}_nw{nwalkers}_nt{ntemps}.h5"

    print('---------------------------------------')
    print(fp)
    tvec, p_tmp, e_tmp, x_tmp, Phi_phi_tmp, Phi_theta_tmp, Phi_r_tmp = traj(M, mu, a, p0, e0, x0, charge*charge/4,T=10.0,err=insp_kwargs['err'],use_rk4=insp_kwargs['use_rk4'])
    print("len", len(tvec))

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
