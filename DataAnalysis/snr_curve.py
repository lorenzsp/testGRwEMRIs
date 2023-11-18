# python snr_curve.py -Tobs 2 -dt 15.0 -dev 0 -M 1e6 -mu 30.0, -e0 0.3 -a 0.95 -x0 1.0
import argparse
import os
os.environ["OMP_NUM_THREADS"] = str(1)
os.system("OMP_NUM_THREADS=1")
print("PID:",os.getpid())
import time
parser = argparse.ArgumentParser(description="MCMC of EMRI source")
parser.add_argument("-Tobs", "--Tobs", help="Observation Time in years", required=True, type=float)
parser.add_argument("-dev", "--dev", help="Cuda Device", required=False, type=int, default=0)
parser.add_argument("-dt", "--dt", help="sampling interval delta t", required=False, type=float, default=10.0)
parser.add_argument("-M", "--M", help="MBH Mass in solar masses", required=True, type=float)
parser.add_argument("-mu", "--mu", help="Compact Object Mass in solar masses", required=True, type=float)
parser.add_argument("-a", "--a", help="dimensionless spin", required=True, type=float)
parser.add_argument("-e0", "--e0", help="Eccentricity", required=True, type=float)
parser.add_argument("-x0", "--x0", help="prograde orbits", default=1.0, required=False, type=float)

args = vars(parser.parse_args())

os.system("CUDA_VISIBLE_DEVICES="+str(args['dev']))
os.environ["CUDA_VISIBLE_DEVICES"] = str(args['dev'])
os.system("echo $CUDA_VISIBLE_DEVICES")
import sys
sys.path.append('/data/lsperi/lisa-on-gpu/')
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

from fastlisaresponse import ResponseWrapper

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
    "max_init_len": int(1e5),
    "func":"KerrEccentricEquatorial"
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

few_gen = GenerateEMRIWaveform(
    AAKWaveformBase, 
    EMRIInspiral,
    AAKSummation,
    return_list=False,
    inspiral_kwargs=insp_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    frame="detector"
)


from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.cosmology import Planck13, z_at_value

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

z_at_value(cosmo.luminosity_distance, cosmo.luminosity_distance(4))

def get_redshift(distance):
    return float(z_at_value(cosmo.luminosity_distance, distance * u.Gpc ))

traj = EMRIInspiral(func="KerrEccentricEquatorial")
def get_p0(M, mu, a, e0, x0, Tobs):
    # fix p0 given T
    p0 = get_p_at_t(traj,Tobs * 0.9999,[M, mu, a, e0, x0, 0.0],bounds=[get_separatrix(a,e0,x0)+0.1, 30.0])
    print("new p0 fixed by Tobs, p0=", p0, traj(M, mu, a, p0, e0, x0, T=10.0)[0][-1]/YRSID_SI)
    return p0

# function call
def run_emri_pe(
    emri_injection_params, 
    Tobs,
    dt,
    emri_kwargs={},
):

    # sets the proper number of points and what not
    print("use gpus", use_gpu)
    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    t_arr = xp.arange(N_obs) * dt

    # frequencies
    freqs = xp.fft.rfftfreq(N_obs, dt)


    orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen = "1st generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 40000.0  # throw away on both ends when our orbital information is weird
   
    wave_gen = ResponseWrapper(
        few_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )
    # wave_gen = few_gen

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 15,
       "fill_values": np.array([ emri_injection_params[5], 0.0, 0.0]), # spin and inclination and Phi_theta
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

    # get the right parameters
    # log of large mass
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[1] = np.log(emri_injection_params[1])
    emri_injection_params[7] = np.cos(emri_injection_params[7]) 
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9]) 
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)

    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(1e5), np.log(5e6)),  # ln M
                1: uniform_dist(np.log(1.0), np.log(100.0)),  # ln mu
                2: uniform_dist(0.0, 0.99),  # a
                3: uniform_dist(7.0, 16.0),  # p0
                4: uniform_dist(0.05, 0.45),  # e0
                5: uniform_dist(0.01, 100.0),  # dist in Gpc
                6: uniform_dist(-0.99999, 0.99999),  # qS
                7: uniform_dist(0.0, 2 * np.pi),  # phiS
                8: uniform_dist(-0.99999, 0.99999),  # qK
                9: uniform_dist(0.0, 2 * np.pi),  # phiK
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
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

    # get injected parameters after transformation

    inner_kw = dict(dt=dt,PSD="noisepsd_AE",PSD_args=(),PSD_kwargs={},use_gpu=use_gpu)
    
    def get_snr(inp):
        data_channels = wave_gen(*inp, **emri_kwargs)
        return snr([data_channels[0], data_channels[1]],**inner_kw).get()
        

    def get_snr_avg(logM, logmu, spin, e0, x0, Tobs, avg_n=50):
        inp_here = priors['emri'].rvs(avg_n)
        inp_here[:,0] = logM
        inp_here[:,1] = logmu
        inp_here[:,2] = spin
        inp_here[:,4] = e0
        inp_here[:,3] = get_p0(np.exp(logM), np.exp(logmu), spin, e0, x0, Tobs)
        inp_here[:,5] = x0
        inp_here[:,6] = 1.0
        
        injection_in = transform_fn.both_transforms(inp_here)
        # get AE
        snr_vec = np.asarray([get_snr(el) for el in injection_in])
        return np.median(snr_vec)
    
    logmu = np.log(mu)
    logM = np.log(M)
    snrhere = get_snr_avg(logM, logmu, a, e0, x0, Tobs, avg_n=500)
    d_L = snrhere/20.0
    np.savetxt(f'./horizon_z/M{M}_redshift.txt',np.asarray([logM, logmu, a, e0, x0, Tobs, d_L]))

    # plt.figure()
    # plt.title(f"spin={a}, e0={e0}, mu={mu}", fontsize=15)
    # plt.semilogx(np.exp(logMvec), z_vec)
    # plt.ylabel('$z$',fontsize=15)
    # plt.xlabel('Mass [M$_\odot$]',fontsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('horizon_distance')
    return

if __name__ == "__main__":
    # set parameters
    M = args["M"]  # 1e6
    a = args["a"]
    mu = args["mu"]  # 10.0
    p0 = 12.0
    e0 = args["e0"]  # 0.35
    x0 = args["x0"]  # will be ignored in Schwarzschild waveform
    qK = np.pi/12  # polar spin angle
    phiK = np.pi  # azimuthal viewing angle
    qS = np.pi/3 # polar sky angle
    phiS = 3*np.pi/4 # azimuthal viewing angle
    # get_plot_sky_location(qK,phiK,qS,phiS)
    dist = 1.0  # distance
    Phi_phi0 = np.pi/2
    Phi_theta0 = 0.0
    Phi_r0 = np.pi/2
    charge = 0.0

    Tobs = args["Tobs"]  # years
    dt = args["dt"]  # seconds

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
        "mich": False,
    }

    run_emri_pe(
        emri_injection_params, 
        Tobs,
        dt,
        emri_kwargs=waveform_kwargs,
    )
