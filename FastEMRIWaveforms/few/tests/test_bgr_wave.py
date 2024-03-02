import unittest
import numpy as np
import warnings
import time

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

T = 5.0
dt = 10.0
from few.waveform import AAKWaveformBase, Pn5AAKWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import GenerateEMRIWaveform
np.random.seed(26011996)

class ModuleTest(unittest.TestCase):

    def test_bgr_KerrEccentricEquatorial(self):

        err = 1e-10
        
        # initialize trajectory class
        traj = EMRIInspiral(func="KerrEccentricEquatorial")
        # set initial parameters
        # set parameters
        M = 1e6
        a = 0.0
        mu = 50.0
        p0 = 10.0
        e0 = 0.4
        x0 = 1.0
        qK = np.pi/5  # polar spin angle
        phiK = np.pi/4  # azimuthal viewing angle
        qS = np.pi/2  # polar sky angle
        phiS = np.pi/6  # azimuthal viewing angle
        dist = 3.0  # distance
        Phi_phi0 = np.pi/2
        Phi_theta0 = 0.0
        Phi_r0 = np.pi/2
        

        insp_kwargs = {
            "err": 1e-10,
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e4),
            "func":"KerrEccentricEquatorialAPEX"
            }

        # keyword arguments for summation generator (AAKSummation)
        sum_kwargs = {
            "use_gpu": gpu_available,  # GPU is availabel for this type of summation
            "pad_output": True,
        }

        few_gen = GenerateEMRIWaveform(
            AAKWaveformBase, 
            EMRIInspiral,
            AAKSummation,
            return_list=False,
            inspiral_kwargs=insp_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )
        
        few_gen_Schw = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux",
            return_list=False,
            use_gpu=gpu_available,
        )

        Tobs = 2.0
        dt = 15.0

        charge = 0.0
        h_p_c = few_gen(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, charge, T=Tobs, dt=dt)
        charge = 0.0
        h_p_c_bgr = few_gen_Schw(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, charge, T=Tobs, dt=dt)
        
        traj_Schw = EMRIInspiral(func="SchwarzEccFlux")
        tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(M, mu, a, p0, e0, 1.0)
        mask = (Phi_rS!=0.0)
        traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
        t, p, e, x, Phi_phiAPEX, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=T, new_t=tS, upsample=True, rk4=True)
        traj = EMRIInspiral(func="KerrEccentricEquatorial")
        t, p, e, x, Phi_phiELQ, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=T, new_t=tS, upsample=True, rk4=True)
        
        # import matplotlib.pyplot as plt
        # plt.figure(); plt.plot(p,e,label='AAK'); plt.plot(pS,eS,'--',label='Schw'); plt.legend(); plt.savefig('test_traj')
        # plt.figure(); plt.ylim(1e-3,10.0); plt.semilogy(tS[mask],np.abs(Phi_phiS[mask] - Phi_phiELQ[mask]),label='ELQ');plt.semilogy(tS[mask],np.abs(Phi_phiS[mask] - Phi_phiAPEX[mask]),label='APEX');plt.legend(); plt.savefig('test_phase')
        # t_ax = np.arange(len(h_p_c.get()))
        # for ii in range(0,574669,10000):
        #     plt.figure(); plt.plot(t_ax[ii:ii+500], -h_p_c.get()[ii:ii+500].real,label='AAK'); plt.plot(t_ax[ii:ii+500],h_p_c_bgr.get()[ii:ii+500].real,label='Schw'); plt.legend(); plt.savefig(f'test_real_{ii}')
        # plt.figure(); plt.plot(-h_p_c.get()[-500:].real,label='AAK'); plt.plot(h_p_c_bgr.get()[-500:].real,label='Schw'); plt.legend(); plt.savefig('test_real')
        # plt.figure(); plt.plot(-h_p_c.get()[:500].imag,label='AAK'); plt.plot(h_p_c_bgr.get()[:500].imag,label='Schw'); plt.legend(); plt.savefig('test_imag')
        
        if gpu_available:
            for i in range(100):
                e0 = np.random.uniform(0.01, 0.499)
                a = np.random.uniform(-0.99, 0.99)
                charge = np.random.uniform(0.0,1.0)
                p0 = np.random.uniform(get_separatrix(np.abs(a),e0,np.sign(a)*1.0)+1.0,17.0)
                # print(p0,e0,a)
                h_p_c = few_gen(M, mu, np.abs(a), p0, e0, np.sign(a)*1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, charge, T=Tobs, dt=dt)
