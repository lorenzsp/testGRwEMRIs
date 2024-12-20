#python -m unittest few/tests/test_traj.py 
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

T = 100.0
dt = 10.0

insp_kw = {
"T": T,
"dt": dt,
"err": 1e-10,
"DENSE_STEPPING": 0,
"max_init_len": int(1e4),
"use_rk4": True,
"upsample": False,
}

np.random.seed(26011996)

class ModuleTest(unittest.TestCase):
    def test_trajectory_pn5(self):

        # initialize trajectory class
        traj = EMRIInspiral(func="pn5")

        # set initial parameters
        M = 1e5
        mu = 1e1
        np.random.seed(42)
        for i in range(10):
            p0 = np.random.uniform(10.0,15)
            e0 = np.random.uniform(0.0, 1.0)
            a = np.random.uniform(0.0, 1.0)
            Y0 = np.random.uniform(-1.0, 1.0)

            # do not want to be too close to polar
            if np.abs(Y0) < 1e-2:
                Y0 = np.sign(Y0) * 1e-2

            # run trajectory
            #print("start", a, p0, e0, Y0)
            # t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, Y0, **insp_kw)

    def test_trajectory_SchwarzEccFlux(self):
        # initialize trajectory class
        traj = EMRIInspiral(func="SchwarzEccFlux")

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0)

    def test_trajectory_KerrEccentricEquatorial(self):

        err = 1e-10
        
        # initialize trajectory class
        traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")

        # set initial parameters
        M = 1e4
        mu = 1.
        p0 = 12.0
        e0 = 0.1
        a=-0.987
        x0=1.0
        charge = 0.1

        # problematic point r3-rp
        # traj.get_derivative(mu/M, 0.876000 , 8.241867 , 0.272429 , 1.000000, np.asarray([charge]) )
        # print("finalt ",traj(M, mu, 0.876, 8.24187, 0.272429, x0, charge)[0][-1])
        elapsed_time = []
        elapsed_time_rk8 = []
        err = 1e-10
        for i in range(100):
            # beginning E =0.875343   L=2.36959       Q=0
            p0 = np.random.uniform(9.5,30.0)
            e0 = np.random.uniform(0.1, 0.45)
            a = np.random.uniform(-0.99, 0.99)
            print('-----------------------------------------------')
            print(a,p0,e0)
            
            # run trajectory
            tic = time.perf_counter()
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, np.abs(a), p0, e0, np.sign(a)*1.0, charge, use_rk4=True, err=err, T=100.0)
            toc = time.perf_counter()
            print('rk 4 elapsed time', toc-tic, ' number of points', len(t) )
            elapsed_time.append([toc-tic,len(t)])
            tic = time.perf_counter()
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, np.abs(a), p0, e0, np.sign(a)*1.0, charge, use_rk4=False, err=err, T=100.0)
            toc = time.perf_counter()
            print('elapsed time', toc-tic, ' number of points', len(t) )
            elapsed_time_rk8.append([toc-tic,len(t)])
            # if (toc-tic)>1.0:
            #     print("a=",a,"p0=",p0,"e0=",e0)
            #     # import matplotlib.pyplot as plt
            #     # plt.figure(); plt.plot(p,e,'.',alpha=0.4); plt.show()
            # breakpoint()
        
        import matplotlib.pyplot as plt
        lab = [('rk8','^'), ('rk4','o')]
        elapsed_time_rk8, elapsed_time = np.asarray(elapsed_time_rk8),np.asarray(elapsed_time)
        for ii,data in enumerate([elapsed_time_rk8,elapsed_time]):
            # Extracting time and length into separate lists
            duration = [row[0] for row in data]
            length = [row[1] for row in data]

            # Plotting
            plt.loglog(length,duration,lab[ii][1],label=lab[ii][0],alpha=0.8)
        plt.legend()
        plt.xlabel('Number of trajectory points')
        plt.ylabel('Timing of trajectory [s]')
        plt.title(f'Error tolerance {err:.2e}')
        plt.grid(True)
        plt.savefig('timing.png')
        
        # test against Schwarz
        traj_Schw = EMRIInspiral(func="SchwarzEccFlux")
        traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
        a=0.0
        charge = 0.0

        # check against Schwarzchild
        for i in range(100):
            p0 = np.random.uniform(9.0,15)
            e0 = np.random.uniform(0.1, 0.45)
            
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=4.0, max_init_len=int(1e5))
            tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(M, mu, 0.0, p0, e0, 1.0, T=4.0, new_t=t, upsample=True, max_init_len=int(1e5))
            mask = (Phi_rS!=0.0)
            diff =  np.abs(Phi_phi[mask] - Phi_phiS[mask])
            print(p0,e0,diff[-1])
            # self.assertLess(np.max(diff),2.0)