#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *

traj = EMRIInspiral(func="KerrEccentricEquatorial")
# run trajectory
err = 1e-10
insp_kw = {
    "err": err,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    }

np.random.seed(32)
import matplotlib.pyplot as plt
import time, os
print(os.getpid())

# initialize trajectory class
traj = EMRIInspiral(func="KerrEccentricEquatorial")
traj_Schw = EMRIInspiral(func="SchwarzEccFlux")

# set initial parameters
M = 1e6
mu = 1e1
p0 = 12.0
e0 = 0.2
epsilon = mu/M
a=0.00
charge = 0.0

# check against Schwarzchild
plt.figure()
for i in range(100):
    p0 = np.random.uniform(10.0,15)
    e0 = np.random.uniform(0.1, 0.5)
    
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=2.0, max_init_len=int(1e4))
    tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(M, mu, 0.0, p0, e0, 1.0, T=2.0, new_t=t, upsample=True, max_init_len=int(1e4))
    mask = (Phi_rS!=0.0)
    diff = Phi_phi[mask] - Phi_phiS[mask]
    if np.abs(np.max(diff))>0.1:
        print('----------------------------------')
        print(p0,e0)
        print(np.max(diff),len(p),len(pS))
        # check p-e
        plt.plot(p, e,'.',label=f'{len(p)}',alpha=0.3)

        # check pdot
        # edot = [traj.get_derivative(epsilon, a, pp,ee, 1.0, np.asarray([charge]))[1] for pp,ee in zip(p,e)]
        # sep = [get_separatrix(a, ee, 1.0) for pp,ee in zip(p,e)]
        # plt.plot(t[:-1], np.diff(edot),'.',label=f'{len(p)}')
        # plt.plot(t, sep,'.',label=f'{len(p)}')

        # check against schwarzschild
        # plt.figure(); 
        # plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
        # plt.semilogy(t[mask], diff); plt.xlabel('t [s]'); plt.ylabel('Phase difference'); 
        # plt.show()
        # plt.figure(); plt.semilogy(t[-50:], Phi_phi[-50:],'.', label='a->0'); plt.semilogy(tS[-50:], Phi_phiS[-50:],'--',label='Schwarz'); plt.legend(); plt.xlabel('t [s]'); plt.ylabel('Phase difference'); plt.show()
plt.legend()
plt.show()
breakpoint()
# for i in range(100):
# p0 = np.random.uniform(10.0,15)
# e0 = np.random.uniform(0.1, 0.5)
# a = np.random.uniform(0.0, 1.0)
t_S, p_S, e_S, F1, F2, Om1, Om2 = np.loadtxt("ev_GR_rp6_ra11.dat").T
a=0.9
p0, e0 = p_S[0], e_S[0]
# run trajectory
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=1.0)

for ii in range(len(p_S)):
    pdot, edot, Ydot, Omega_phi, Omega_theta, Omega_r = traj.get_derivative(epsilon, a, p_S[ii], e_S[ii], 1.0, np.asarray([charge]))
    print(pdot/ F1[ii],edot/ F2[ii], Omega_phi/Om1[ii])
    traj.get_derivative(epsilon, a, p_S[ii], e_S[ii], 1.0, np.asarray([charge]))
    

    # breakpoint()
print('run',p0,e0,a)

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.plot(t_S, p_S,'-',label=f"Susanna")
plt.plot(t, p,':',label=f"FEW")
plt.xlabel('p')
plt.ylabel('e')
plt.legend()
plt.show()

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.plot(p_S, e_S,'-',label=f"Susanna")
plt.plot(p, e,':',label=f"FEW")
plt.xlabel('p')
plt.ylabel('e')
plt.legend()
plt.show()