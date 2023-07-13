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

traj = EMRIInspiral(func="KerrEccentricEquatorial")
# run trajectory
err = 1e-10
insp_kw = {
    "T": 1.0,
    "dt": 10.0,
    "err": err,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    # "upsample": True,
    # "fix_T": True

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
for i in range(10):
    p0 = np.random.uniform(10.0,15)
    e0 = np.random.uniform(0.1, 0.5)

    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=4.0)
    tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(M, mu, 0.0, p0, e0, 1.0, T=4.0, new_t=t, upsample=True)
    mask = (Phi_rS!=0.0)
    diff = Phi_phi[mask] - Phi_phiS[mask]
    print(np.max(diff))

    # check against schwarzschild
    # plt.figure(); 
    # plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
    # plt.semilogy(t[mask], diff); plt.xlabel('t [s]'); plt.ylabel('Phase difference'); 
    # plt.show()
    # plt.figure(); plt.semilogy(t[-50:], Phi_phi[-50:],label='a->0'); plt.semilogy(tS[-50:], Phi_phiS[-50:],'--',label='Schwarz'); plt.legend(); plt.xlabel('t [s]'); plt.ylabel('Phase difference'); plt.show()


# for i in range(100):
# p0 = np.random.uniform(10.0,15)
# e0 = np.random.uniform(0.1, 0.5)
# a = np.random.uniform(0.0, 1.0)
t_S, p_S, e_S, F1, F2, Om1, Om2 = np.loadtxt("ev_GR_rp6_ra11.dat").T
a=0.9
p0, e0 = p_S[0], e_S[0]
# run trajectory
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=4.0)

for ii in range(len(p_S)):

    pdot, edot, Ydot, Omega_phi, Omega_theta, Omega_r = traj.get_derivative(epsilon, a, p_S[ii], e_S[ii], 1.0, np.asarray([charge]))
    print(pdot/ F1[ii],edot/ F2[ii], Omega_phi/Om1[ii])

    # breakpoint()
print('run',p0,e0,a)

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.plot(p_S, e_S,'-',label=f"Susanna")
plt.plot(p, e,':',label=f"FEW")
plt.xlabel('p')
plt.ylabel('e')
plt.legend()
plt.show()