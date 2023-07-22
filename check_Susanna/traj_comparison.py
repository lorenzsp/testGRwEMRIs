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

# Susanna trajectory
charge = 0.0
t_S, p_S, e_S, F1, F2, Om1, Om2, PhiphiS, PhirS = np.loadtxt("evolution_GR_a09_rp6_ra11_newmsun.dat").T

a=0.9
M=1e6
mu=1e1
p0, e0 = p_S[0], e_S[0]
# run trajectory
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=1.5)

print('run',p0,e0,a)

interp = CubicSplineInterpolant(t, Phi_phi)
interp2 = CubicSplineInterpolant(t, Phi_r)

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.semilogy(t_S, np.abs(interp(t_S) - PhiphiS),'-',label=f"phi")
plt.semilogy(t_S, np.abs(interp2(t_S) - PhirS),'-',label=f"r")
plt.xlabel('t [s]')
plt.ylabel('Phase difference')
plt.legend()
plt.savefig('Phase_difference')