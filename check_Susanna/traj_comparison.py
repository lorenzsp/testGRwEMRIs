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
err = 1e-12
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

filename = "evolution_d01_a09_rp6_ra11"#"evolution_GR_a01234_rp6_ra11"
# Susanna trajectory
charge = 0.1
t_S, p_S, e_S, F1, F2, Om1, Om2, PhiphiS, PhirS = np.loadtxt(filename + ".dat").T

a=0.9
M=1e6
mu=1e1
p0, e0 = p_S[0], e_S[0]
# run trajectory
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, -1.0, charge, T=4.0)

print('run',p0,e0,a)

interp = CubicSplineInterpolant(t, Phi_phi)
interp2 = CubicSplineInterpolant(t, Phi_r)
last_p = 5
t_S = t_S[:-last_p]
PhiphiS = PhiphiS[:-last_p]
PhirS = PhirS[:-last_p]
p_S = p_S[:-last_p]
e_S = e_S[:-last_p]

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.semilogy(t_S, np.abs(interp(t_S) - PhiphiS),'-',label=f"phi")
plt.semilogy(t_S, np.abs(interp2(t_S) - PhirS),'-',label=f"r")
plt.xlabel('t [s]')
plt.ylabel('Phase difference')
plt.legend()
plt.savefig('Phase_difference_'+filename)

interp = CubicSplineInterpolant(t, p)
interp2 = CubicSplineInterpolant(t, e)
last_p = 5

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.semilogy(t_S, np.abs(interp(t_S) - p_S),'-',label=f"p")
plt.semilogy(t_S, np.abs(interp2(t_S) - e_S),'-',label=f"e")
plt.xlabel('t [s]')
plt.ylabel('orbital difference')
plt.legend()
plt.savefig('p_e_difference_'+filename)

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={charge:.2e}")
plt.semilogy(p_S, e_S,'-',label=f"S")
plt.semilogy(p, e,'--',label=f"FEW")
plt.xlabel('p')
plt.ylabel('e')
plt.legend()
plt.savefig('p_e_plane_'+filename)