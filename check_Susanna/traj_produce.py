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


M=1e6
mu=1e1
T=4.0
for ii in range(5):
    p0, e0 = np.random.uniform(8.0,15.0), np.random.uniform(0.01, 0.5)
    a=np.random.uniform(0.0,0.99)
    # run trajectory
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=T)

    out = np.asarray((t, p, e, x, Phi_phi, Phi_theta, Phi_r)).T

    np.savetxt(f"output_traj_M{M}_mu{mu}_a{a}_p{p0}_e{e0}_charge{charge}_T{T}yrs",out,header="t p e x Phi_phi Phi_theta Phi_r")