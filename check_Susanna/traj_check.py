#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time, os
import matplotlib.colors as mcolors

import glob
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *


print(os.getpid())

# initialize trajectory class
traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
use_rk4=False

grid = np.loadtxt("../mathematica_notebooks_fluxes_to_Cpp/final_grid/data_total.dat")

# reference system
M = 1e6
mu = 1e1
a = 0.95
p0 = 8.343242843079224
e0 = 0.4
x0 = 1.0


def get_t_dphi_dom(err,charge):
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, 0.0, T=4.0, dt=10.0, err=1e-10, use_rk4=use_rk4)
    omPhi, omTh, omR = get_fundamental_frequencies(a,p,e,x)
    interp = CubicSplineInterpolant(t, np.vstack((Phi_phi,omPhi)) )
                
            
    t_d, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge, T=4.0, dt=10.0, err=err, use_rk4=use_rk4)
    omPhi, omTh, omR = get_fundamental_frequencies(a,p,e,x)
    interp_d = CubicSplineInterpolant(t_d, np.vstack((Phi_phi,omPhi)) )
    print(len(t_d),err)
    new_t = t if t[-1]<t_d[-1] else t_d
    new_t = new_t[new_t>3600*24]
    diff = np.abs(interp(new_t) - interp_d(new_t))
    return np.vstack((new_t[None,:], diff))

def get_t_dphi_dom_fixed_err(err,charge):
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, 0.0, T=4.0, dt=10.0, err=err, use_rk4=use_rk4)
    omPhi, omTh, omR = get_fundamental_frequencies(a,p,e,x)
    interp = CubicSplineInterpolant(t, np.vstack((Phi_phi,omPhi)) )
                
            
    t_d, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge, T=4.0, dt=10.0, err=err, use_rk4=use_rk4)
    omPhi, omTh, omR = get_fundamental_frequencies(a,p,e,x)
    interp_d = CubicSplineInterpolant(t_d, np.vstack((Phi_phi,omPhi)) )
    print(len(t_d),err)
    new_t = t if t[-1]<t_d[-1] else t_d
    new_t = new_t[new_t>3600*24]
    diff = np.abs(interp(new_t) - interp_d(new_t))
    return np.vstack((new_t[None,:], diff))


charge_vec=10**np.linspace(-10,-2,num=20)
# err_vec = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
err_vec = [1e-9, 1e-10,  0.5e-10]#, 1e-12, 1e-13,]
simbols = ['-o',  '-x',  '-^',  '-d',  '-*', '-+']
colors = plt.cm.tab10.colors

plt.figure()
plt.title('Phase difference before plunge')
for err, simb, color in zip(err_vec, simbols, colors):
    deph = np.asarray([get_t_dphi_dom(err, ch)[1,-1] for ch in charge_vec])
    plt.loglog(charge_vec, np.abs(deph), simb, color=color, label=rf'error=$10^{{{int(np.log10(err))}}}$')
plt.loglog(charge_vec, charge_vec**2 * deph[-1]/charge_vec[-1]**2 , 'k--', label=rf'$\propto d^2$')
plt.legend()
plt.xlabel('Charge')
plt.ylabel('Phase Difference')
plt.show()