#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
import glob
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
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

M=1e6
mu=1e1
files = glob.glob('evolution_*.dat')
for filename in files:

    print(filename)
    if filename.split('_')[1] == 'GR':
        charge = 0.0
    else:
        charge = float(filename.split('_')[1].split('d')[1] )

    # # define parameters
    a= float(filename.split('_')[2].split('a')[1])
    print('set charge and spin to',charge,a)
    x0 = np.sign(a) * 1.0
    a = np.abs(a)

    t_S, p_S, e_S, F1, F2, Om1, Om2, PhiphiS, PhirS = np.loadtxt(filename ).T

    p0, e0 = p_S[0], e_S[0]

    # run trajectory
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge, T=3.0, dt=10.0, **insp_kw)
    out_deriv = np.asarray([traj.get_rhs_ode(M, mu, a, pp, ee, xx, charge) for pp,ee,xx in zip(p, e, x)])

    print('length', len(t) )
    # interpolate to compare
    interp = CubicSplineInterpolant(t, Phi_phi)
    interp2 = CubicSplineInterpolant(t, Phi_r)
    last_p = 1
    t_S = t_S[:-last_p]
    PhiphiS = PhiphiS[:-last_p]
    PhirS = PhirS[:-last_p]
    p_S = p_S[:-last_p]
    e_S = e_S[:-last_p]

    filename = filename.split('.dat')[0] + '.png'
    plt.figure()
    plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, charge={charge:.2e}")
    plt.loglog(t_S, np.abs(interp(t_S) - PhiphiS),'-',label=f"phi")
    plt.loglog(t_S, np.abs(interp2(t_S) - PhirS),'-',label=f"r")
    plt.ylim(1e-4,30.5)
    plt.xlabel('t [s]')
    plt.ylabel('Phase difference')
    plt.legend()
    plt.grid()
    plt.savefig('Phase_difference_'+filename)

    interp = CubicSplineInterpolant(t, p)
    interp2 = CubicSplineInterpolant(t, e)
    last_p = 5

    # plt.figure()
    # plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, charge={charge:.2e}")
    # plt.semilogy(t_S, np.abs(interp(t_S) - p_S),'-',label=f"p")
    # plt.semilogy(t_S, np.abs(interp2(t_S) - e_S),'-',label=f"e")
    # plt.xlabel('t [s]')
    # plt.ylabel('orbital difference')
    # plt.legend()
    # plt.savefig('p_e_difference_'+filename)

    grid = np.loadtxt("../mathematica_notebooks_fluxes_to_Cpp/grav_Edot_Ldot/grav_data.dat")
    plt.figure()
    plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, charge={charge:.2e}")
    plt.semilogy(p_S, e_S,'-',label=f"S")
    plt.semilogy(p, e,'.',label=f"FEW",alpha=0.4)
    # plt.plot(grid[:,1], grid[:,2],'x')
    plt.xlabel('p')
    plt.ylabel('e')
    plt.legend()
    plt.tight_layout()
    plt.savefig('p_e_plane_'+filename)
    # plt.show()

    

