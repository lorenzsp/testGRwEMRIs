#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
import glob
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_kerr_geo_constants_of_motion, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *

# initialize trajectory class
traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
# run trajectory
err = 1e-10
insp_kw = {
    "err": err,
    "DENSE_STEPPING": 0,
    "use_rk4": False,
    }

# get Edot function
trajELQ = EMRIInspiral(func="KerrEccentricEquatorial")
def get_Edot(M, mu, a, p0, e0, x0, charge):
    trajELQ(M, mu, a, p0, e0, x0, charge, T=1e-2, dt=10.0, **insp_kw)
    y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)
    y0 = np.array([y1, y2, y3])
    return trajELQ.inspiral_generator.integrator.get_derivatives(y0)[0] / (mu/M)

def get_delta_Edot(M, mu, a, p0, e0, x0, charge):
    
    Edot_Charge = get_Edot(M, mu, a, p0, e0, x0, charge)
    Edot_ZeroCharge = get_Edot(M, mu, a, p0, e0, x0, 0.0)
    return Edot_Charge - Edot_ZeroCharge

get_delta_Edot(1e6,10,0.9,10.0,0.3,1.0,0.0)

np.random.seed(32)
import matplotlib.pyplot as plt
import time, os
print(os.getpid())


grid = np.loadtxt("../mathematica_notebooks_fluxes_to_Cpp/final_grid/data_total.dat")

# diff = np.abs(grid[:,2] - get_separatrix(np.abs(grid[:,0]),grid[:,2]+1e-16,np.sign(grid[:,0])*1.0))
# plt.figure(); plt.semilogy(diff); plt.savefig('diff')

# plot grid points
plt.figure()
plt.plot(grid[:,1], grid[:,2],'x')
plt.semilogx(get_separatrix(np.abs(grid[:,0]),grid[:,2]+1e-16,np.sign(grid[:,0])*1.0),grid[:,2],'.')
plt.xlabel('p')
plt.ylabel('e')
plt.savefig('p_e_grid.png')

# rhs ode
M=1e6
mu=1e1
a = 0.987
evec = np.linspace(0.01, 0.5, num=50)
epsilon = mu/M
p_all, e_all = np.asarray([temp.ravel() for temp in np.meshgrid( np.linspace(get_separatrix(a, 0.4, 1.0)+0.25 , 14.0, num=50), evec )])
# breakpoint()
# out = np.asarray([traj.get_derivative(epsilon, a, np.asarray([pp, ee, 1.0]), np.asarray([0.0]))  for pp,ee in zip(p_all,e_all)])
# pdot = out[:,0]/epsilon 
# edot = out[:,1]/epsilon
# Omega_phi = out[:,3]
# Omega_r = out[:,5]

# plt.figure()
# cb = plt.tricontourf(p_all, e_all, np.log10(np.abs(pdot)))
# plt.colorbar(cb,label=r'$log_{10} (\dot{p}) $')
# plt.xlabel('p')
# plt.ylabel('e')
# plt.tight_layout()
# # plt.savefig('pdot.png')

# plt.figure()
# cb = plt.tricontourf(p_all, e_all, np.log10(np.abs(edot)))
# plt.colorbar(cb,label=r'$log_{10} (\dot{e}) $')
# plt.xlabel('p')
# plt.ylabel('e')
# plt.tight_layout()
# # plt.savefig(f'edot.png')
# # breakpoint()

files = glob.glob('evolution_*.dat')
for filename in files:

    print(filename)
    if filename.split('_')[1] == 'GR':
        charge = 0.0
    else:
        charge = float(filename.split('_')[1].split('d')[1] )**2/4.0

    # # define parameters
    a= float(filename.split('_')[2].split('a')[1])
    print('set charge and spin to',charge,a)
    x0 = np.sign(a) * 1.0
    a = np.abs(a)

    t_S, p_S, e_S, F1, F2, Om1, Om2, PhiphiS, PhirS = np.loadtxt(filename ).T

    p0, e0 = p_S[0], e_S[0]

    # run trajectory
    print("p0,e0",p0,e0)
    tic = time.time()
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge, T=3.0, dt=10.0, **insp_kw)
    toc = time.time()
    # out_deriv = np.asarray([traj.get_rhs_ode(M, mu, a, pp, ee, xx, charge) for pp,ee,xx in zip(p_S, e_S, np.ones_like(p_S)*x0)])
    # print( np.abs(1-out_deriv[:,3]/Om1).max(), np.abs(1-out_deriv[:,5]/Om2).max() )
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

    # plt.plot(grid[:,1], grid[:,2],'x')

    plt.figure()
    plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, charge={charge:.2e}")
    plt.semilogy(p_S, e_S,'-',label=f"S")
    plt.semilogy(p, e,'.',label=f"FEW",alpha=0.4)
    plt.xlabel('p')
    plt.ylabel('e')
    plt.legend()
    plt.tight_layout()
    plt.savefig('p_e_plane_'+filename)
    # plt.show()

    

