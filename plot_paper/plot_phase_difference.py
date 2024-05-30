#python -m unittest few/tests/test_traj.py 
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.constants import golden

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_fundamental_frequencies
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *

# Set the style of the plot
# import matplotlib.style as style
# style.use('tableau-colorblind10')

# Set the matplotlib parameters
inv_golden = 1. / golden
px = 2*0.0132

plt.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": 'pdflatex',
    "pgf.rcfonts": False,
    "font.family": "serif",
    "figure.figsize": [246.0*px, inv_golden * 246.0*px],
    'legend.fontsize': 12,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.title_fontsize' : 12,
})
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
                
            
    t_d, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge**2 / 4., T=4.0, dt=10.0, err=err, use_rk4=use_rk4)
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
                
            
    t_d, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge**2 / 4, T=4.0, dt=10.0, err=err, use_rk4=use_rk4)
    omPhi, omTh, omR = get_fundamental_frequencies(a,p,e,x)
    interp_d = CubicSplineInterpolant(t_d, np.vstack((Phi_phi,omPhi)) )
    print(len(t_d),err)
    new_t = t if t[-1]<t_d[-1] else t_d
    new_t = new_t[new_t>3600*24]
    diff = np.abs(interp(new_t) - interp_d(new_t))
    return np.vstack((new_t[None,:], diff))


charge_vec=10**np.linspace(-10,-3,num=10)
# err_vec = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
err_vec = [ 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]#, 1e-12, 1e-13,]
simbols = ['-o',  '-x',  '-^',  '-d',  '-*', '-+']
colors = plt.cm.tab10.colors

plt.figure()
# plt.title('Phase difference before plunge')
for err, simb, color in zip(err_vec, simbols, colors):
    deph = [get_t_dphi_dom(err, ch)[1] for ch in charge_vec]
    Npoints = np.min([len(el) for el in deph ])
    deph = np.asarray([el[-1] for el in deph ])
    
    plt.loglog(charge_vec, np.abs(deph), simb, label=rf'error=$10^{{{int(np.log10(err))}}}$, N={Npoints}')
plt.loglog(charge_vec, charge_vec**2 * deph[-1]/charge_vec[-1]**2 , 'k--', label=rf'$\propto d^2$')
plt.legend(ncol=2)
plt.xlabel(r'Scalar charge $d$', fontsize=20)
plt.ylabel(r'Phase difference $\Delta \Phi_\phi$', fontsize=20)
plt.ylim(1e-9,1.0)
plt.tight_layout()
plt.savefig('./figures/phase_difference.pdf')

# check the cycles as a function of the mass ratio
# # create eta_vec log10 spaced 
# eta_vec = 10**np.linspace(-6,-3,num=10)
# plt.figure()
# # create a for loop over eta (mass ratio)
# for eta in eta_vec:
#     t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, M*eta, a, p0, e0, x0, 0.0, T=100.0, dt=10.0, err=1e-10, use_rk4=use_rk4)
#     plt.loglog(1/eta,Phi_phi[-1]/np.pi/2,'o')
# plt.loglog(1/eta_vec,eta_vec[-1]/eta_vec * Phi_phi[-1]/np.pi/2,'--')
# plt.ylabel(r'$\Phi_\varphi/(2 \pi)$',fontsize=20)
# plt.xlabel(r'$1/\eta$',fontsize=20)
# plt.tight_layout()
# plt.show()
