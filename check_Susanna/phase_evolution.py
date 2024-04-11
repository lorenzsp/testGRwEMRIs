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
import matplotlib.style as style
style.use('tableau-colorblind10')

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
Phi_phi, Phi_r = 0.0, 0.0

def get_t_dphi_dom(par, T=2.0):
    M, mu, a, p0, e0, x0, Phi_phi, Phi_r, charge = par
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, x0, charge, Phi_phi=Phi_phi,Phi_r=Phi_r, T=T, dt=10.0, err=1e-10, use_rk4=use_rk4)
    omPhi, omTh, omR = get_fundamental_frequencies(a,p,e,x)

    freq = omPhi/(np.pi*MTSUN_SI*M)
    interp_t = CubicSplineInterpolant(t, Phi_phi)
    interp_f = CubicSplineInterpolant(freq, Phi_phi)

    new_t = np.linspace(0.0, T * YRSID_SI, num=100)
    new_f = 10**np.linspace(-3, -1.5, num=100)
    
    phi_t = interp_t(new_t)
    phi_f = interp_f(new_f)
    mask = (new_f<freq.min())+(new_f>freq.max())
    phi_f[mask] = 0.0
    return new_t, new_f, phi_t, phi_f

inp_par = np.asarray([M, mu, a, p0, e0, x0, Phi_phi, Phi_r, 0.0])
new_t, new_f, phi_t, phi_f =  get_t_dphi_dom(inp_par)

plt.figure()
for ch in 10**np.linspace(-7,-2,num=5):
    temp_t, temp_f, temp_phi_t, temp_phi_f =  get_t_dphi_dom(np.asarray([M, mu, a, p0, e0, x0, Phi_phi, Phi_r, ch]))
    plt.semilogy(temp_f, np.abs(temp_phi_f-phi_f))
plt.show()
# breakpoint()