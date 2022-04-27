import numpy as np
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
from few.utils.utility import (get_overlap, 
                               get_mismatch, 
                               get_fundamental_frequencies, 
                               get_separatrix, 
                               get_mu_at_t, 
                               get_p_at_t, 
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.summation.interpolatedmodesum import CubicSplineInterpolant
traj = EMRIInspiral(func="KerrCircFlux")

# set initial parameters
M = 1e6
mu = 1e1
a = 0.89
p0 = 10.0
e0 = 0.0
iota0 = 0.0
Y0 = np.cos(iota0)
Phi_phi0 = 0.0
Phi_theta0 = 0.0
Phi_r0 = 0.0

dt = 10.0
T = 3.5




args1=np.array([
    0.0,
])

t_general = np.linspace(0.0, T*365*24*3600, num=500)
# run trajectory
t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, Y0,  Phi_phi0, Phi_theta0, Phi_r0, *args1, T=T, dt=dt, upsample=True, new_t=t_general)
print(t[-1]/(365*24*3600))
breakpoint()
#######################
plt.figure()

for qq in [0.02, 0.05, 0.1, 0.2]:

    args2=np.array([
        qq,
    ])

    t2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2 = traj(M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, *args2, T=T, dt=dt, upsample=True, new_t=t_general)

    tfinal = np.min([t[-1], t2[-1]])

    spl2 = CubicSplineInterpolant(t2, p2)
    spl1 = CubicSplineInterpolant(t, p)

    t_new = np.linspace(0,tfinal)

    difference = np.abs(np.gradient(spl2(t_new),t_new)-np.gradient(spl1(t_new),t_new))
    p_ev = spl1(t_new)
    plt.loglog(p_ev, difference , label=f'q = {args2[0]}' )
    plt.loglog(p_ev, difference[5]*(p_ev/p_ev[5])**(-6.0) , '--', label='slope $\propto p^{-6}$' )

[(np.log(difference[i+1])-np.log(difference[i]))/(np.log(p_ev[i+1])-np.log(p_ev[i])) for i in range(len(t_new)-1)]
plt.ylabel('$pdot-pdot_{GR}$')
plt.xlabel('p')
plt.legend()
plt.savefig('pdot')
