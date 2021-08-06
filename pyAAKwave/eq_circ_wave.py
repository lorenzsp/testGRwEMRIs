"""
Python 5pn trajectory developed by Lorenzo Speri, Priti Gupta and Michael Katz 30/06/2021
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from constants import *

from few.utils.utility import (get_fundamental_frequencies, get_separatrix, 
                               get_mu_at_t, 
                               get_p_at_t, 
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from energy_flux import *

class PyTrajFluxes_deriv:

    def __init__(self, epsilon, a):
        self.epsilon = epsilon
        self.a = a

    def __call__(self, t, y):
 
        dJdt = np.zeros ( 6 )
        
        dJdt0, dJdt1, dJdt2 = np.real(fluxes(self.a, y[0], y[1], 0.0))
        # epsilon absorbed in the time derivative t = t * epsilon
        # Edot, Ldot, Qdot
        dJdt[0],dJdt[1],dJdt[2] = dJdt0, dJdt1, dJdt2
        p,e,Y = E_roots_numba(self.a, y[0], y[1], 0.0)

        #x_new = Y_to_xI(self.a, p, e, Y)

        # effects of GR
        dJdt[0],dJdt[1],dJdt[2] = dJdt0, dJdt1, 0.0
        # next line to be substituted 
        Om1,Om2,Om3 = get_fundamental_frequencies(self.a, p, 0.00001, 0.9999)
        dJdt[3],dJdt[4],dJdt[5] = Om1/self.epsilon, Om2/self.epsilon, Om3/self.epsilon

        return dJdt

# we need to import an integrator and elliptic integrals
#from scipy.integrate import DOP853
from scipy.integrate import solve_ivp 
# base classes
from few.utils.baseclasses import TrajectoryBase

class PyKerrGenericFluxInspiral(TrajectoryBase):

    # for common interface with *args and **kwargs
    def __init__(self, *args, **kwargs):
        pass
    
    # required by the trajectory base class
    def get_inspiral(self, M, mu, a, p0, e0, Y0, T=1.0, Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, **kwargs):

        # set up quantities and integrator
        E0, L0, Q0 = get_kerr_geo_constants_of_motion(a, p0, e0, Y0)
        y0 = [E0, L0, Q0, Phi_phi0, Phi_r0, Phi_theta0]

        # max T in seconds
        T = T * YRSID_SI / (M * MTSUN_SI)
        # mass ratio
        epsilon = mu/M
        # Slow Time 
        T = T * epsilon

        # need to create the derivative function
        TrajEvol = PyTrajFluxes_deriv(epsilon, a)
        
        # terminate when close to the separatrix
        def term_f(t,y):
            p,e,Y = E_roots_numba(a,y[0],y[1], 0.0)
            #x_new = Y_to_xI(a, p, e, Y)
            #p_sep = get_separatrix(a, e, x_new)
            #print((p - p_sep)/p_sep -.1)
            #print('p,e,Y',p,e,Y)
            return (p - 6.)/6. -.1#(p - p_sep)/p_sep -.1 # this defines the threshold
        
        term_f.terminal = True
        term_f.direction = 0
        # integrate
        r = solve_ivp(TrajEvol, [0.0, T], y0, method='BDF', events=term_f, rtol = 1e-6) # reduce rtol for using 'RK45'
            
        # store solutions
        tsec = GMSUN*M/C_SI**3.0 / epsilon
        t = np.asarray(r.t)*tsec
        En = np.asarray(r.y[0,:]).copy()
        Lz = np.asarray(r.y[1,:]).copy()
        Q = np.asarray(r.y[2,:]).copy()
        Phi_phi = np.asarray(r.y[3,:]).copy()
        Phi_theta = np.asarray(r.y[4,:]).copy()
        Phi_r = np.asarray(r.y[5,:]).copy()
        # orbital elements
        p_e_Y = np.array([E_roots_numba(a, En[i], Lz[i], 0.0) for i in range(len(En))])
        p = np.asarray(p_e_Y[:,0]).copy()
        e = np.ones_like(np.asarray(p_e_Y[:,1]).copy())*e0
        Y = np.ones_like(np.asarray(p_e_Y[:,2]).copy())*Y0
        return (t, p, e, Y, Phi_phi, Phi_theta, Phi_r)


import time
Fluxtraj = PyKerrGenericFluxInspiral()

# set initial parameters
M = 1e6
mu = 14.7
a = 0.5
p0 = 10.0
e0 = 5e-6
Y0 = np.cos(1e-3)
T = 1.0
Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0


# run trajectory to compile it
t, p, e, Y, Phi_phi, Phi_theta, Phi_r = Fluxtraj(M, mu, a, p0, e0, Y0, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

plt.figure()
plt.plot(t,Phi_phi,label='phi')
plt.plot(t,Phi_theta,label='theta')
plt.plot(t,Phi_r,label='r')
plt.legend()
plt.show()
############################################################
from few.utils.baseclasses import Pn5AAK,GPUModuleBase
from few.waveform import Pn5AAKWaveform
from few.summation.aakwave import AAKSummation

# from right to left
# if Pn5 has the same function of GPU, the GPU function will be substituted by the one of the Pn5 
# Abstract Base Class

class ScalarChargePn5AAKWaveform(Pn5AAKWaveform, Pn5AAK, GPUModuleBase):

    def __init__(self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False):

        GPUModuleBase.__init__(self, use_gpu=use_gpu)
        Pn5AAK.__init__(self)

        sum_kwargs = self.adjust_gpu_usage(use_gpu, sum_kwargs)

        # kwargs that are passed to the inspiral call function
        self.inspiral_kwargs = inspiral_kwargs

        # function for generating the inpsiral
        self.inspiral_generator = PyKerrGenericFluxInspiral() # RunKerrGenericPn5Inspiral(**inspiral_kwargs)

        # summation generator
        self.create_waveform = AAKSummation(**sum_kwargs)

wave_gen = ScalarChargePn5AAKWaveform()

dt=10.
mich=False
dist = 1.
start = time.time()
h_migr = wave_gen(M, mu, a, p0, e0, Y0, dist, np.pi/4, np.pi/4, np.pi/4, np.pi/4,  Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T, dt=dt, mich=mich)
print('time', time.time()-start)

hplus = np.real(h_migr)
time = np.arange(0,len(hplus)*dt,dt)
plt.plot(time, hplus)
plt.show()