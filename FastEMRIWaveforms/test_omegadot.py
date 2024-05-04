from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_fundamental_frequencies, get_separatrix
import numpy as np
import matplotlib.pyplot as plt
import TPI
traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")

# create a plot to show omegadot as a function of p and e
a = 0.9
ecc = np.linspace(0.001,0.5,100)
semil = np.linspace(get_separatrix(a, 0.5, 1.0)+0.1,8.0,120)


omdot_phi = np.asarray([[traj.get_omegadot(10,10, a, p, e, 1.0, 0.0)[0]/(get_fundamental_frequencies(a, p, e, 1.)[0]**2) for e in ecc] for p in semil]).T
omdot_r = np.asarray([[traj.get_omegadot(10,10, a, p, e, 1.0, 0.0)[1]/(get_fundamental_frequencies(a, p, e, 1.)[2]**2) for e in ecc] for p in semil]).T

plt.figure()
plt.contourf(semil, ecc, np.log10(np.abs(omdot_r)), levels=10)
plt.colorbar(label=r'$\log_{10}|\dot \Omega/\Omega^2|$')
plt.ylabel('eccentricity')
plt.xlabel('semilatus rectum')
plt.title(r'$\dot \Omega_r/\Omega_r ^2$')
plt.savefig('omegaR_dot.png')

plt.figure()
plt.contourf(semil, ecc, np.log10(np.abs(omdot_phi)), levels=10)
plt.colorbar(label=r'$\log_{10}|\dot \Omega/\Omega^2|$')
plt.ylabel('eccentricity')
plt.xlabel('semilatus rectum')
plt.title(r'$\dot \Omega_\phi/\Omega_\phi ^2$')
plt.savefig('omegaPhi_dot.png')

plt.figure()
plt.contourf(semil, ecc, np.log10(np.abs(omdot_r/omdot_phi)), levels=10)
plt.colorbar(label=r'$\log_{10}|\dot \Omega_r/\dot \Omega_\phi|$')
plt.ylabel('eccentricity')
plt.xlabel('semilatus rectum')
plt.savefig('omega_dot_ratio.png')

# M,mu=1e6,1e1
# p0=7.25
# e0=0.3
# generate an EMRI inspiral and check the omegadot evolution
# t, pp, ee, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, 0.0,T=0.3)

# omdot_phi = np.asarray([traj.get_omegadot(M, mu, a, p, e, 1.0, 0.0)[-2] for e,p in zip(ee,pp)])
# om_phi = np.asarray([get_fundamental_frequencies(a, p, e, 1.)[0] for e,p in zip(ecc,semil)])

# plt.figure()
# plt.semilogy(pp, np.abs(omdot_phi))
# plt.savefig('omega_dot_inspiral.png')

# mask = (omdot_phi>10.0)
# new_inp = np.log10(np.abs(omdot_phi))
# new_inp[mask] = 1.0
# plt.figure()
# plt.contourf(semil, ecc, new_inp, levels=10)
# plt.colorbar(label=r'$\log_{10}|\dot \Omega/\Omega^2|$')
# plt.ylabel('eccentricity')
# plt.xlabel('semilatus rectum')
# plt.title(r'$\dot \Omega_\phi/\Omega_\phi ^2$')
# plt.savefig('omegaPhi_dot.png')

Nspin = 30
Necc = 30
Ninc = 30
Np = 30

# spin
x1 = np.linspace(1e-10, 0.998, Nspin)
# eccentricity
x2 = np.linspace(1e-8, 0.99, Necc)
# inclination
x3 = np.linspace(-1+1e-8, 1-1e-8, Ninc)
# p-psep
x4 = np.linspace(0.1, 50.0, Np)

xx1, xx2, xx3, xx4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
psep = get_separatrix(xx1.flatten(),xx2.flatten(),xx3.flatten()).reshape(xx1.shape)

def get_p_from_u(a,u,e,x):
    return get_separatrix(a, e, x) + u

def get_u_from_p(a,p,e,x):
    return p - get_separatrix(a, e, x)

F = get_fundamental_frequencies(xx1.flatten(), xx4.flatten()+psep.flatten(), xx2.flatten(), xx3.flatten())[0].reshape(xx1.shape)
X = [x1, x2, x3, x4]

TPint = TPI.TP_Interpolant_ND(X)
TPint.TPInterpolationSetupND()
TPint.ComputeSplineCoefficientsND(F)

# Check coefficients
c_TPI = TPint.GetSplineCoefficientsND()

# Check evaluated interpolant
def get_interpolant(a, e, x, u):
    return 1-TPint.TPInterpolationND(np.array([a, e, x, u])) /get_fundamental_frequencies(a,get_p_from_u(a,u,e,x),e,x) [0]

# check interpolant at random points
# draw random x1, x2, x3, x4

x1_test = np.random.uniform(1e-10, 0.998,size=100)
x2_test = np.random.uniform(1e-8, 0.99,size=100)
x3_test = np.random.uniform(-1+1e-8, 1-1e-8,size=100)
x4_test = np.random.uniform(0.5, 50,size=100)

import time
tic = time.perf_counter()    
[TPint.TPInterpolationND(np.array([x1,x2,x3,x4])) for x1,x2,x3,x4 in zip(x1_test,x2_test,x3_test,x4_test)]
toc = time.perf_counter()
print(f"Interpolation time {(toc - tic)/100} seconds")
