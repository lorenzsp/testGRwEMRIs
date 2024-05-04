from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_fundamental_frequencies, get_separatrix
import numpy as np
import matplotlib.pyplot as plt
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



