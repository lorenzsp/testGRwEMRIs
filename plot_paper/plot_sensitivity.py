from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
from lisatools.sensitivity import *
S_git = np.genfromtxt('../../EMRI_FourierDomainWaveforms/LISA_Alloc_Sh.txt')
Sh_X = CubicSpline(S_git[:,0], S_git[:,1])
cornish_lisa_psd(1e-3)/lisasens(1e-3)
# 3 plot the different sens
plt.figure()
freq = 10**np.linspace(-5,0.0,1000)
plt.loglog(freq, lisasens(freq)**0.5, label='lisasens')
plt.loglog(freq, Sh_X(freq)**0.5,'--', label='LISA Alloc')
plt.loglog(freq, cornish_lisa_psd(freq)**0.5,':', label='Cornish ')
plt.legend()
plt.grid()
plt.savefig('sens')