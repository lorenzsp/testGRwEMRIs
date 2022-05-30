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

xs2, t2 = np.load("p_2.npy")
xs, t = np.load("p_1.npy")
xs3, t3 = np.load("p_3.npy")

plt.figure()
plt.plot(t, xs)
plt.plot(t2, xs2, '--', label='2')
plt.plot(t3, xs3, '.', label='3')
plt.legend()
plt.show()