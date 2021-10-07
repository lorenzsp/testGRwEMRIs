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



# set initial parameters
M = 1e6
mu = 1e1
a = 0.9
p0 = 15.0
e0 = 0.0
iota0 = 0.0
Y0 = np.cos(iota0)
Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

dt = 11.0
T = 2.0


args=np.array([
    0.0,
    0.0,
    0.0
])

traj = EMRIInspiral(func="KerrCircFlux")

# run trajectory
import time
start = time.time()
t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, Y0,  Phi_phi0, Phi_theta0, Phi_r0,  T=T, dt=dt)
print('time', time.time()-start)


args=np.array([
    1e-20, # Amplitude
    0.0, # n
    0.0 # m
])

start = time.time()
t2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2 = traj(M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, *args, T=T, dt=dt)
print('time', time.time()-start)

########################################################
fig, axes = plt.subplots(2, 4)
plt.subplots_adjust(wspace=0.5)
fig.set_size_inches(14, 8)
axes = axes.ravel()

ylabels = [r'$e$', r'$p$', r'$e$', r'$Y$', r'$\Phi_\phi$', r'$\Phi_r$', r'$\Phi_\theta$']
xlabels = [r'$p$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$']
ys = [e, p, e, Y, Phi_phi, Phi_r, Phi_theta]
xs = [p, t, t, t, t, t, t]

ys2 = [e2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2]
xs2 = [p2, t2, t2, t2, t2, t2, t2]

for i, (ax, x, y, x2, y2, xlab, ylab) in enumerate(zip(axes, xs, ys, xs2, ys2, xlabels, ylabels)):
    ax.plot(x, y, label='Acc+Flux')#
    ax.plot(x2, y2, '--', label='Flux')
    ax.set_xlabel(xlab, fontsize=16)
    ax.set_ylabel(ylab, fontsize=16)
    ax.legend()


axes[-1].set_visible(False)
plt.show()
########################################################


from few.utils.baseclasses import Pn5AAK, ParallelModuleBase

class NewPn5AAKWaveform(AAKWaveformBase, Pn5AAK, ParallelModuleBase):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None
    ):

        AAKWaveformBase.__init__(
            self,
            EMRIInspiral,  # trajectory class
            AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
        )

qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
mich = False


injection_params = np.array(
    [
        M,
        mu,
        a,
        p0,
        e0,
        Y0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        args[0],
        args[1],
        args[2],
    ]
)

inspiral_kwargs={}
inspiral_kwargs["func"] = "KerrCircFlux"

wave_generator = NewPn5AAKWaveform(inspiral_kwargs=inspiral_kwargs)
wave1 = wave_generator(*injection_params, mich=False, dt=dt, T=T).real

injection_params[-3]=0.0
injection_params[-1]=0.0
injection_params[-2]=0.0

inspiral_kwargs={}
inspiral_kwargs["func"] = "KerrCircFlux"
wave_generator2 = NewPn5AAKWaveform(inspiral_kwargs=inspiral_kwargs)
wave2 = wave_generator2(*injection_params, mich=False, dt=dt, T=T).real

#plt.figure()
#plt.loglog( (np.fft.fft(wave1)) )
#plt.loglog( (np.fft.fft(wave2)) , '--')
#plt.show()





def PowerSpectralDensity(f):

    
    sky_averaging_constant = 1.0 # set to one for one source
    #(20/3) # Sky Averaged <--- I got this from Jonathan's notes
    L = 2.5*10**9   # Length of LISA arm
    f0 = 19.09*10**(-3)    

    Poms = ((1.5e-11)*(1.5e-11))*(1 + np.power((2e-3)/f, 4))  # Optical Metrology Sensor
    Pacc = (3e-15)*(3e-15)* (1 + (4e-4/f)*(4e-4/f))*(1 + np.power(f/(8e-3),4 ))  # Acceleration Noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    Sc = 0 #9e-45 * np.power(f,-7/3)*np.exp(-np.power(f,alpha) + beta*f*np.sin(k*f)) * (1 + np.tanh(gamma*(f_k- f)))  

    PSD = (sky_averaging_constant)* ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD

    return PSD

def InnerProd_LISA(sig1,sig2,delta_t):


    if len(sig1) != len(sig2):
        print("Signals do not have the same length")
    
    N = len(sig1)   # Calculate the length of the signal
    freq_bin = np.delete(np.fft.rfftfreq(N,delta_t),0)  # Sample individual fourier frequencies f_{j} = j/(N*delta_t)
    
    n_f = len(freq_bin)
    fft_1 = np.delete(np.fft.rfft(sig1),0)
    fft_2 = np.delete(np.fft.rfft(sig2),0)
    PSD =PowerSpectralDensity(np.abs(freq_bin))
    # notice that we did not multiply the fourier transform by dt because we consider that now!
    return (4*delta_t)*np.real(np.sum( (fft_1)* np.conj(fft_2)/(PSD *N) ) )


def Overlap_LISA(sig1,sig2,delta_t):
    numerator = InnerProd_LISA(sig1,sig2,delta_t)
    denominator = np.sqrt(InnerProd_LISA(sig1,sig1,delta_t) \
                          * InnerProd_LISA(sig2,sig2,delta_t))
    return numerator/denominator

print(Overlap_LISA(wave1, wave2, dt))