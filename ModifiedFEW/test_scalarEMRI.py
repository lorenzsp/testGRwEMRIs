# Test waveform and trajectory python -m unittest test_scalar_wave.py 
import unittest
import numpy as np
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase, SchwarzschildEccentricWaveformBase
from few.utils.utility import (get_overlap, 
                               get_mismatch, 
                               get_fundamental_frequencies, 
                               get_separatrix, 
                               get_mu_at_t, 
                               get_p_at_t, 
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.summation.interpolatedmodesum import CubicSplineInterpolant, InterpolatedModeSum
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase
from few.amplitude.romannet import RomanAmplitude

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class ModuleTest(unittest.TestCase):
    def test_trajectoryKerr(self):
        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.9
        p0 = 7.2
        e0 = 0.0
        iota0 = 0.0
        Y0 = np.cos(iota0)
        Phi_phi0 = 0.0
        Phi_theta0 = 0.0
        Phi_r0 = 0.0

        dt = 10.0
        T = 1.0

        args1=np.array([
            0.05,
        ])

        traj = EMRIInspiral(func="KerrCircFlux")

        # run trajectory
        t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, Y0,  Phi_phi0, Phi_theta0, Phi_r0, *args1,  T=T, dt=dt)


        args=np.array([
            0.0,
        ])

        t2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2 = traj(M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, *args, T=T, dt=dt, upsample=True, new_t=t)

        tfinal = np.min([t[-1], t2[-1]])

        spl2 = CubicSplineInterpolant(t2, Phi_phi2)
        spl1 = CubicSplineInterpolant(t, Phi_phi)

        self.assertAlmostEqual(np.abs(spl2(tfinal)-spl1(tfinal)+451890.89951528877)/451890.89951528877, 0.0,places=4)

    def test_trajectorySchwarzchild(self):
        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.0
        p0 = 7.2
        e0 = 0.4
        iota0 = 0.0
        Y0 = np.cos(iota0)
        Phi_phi0 = 0.0
        Phi_theta0 = 0.0
        Phi_r0 = 0.0

        dt = 10.0
        T = 1.0

        args1=np.array([
            0.05,
        ])

        traj = EMRIInspiral(func="ScalarSchwarzEccFlux")

        # run trajectory
        t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, Y0,  Phi_phi0, Phi_theta0, Phi_r0, *args1,  T=T, dt=dt)


        args=np.array([
            0.0,
        ])

        t2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2 = traj(M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, *args, T=T, dt=dt, upsample=True, new_t=t)

        tfinal = np.min([t[-1], t2[-1]])

        spl2 = CubicSplineInterpolant(t2, Phi_phi2)
        spl1 = CubicSplineInterpolant(t, Phi_phi)
        # breakpoint()
        print(np.abs(spl2(tfinal)-spl1(tfinal)) )


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



########################################################

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

class ScalarFEW(SchwarzschildEccentricWaveformBase, Pn5AAK, ParallelModuleBase):

    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        inspiral_kwargs["func"] = "ScalarSchwarzEccFlux"

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            EMRIInspiral,
            RomanAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )

class WaveformTest(unittest.TestCase):
    def test_Kerr(self):
        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.9
        p0 = 7.2
        e0 = 0.0
        iota0 = 0.0
        Y0 = np.cos(iota0)
        Phi_phi0 = 0.0
        Phi_theta0 = 0.0
        Phi_r0 = 0.0

        dt = 10.0
        T = 1.0
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
                1e-2
            ]
        )

        inspiral_kwargs={}
        inspiral_kwargs["func"] = "KerrCircFlux"

        wave_generator = NewPn5AAKWaveform(inspiral_kwargs=inspiral_kwargs)
        wave1 = wave_generator(*injection_params, mich=False, dt=dt, T=T).real

        injection_params[-1]=0.0

        inspiral_kwargs={}
        inspiral_kwargs["func"] = "KerrCircFlux"
        wave_generator2 = NewPn5AAKWaveform(inspiral_kwargs=inspiral_kwargs)
        wave2 = wave_generator2(*injection_params, mich=False, dt=dt, T=T).real

        self.assertAlmostEqual(Overlap_LISA(wave1, wave2, dt),0.2935622794001746,places=3)

    def test_Schwarz(self):
        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.0
        p0 = 7.2
        e0 = 0.4
        iota0 = 0.0
        Y0 = np.cos(iota0)
        Phi_phi0 = 0.0
        Phi_theta0 = 0.0
        Phi_r0 = 0.0

        dt = 10.0
        T = 1.0
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
                1e-2
            ]
        )
        
        inspiral_kwargs = {
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "max_init_len": int(
                1e3
            )  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {}

        wave_generator = SchwarzschildEccentricWaveformBase(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        wave1 = wave_generator(*injection_params, mich=False, dt=dt, T=T).real

        injection_params[-1]=0.0


        wave2 = wave_generator(*injection_params, mich=False, dt=dt, T=T).real

        print(Overlap_LISA(wave1, wave2, dt))


