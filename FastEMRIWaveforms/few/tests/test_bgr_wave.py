import unittest
import numpy as np
import warnings
import time

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

T = 5.0
dt = 10.0
from few.waveform import AAKWaveformBase, Pn5AAKWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.waveform import GenerateEMRIWaveform
np.random.seed(42)

class ModuleTest(unittest.TestCase):

    def test_bgr_KerrEccentricEquatorial(self):

        err = 1e-10
        
        # initialize trajectory class
        traj = EMRIInspiral(func="KerrEccentricEquatorial")
        # set initial parameters
        M = 1e6
        mu = 5e1
        p0 = 12.0
        e0 = 0.1
        a=-0.987
        x0=1.0
        charge = 0.1

        insp_kwargs = {
            "err": 1e-10,
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e4),
            "func":"KerrEccentricEquatorial"
            }

        # keyword arguments for summation generator (AAKSummation)
        sum_kwargs = {
            "use_gpu": gpu_available,  # GPU is availabel for this type of summation
            "pad_output": True,
        }

        few_gen = GenerateEMRIWaveform(
            AAKWaveformBase, 
            EMRIInspiral,
            AAKSummation,
            return_list=False,
            inspiral_kwargs=insp_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        val = np.array([[1.38154677e+01, 9.99980977e+00, 8.99980810e-01, 8.53781839e+00,
        1.00001478e-01, 5.19419862e+00, 5.09203890e-01, 1.04295016e+00,
        5.09203886e-01, 1.04295016e+00, 1.11756571e+00, 1.23079432e+00,
        2.37556467e-04],
       [1.38154980e+01, 9.99999167e+00, 8.99994904e-01, 8.53761299e+00,
        1.00004733e-01, 5.29273798e+00, 4.90734877e-01, 1.05214844e+00,
        4.90734856e-01, 1.05214844e+00, 6.24405950e+00, 1.24314698e+00,
        3.15615718e-05],
       [1.38154753e+01, 9.99813461e+00, 8.99973912e-01, 8.53801693e+00,
        9.98225295e-02, 5.06077958e+00, 4.97165058e-01, 1.04384210e+00,
        4.97165015e-01, 1.04384210e+00, 9.74736657e-01, 7.35249302e-01,
        1.82949819e-03],
       [1.38154885e+01, 9.99857663e+00, 8.99981467e-01, 8.53785474e+00,
        9.98817398e-02, 5.17212281e+00, 4.63074798e-01, 1.03794602e+00,
        4.63074752e-01, 1.03794602e+00, 3.28248483e+00, 1.58806705e+00,
        1.33978260e-03],
       [1.38155033e+01, 9.99900773e+00, 8.99992247e-01, 8.53773484e+00,
        9.98902215e-02, 5.10226206e+00, 4.81950111e-01, 1.05242426e+00,
        4.81950100e-01, 1.05242426e+00, 4.30189806e+00, 1.32596589e+00,
        9.98910810e-04],
       [1.38154509e+01, 9.99942359e+00, 8.99972269e-01, 8.53798064e+00,
        9.99577308e-02, 5.15456300e+00, 5.05124936e-01, 1.02994222e+00,
        5.05124936e-01, 1.02994222e+00, 3.43971747e+00, 1.10129051e+00,
        6.44488648e-04],
       [1.38154369e+01, 9.99942238e+00, 8.99967284e-01, 8.53805685e+00,
        9.99800771e-02, 4.99760848e+00, 5.16083006e-01, 1.03030270e+00,
        5.16083000e-01, 1.03030271e+00, 4.95033280e+00, 5.99362227e-01,
        6.52731433e-04],
       [1.38155138e+01, 9.99984983e+00, 9.00003202e-01, 8.53756782e+00,
        9.99577851e-02, 5.29081493e+00, 4.91133876e-01, 1.06355440e+00,
        4.91133871e-01, 1.06355440e+00, 1.78079293e+00, 1.35175346e+00,
        2.36835231e-04],
       [1.38155074e+01, 9.99964033e+00, 8.99995152e-01, 8.53758722e+00,
        9.99819827e-02, 5.11316601e+00, 4.93318608e-01, 1.04985691e+00,
        4.93318619e-01, 1.04985691e+00, 6.19328830e+00, 1.70818960e+00,
        2.92973021e-04],
       [1.38155309e+01, 9.99968307e+00, 9.00007099e-01, 8.53746005e+00,
        9.99598571e-02, 5.13816858e+00, 5.01264538e-01, 1.05393110e+00,
        5.01264552e-01, 1.05393110e+00, 2.59934726e+00, 1.64453838e+00,
        2.82547105e-04],
       [1.38155288e+01, 9.99943153e+00, 9.00003985e-01, 8.53751275e+00,
        9.99280971e-02, 5.17326348e+00, 4.93882867e-01, 1.05241237e+00,
        4.93882847e-01, 1.05241236e+00, 1.59681582e-01, 1.85488731e+00,
        5.30607614e-04],
       [1.38155023e+01, 9.99975543e+00, 8.99996649e-01, 8.53762605e+00,
        9.99715456e-02, 5.08296592e+00, 4.95426304e-01, 1.04369061e+00,
        4.95426297e-01, 1.04369061e+00, 1.00630068e+00, 1.33045208e+00,
        2.70046515e-04],
       [1.38155449e+01, 9.99922410e+00, 9.00009267e-01, 8.53744886e+00,
        9.99046380e-02, 5.07094215e+00, 4.76565013e-01, 1.07209026e+00,
        4.76564969e-01, 1.07209026e+00, 4.75448623e+00, 1.79040186e+00,
        7.16421152e-04],
       [1.38154936e+01, 9.99983778e+00, 8.99993714e-01, 8.53766332e+00,
        9.99896979e-02, 5.07332797e+00, 4.92021386e-01, 1.04248541e+00,
        4.92021370e-01, 1.04248541e+00, 6.51268924e-02, 1.24936425e+00,
        1.98377100e-04],
       [1.38155190e+01, 9.99949792e+00, 8.99997004e-01, 8.53752869e+00,
        9.99664990e-02, 5.28021827e+00, 4.88324353e-01, 1.04184345e+00,
        4.88324352e-01, 1.04184345e+00, 6.43098412e-01, 2.01449772e+00,
        3.73856003e-04]])
        

        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge)