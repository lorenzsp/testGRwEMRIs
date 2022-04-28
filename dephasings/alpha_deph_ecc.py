dev = 2
import os
print("process", os.getpid() )

os.system(f"CUDA_VISIBLE_DEVICES={dev}")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{dev}"
os.system("echo $CUDA_VISIBLE_DEVICES")

os.system("export OMP_NUM_THREADS=1")
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation

from few.waveform import Pn5AAKWaveform, AAKWaveformBase, SchwarzschildEccentricWaveformBase, GenerateEMRIWaveform
from few.utils.constants import *
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

import matplotlib.colors as mcol
import matplotlib.cm as cm
from lisatools.utils.transform import TransformContainer
from lisatools.sensitivity import get_sensitivity
from lisatools.diagnostic import (
    inner_product,
    snr,
    fisher,
    covariance,
    mismatch_criterion,
    cutler_vallisneri_bias,
    scale_snr,
)
from few.utils.baseclasses import Pn5AAK, ParallelModuleBase

# cosmological functions
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import constants as const
import astropy.units as unit

use_gpu = True

# set initial parameters
M = 1e6
mu = 1e1
a = 0.0
p0 = 8.0
e0 = 0.4
iota0 = 0.0
Y0 = np.cos(iota0)
Phi_phi0 = 0.0
Phi_theta0 = 0.0
Phi_r0 = 0.0

dt = 10.0
T = 4.0
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
        5e-3
    ]
)

inspiral_kwargs = {
"DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
"max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
"func": "ScalarSchwarzEccFlux"
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

wave_kw = dict(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)


args = [
EMRIInspiral,
RomanAmplitude,
InterpolatedModeSum,
]

wave = GenerateEMRIWaveform(SchwarzschildEccentricWaveformBase,*args, **wave_kw)

wave1 = wave(*injection_params, dt=dt, T=T).real
print(wave1)

###################################################################################
snr_goal = 30.0
# for SNR and covariance calculation
inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd", use_gpu=use_gpu)
# copy parameters to check values
check_params = injection_params.copy()
# INJECTION kwargs
waveform_kwargs = {"T": T, "dt": dt}
check_sig = wave(*check_params, **waveform_kwargs)

def sqrt_alpha_to_d(a, b):
    alpha = b**2
    ratio = 0.5*alpha/np.exp(2*a)
    return [np.exp(a), ratio * (1 + (73/480) * ratio**2 )  ]

dt = 10.0
T = 1.0

traj = EMRIInspiral(func="ScalarSchwarzEccFlux")

#######################################################################
num = 3
p_vec = np.linspace(7.5,8.5, num=num)
e_vec = np.linspace(0.2,0.4, num=num)

n = len(p_vec)

P = []
Ecc = []
SPIN = []
MU = []
dphi = []
Mass = []
SNR = []
z_h = []
sqrtalpha_charge = 1.0

for p0 in p_vec:
    for e0 in e_vec:

        print('-------------------------------')

        
        # traj_args = [M, 0.0, p0, e0, 1.0]
        # traj_kwargs = {}
        # index_of_mu = 1

        # t_out = T*0.999
        # # run trajectory
        # mu_new = get_mu_at_t(
        #     traj,
        #     t_out,
        #     traj_args,
        #     index_of_mu=index_of_mu,
        #     traj_kwargs=traj_kwargs,
        #     xtol=2e-12,
        #     rtol=8.881784197001252e-16,
        #     bounds=None,
        # )
        # mu = mu_new

        # max frequency
        # p_sep = get_separatrix(a, 0.0, 1.0)
        # freq_sep = 1.0 / (p_sep**1.5 + a) / (M * MTSUN_SI * np.pi)
        # dt = 0.5 * 1/freq_sep
        # print("dt",dt)
        
        waveform_kwargs = {"T": T, "dt": dt}#, "eps": 1e-2}
        check_params[0] = M
        check_params[1] = mu
        check_params[2] = a
        check_params[3] = p0
        check_params[4] = e0
        tru_sig = wave(*check_params, **waveform_kwargs)
        snr_temp = np.sqrt(inner_product(tru_sig, tru_sig, normalize=False, **inner_product_kwargs).get())
        SNR.append(snr_temp)

        # run trajectory
        _, charge = sqrt_alpha_to_d( np.log(mu), sqrtalpha_charge )
        args=np.array([charge])
        t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, Y0,  Phi_phi0, Phi_theta0, Phi_r0, *args,  T=T, dt=dt)
        args=np.array([0.0])
        t2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2 = traj(M, mu, a, p0, e0, Y0, Phi_phi0, Phi_theta0, Phi_r0, *args, T=T, dt=dt)#, upsample=True, new_t=t)
        
        tfinal = np.min([t[-1], t2[-1]])*0.99

        spl2 = CubicSplineInterpolant(t2, Phi_phi2)
        spl1 = CubicSplineInterpolant(t, Phi_phi)

        t_new = np.linspace(0,tfinal)
        delta_phi = np.abs(spl2(t_new[-1]) -spl1(t_new[-1]) )/(2*np.pi) 
        print("mu=",mu,"d=", charge,"deltaphi=", delta_phi)

        # redshift
        cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3075)
        d_new = snr_temp/20.0
        z_horizon = z_at_value(cosmo.luminosity_distance ,d_new*1e3*unit.Mpc)
        print('z', z_horizon)
        z_h.append(z_horizon)

        P.append(p0)
        Ecc.append(e0)
        MU.append(mu)
        SPIN.append(a)
        dphi.append(delta_phi)
        Mass.append(M)



########################################################


x = np.array(P)
y = np.array(Ecc)
z = np.log10(dphi)#mism


string = "snr20horizon_a{}_T{}".format(a, T)

with open(f"dephasing_"+string+".txt","w") as f:
    f.write("# p0"+"\t"+"e0"+"\t"+"log10Nphi"+"\t"+"z_horizon \n")
    for i in range(len(x)):
        f.write(f"{x[i]}\t{y[i]}\t{z[i]}\t{z_h[i]}\n")

breakpoint()