dev = 4
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
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
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

from few.summation.interpolatedmodesum import CubicSplineInterpolant
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

class ScalarAAKWaveform(AAKWaveformBase, Pn5AAK, ParallelModuleBase):
    def __init__(
        self, inspiral_kwargs={}, sum_kwargs={}, use_gpu=False, num_threads=None, return_list=True,
    ):

        AAKWaveformBase.__init__(
            self,
            EMRIInspiral,  # trajectory class
            AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            num_threads=num_threads,
            return_list=return_list
        )

use_gpu = True

inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "func": "KerrCircFlux"
}

wave_gen = ScalarAAKWaveform(
            sum_kwargs=dict(pad_output=True),
            inspiral_kwargs=inspiral_kwargs, 
            use_gpu=use_gpu,
            return_list=True,
            )

# define injection parameters
M = 1e6
mu = 50.0
p0 = 12.
e0 = 0.0
Y0 = 1.0

# Acc
Phi_phi0 = 0.0
Phi_theta0 = 0.0
Phi_r0 = 0.0
# define other parameters necessary for calculation
a = 0.99
qS = 0.5420879369091457
phiS = 5.3576560705195275
qK = 1.7348119514252445
phiK = 3.2004167279159637
dist = 1.0

# injection array
injection_params = np.array(
    [
        np.log(M),
        np.log(mu),
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
        0.0
    ]
)

# define other quantities
T = 4.0   # years
dt = 10.0

###################################################################################
snr_goal = 30.0
# for SNR and covariance calculation
inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd", use_gpu=use_gpu)
transform_fn_in ={0: (lambda x: np.exp(x)),
                    1: (lambda x: np.exp(x)),
                    }
# use the special transform container
transform_fn = TransformContainer(transform_fn_in)
# copy parameters to check values
check_params = injection_params.copy()
# this transforms but also gives the transpose of the input dimensions due to inner sampler needs
check_params = transform_fn.transform_base_parameters(check_params).T
# INJECTION kwargs
waveform_kwargs = {"T": T, "dt": dt, "mich":False}
check_sig = wave_gen(*check_params, **waveform_kwargs)

def sqrt_alpha_to_d(a, b):
    alpha = b**2
    ratio = 0.5*alpha/np.exp(2*a)
    return [np.exp(a), ratio * (1 + (73/480) * ratio**2 )  ]

dt = 10.0
T = 4.0

traj = EMRIInspiral(func="KerrCircFlux")

def plot_config(ECC, IOTA, SPIN, var, charz, filename, ang1, ang2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start = np.min(SPIN)
    end = np.max(SPIN)

    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["gold","b"])

    # Make a normalizer that will map the time values from
    # [start_time,end_time+1] -> [0,1].
    cnorm = mcol.Normalize(vmin=start,vmax=end)

    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])

    plt.colorbar(cpick,label=r"SNR", pad=0.1)

    # olde plotting
    # for I in range(0, len(ECC)):
    #     ax.scatter(ECC[I], IOTA[I], var[I], c=cpick.to_rgba(SPIN[I]), alpha=0.7)#, markersize=SPIN[I],s=(10*SPIN[I])**(2*SPIN[I]),)
    
    # new
    X,Y = np.meshgrid( np.unique(ECC), np.unique(IOTA) )
    nshape = len(np.unique(ECC))
    Z = np.reshape(var, (nshape, nshape))
    col = np.reshape(SPIN, (nshape, nshape))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,facecolors=cpick.to_rgba(col))#,linewidth=0, antialiased=False, shade=False )
    
    ax.set_xlabel(r'$\mu$',fontsize=12)
    ax.set_ylabel(r'$\log_{10}M$',fontsize=12)
    ax.set_zlabel(charz,fontsize=12)

    ax.view_init(ang1, ang2)
    plt.tight_layout()

    plt.savefig(filename)

#######################################################################
num = 3
p_vec = np.linspace(8.0,18.0, num=num)
spin = np.linspace(0.01,0.99, num=num)
M_vec = 10**np.linspace(5.5,7.0, num=num)
eta_vec = 10**np.linspace(-5.5,-4.0, num=num)
# mu_vec = np.linspace(5.0,50.0, num=num)
mu_vec = np.linspace(5.0,20.0, num=num)
n = len(p_vec)

P = []
SPIN = []
MU = []
dphi = []
Mass = []
SNR = []
z_h = []
sqrtalpha_charge = 1.0

# for p0 in p_vec:
#     for a in spin:

for M in M_vec:
    for eta in eta_vec:
        print('-------------------------------')
        mu = eta*M
        # set initial parameters
        traj_args = [M, mu, a, e0, Y0]
        traj_kwargs = {}
        index_of_mu = 3

        t_out = T*0.999
        # run trajectory
        sep = get_separatrix(a, 0.0, 1.0)
        p_new = get_p_at_t(
            traj,
            t_out,
            traj_args,
            traj_kwargs=traj_kwargs,
            xtol=2e-8,
            rtol=8.881784197001252e-10,
            bounds=[sep+0.1, 30.0],
        )

        print('p0 = {} will create a waveform that is {} years long, given the other input parameters.'.format(p_new, t_out))
        p0 = p_new
        # max frequency
        p_sep = get_separatrix(a, 0.0, 1.0)
        freq_sep = 1.0 / (p_sep**1.5 + a) / (M * MTSUN_SI * np.pi)
        dt = 0.5 * 1/freq_sep
        print("dt",dt)
        
        waveform_kwargs = {"T": T, "dt": dt, "mich":False}#, "eps": 1e-2}
        check_params[0] = M
        check_params[1] = mu
        check_params[2] = a
        check_params[3] = p0
        tru_sig = wave_gen(*check_params, **waveform_kwargs)
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
        MU.append(mu)
        SPIN.append(a)
        dphi.append(delta_phi)
        Mass.append(M)



########################################################


x = np.log10(np.array(Mass))
y = np.log10(np.array(MU)/np.array(Mass))
z = np.log10(dphi)#mism


string = "snr20horizon_a{}_T{}".format(a, T)

with open(f"dephasing_"+string+".txt","w") as f:
    f.write("# log10M1"+"\t"+"log10eta"+"\t"+"log10Nphi"+"\t"+"z_horizon \n")
    for i in range(len(x)):
        f.write(f"{x[i]}\t{y[i]}\t{z[i]}\t{z_h[i]}\n")

breakpoint()

#######################################################
from mpl_toolkits.mplot3d import Axes3D  

# x = np.asarray(MU)#np.log10(np.asarray(MU)/np.asarray(Mass)) #SPIN
# y = np.log10(np.asarray(Mass))
# z = np.log10(np.array(dphi)) 

plot_config(x, y, SNR, z, '$\log_{10}\mathcal{N}_\phi$', f'dephasing_sqrtAlpha{sqrtalpha_charge}_Tyr{T}.pdf', 30, 40)
# plot_config(SPIN, P, MU, dphi, '$\Delta \Phi_\phi/(2\pi)$', f'dephasing_sqrtAlpha{sqrtalpha_charge}_Tyr{T}.pdf', 30, 150)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)

ax.set_xlabel('$a$')
ax.set_ylabel('$p_0$')
ax.set_zlabel('$\Delta \Phi_\phi/(2\pi)$')
ax.view_init(30, 150)
# plt.savefig("surf_deph_M{}_Tyr{}_charge{}.pdf".format(M, T, charge) )
