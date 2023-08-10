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

from few.summation.interpolatedmodesum import CubicSplineInterpolant
import matplotlib.colors as mcol
import matplotlib.cm as cm

# set initial parameters
M = 1e6
mu = 1e1
a = 0.9
p0 = 7.2
e0 = 0.3
iota0 = 0.0
Y0 = np.cos(iota0)
Phi_phi0 = 0.0
Phi_theta0 = 0.0
Phi_r0 = 0.0

dt = 10.0
T = 4.0

traj = EMRIInspiral(func="KerrEccentricEquatorial")

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

    plt.colorbar(cpick,label=r"$\mu$", pad=0.1)

    # olde plotting
    # for I in range(0, len(ECC)):
    #     ax.scatter(ECC[I], IOTA[I], var[I], c=cpick.to_rgba(SPIN[I]), alpha=0.7)#, markersize=SPIN[I],s=(10*SPIN[I])**(2*SPIN[I]),)
    
    # new
    X,Y = np.meshgrid( np.unique(ECC), np.unique(IOTA) )
    nshape = len(np.unique(ECC))
    Z = np.reshape(var, (nshape, nshape))
    col = np.reshape(SPIN, (nshape, nshape))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,facecolors=cpick.to_rgba(col))#,linewidth=0, antialiased=False, shade=False )
    
    ax.set_xlabel(r'$a$',fontsize=12)
    ax.set_ylabel(r'$p_0$',fontsize=12)
    ax.set_zlabel(charz,fontsize=12)

    ax.view_init(ang1, ang2)
    plt.tight_layout()

    plt.savefig(filename)

#######################################################################
num = 4
p_vec = np.linspace(8.0,18.0, num=num)
spin = np.linspace(0.01,0.99, num=num)
n = len(p_vec)

P = []
SPIN = []
MU = []
dphi = []

charge = 0.03

for p0 in p_vec:
    for a in spin:
        # set initial parameters
        traj_args = [M, a, p0, e0, Y0]
        traj_kwargs = {}
        index_of_mu = 1

        t_out = T*1.001
        # run trajectory
        mu_new = get_mu_at_t(
            traj,
            t_out,
            traj_args,
            index_of_mu=index_of_mu,
            traj_kwargs=traj_kwargs,
            xtol=2e-8,
            rtol=8.881784197001252e-10,
            bounds=None,
        )

        print('mu = {} will create a waveform that is {} years long, given the other input parameters.'.format(mu_new, t_out))
        mu = mu_new
        
        
        # run trajectory
        args=np.array([charge])
        t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, Y0, *args, Phi_phi0, Phi_theta0, Phi_r0,   T=T, dt=dt)
        args=np.array([0.0])
        t2, p2, e2, Y2, Phi_phi2, Phi_r2, Phi_theta2 = traj(M, mu, a, p0, e0, Y0, *args, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)#, upsample=True, new_t=t)
        
        
        tfinal = np.min([t[-1], t2[-1]])

        spl2 = CubicSplineInterpolant(t2, Phi_phi2)
        spl1 = CubicSplineInterpolant(t, Phi_phi)

        t_new = np.linspace(0,tfinal)

        P.append(p0)
        MU.append(mu)
        SPIN.append(a)
        dphi.append(np.abs(spl2(t_new[-1]) -spl1(t_new[-1]) )/(2*np.pi) )


########################################################

from mpl_toolkits.mplot3d import Axes3D  

plot_config(SPIN, P, MU, dphi, '$\Delta \Phi_\phi/(2\pi)$', f'dephasing_charge{charge}_Tyr{T}.pdf', 30, 150)

x = SPIN
y = P
z = dphi

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)

ax.set_xlabel('$a$')
ax.set_ylabel('$p_0$')
ax.set_zlabel('$\Delta \Phi_\phi/(2\pi)$')
ax.view_init(30, 150)
# plt.savefig("surf_deph_M{}_Tyr{}_charge{}.pdf".format(M, T, charge) )
