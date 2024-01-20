#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
import glob
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *
from few.utils.utility import get_p_at_t, get_separatrix

traj = EMRIInspiral(func="KerrEccentricEquatorial")
# run trajectory
err = 1e-10
insp_kw = {
    "err": err,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    }



np.random.seed(32)
import matplotlib.pyplot as plt
import time, os
print(os.getpid())

# initialize trajectory class
traj = EMRIInspiral(func="KerrEccentricEquatorial")

grid = np.loadtxt("../mathematica_notebooks_fluxes_to_Cpp/final_grid/data_total.dat")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate some random data for demonstration
np.random.seed(42)

sep = get_separatrix(np.abs(grid[:,0]),grid[:,2]+1e-16,np.sign(grid[:,0])*1.0)
x = np.log10(grid[:,1] - sep)
y = grid[:,2]
z = grid[:,0]

# Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='r', marker='o')
# # ax.scatter(sep, y, z, c='b', marker='o')

# # Set labels for each axis
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# # Set the title of the plot
# ax.set_title('3D Scatter Plot')

# # Show the plot
# plt.show()

def find_closest_value_indices(array, target_value):
    """
    Find the indices where the values in the array are closest to the target value.

    Parameters:
    - array: NumPy array
    - target_value: The value to which you want to find the closest indices

    Returns:
    - indices: Indices where the values are closest to the target value
    """
    # Calculate the absolute differences between array values and the target value
    absolute_diff = np.abs(array - target_value)

    # Find the index with the minimum absolute difference
    closest_index = np.argmin(absolute_diff)

    return closest_index

# rhs ode
M = 1e6
mu = 1e1
a = 0.95
e0=0.4
x0=1.0
e0=0.4

p0 = get_p_at_t(
        traj,
        2.0 * 0.999,
        [M, mu, a, e0, x0, 0.0],
        bounds=[get_separatrix(a,e0,x0)+0.1, 30.0],
    )

# p0=get_separatrix(a,e0,1.0) + 2.0
charge = 0.0

ind = find_closest_value_indices(grid[:,0],a,)

mask = (grid[:,0]== grid[ind,0])


plt.figure()
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=3.0, dt=10.0, **insp_kw)
plt.plot(p,e,'.')
plt.plot(grid[mask,1],grid[mask,2],'x')
plt.xlabel('p')
plt.ylabel('e')
plt.xlim(1,5)
plt.tight_layout()
plt.savefig('grid_plot')

out_deriv = np.asarray([traj.get_rhs_ode(M, mu, a, pp, ee, xx, charge) for pp,ee,xx in zip(p, e, np.ones_like(p)*x0)])

plt.figure()
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge, T=3.0, dt=10.0, **insp_kw)
plt.plot(p,out_deriv[:,0],'x')
plt.xlabel('p')
plt.ylabel('flux')
plt.tight_layout()
plt.savefig('rhs_ode')
