import numpy as np
from scipy.constants import golden
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, get_p_at_t
from few.utils.constants import *
import sys
sys.path.append('../DataAnalysis/')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Set the style of the plot
# import matplotlib.style as style
# style.use('tableau-colorblind10')

# Set the default figure size and ratio
default_width = 5.78853 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

import matplotlib.ticker as mticker

# Define a function to format the numbers as 10^n
def format_func(value, tick_number):
    # make the value into a proper scientific notation
    exponent = np.floor(np.log10(abs(value)))
    coeff = value / 10**exponent

    return r"${} \times 10^{{{:2.0f}}}$".format(coeff, exponent)

# Create a FuncFormatter object based on the function
formatter = mticker.FuncFormatter(format_func)

# Set the matplotlib parameters
inv_golden = 1. / golden
px = 2*0.0132
plt.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": 'pdflatex',
    "pgf.rcfonts": False,
    "font.family": "serif",
    "figure.figsize": [246.0*px, inv_golden * 246.0*px],
    'legend.fontsize': 11,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.title_fontsize' : 12,
})

# Load the grid data
grid = np.loadtxt("../mathematica_notebooks_fluxes_to_Cpp/final_grid/data_total.dat")

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

# Set the parameters for the trajectory
a = 0.95
charge = 0.0
x0 = 1.0

# Find the closest value indices in the grid
ind = find_closest_value_indices(grid[:,0], a)
mask = (grid[:,0] == grid[ind,0])

# Initialize the trajectory class
traj = EMRIInspiral(func="KerrEccentricEquatorial")
print(grid[ind,0])
# Plot the grid points
plt.plot(grid[mask,1], grid[mask,2], '.',label=fr'Grid points',alpha=0.7)#,ms=10) of $a=${grid[ind,0]:.2e}

# Set the parameters for the first plot
M = 1e6
mu = 10
e0 = 0.4
p0 = get_p_at_t(traj, 2.0 * 0.999, [M, mu, a, e0, x0, 0.0], bounds=[get_separatrix(a, e0, x0) + 0.1, 30.0])
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge**2/4, T=3.0, dt=10.0)

# Plot the first trajectory

plt.plot(p, e, '--', label=fr"$M=${M/1e6}$\times 10^6 \, M_\odot$" + fr', $\mu=$ {int(mu)} $M_\odot$', lw=3.0)

# Set the parameters for the second plot
M = 1e6
mu = 10
e0 = 0.1
p0 = get_p_at_t(traj, 2.0 * 0.999, [M, mu, a, e0, x0, 0.0], bounds=[get_separatrix(a, e0, x0) + 0.1, 30.0])
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge**2/4, T=3.0, dt=10.0)

# Plot the second trajectory
plt.plot(p, e, '-.', label=fr"$M=${M/1e6}$\times 10^6 \, M_\odot$" + fr', $\mu=$ {int(mu)} $M_\odot$', lw=3.0)

# Set the parameters for the third plot
M = 1e6
mu = 5
e0 = 0.4
p0 = get_p_at_t(traj, 2.0 * 0.999, [M, mu, a, e0, x0, 0.0], bounds=[get_separatrix(a, e0, x0) + 0.1, 30.0])
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge**2/4, T=3.0, dt=10.0)

# Plot the third trajectory
plt.plot(p, e, ':', label=fr"$M=${M/1e6}$\times 10^6 \, M_\odot$" + fr', $\mu=$ {int(mu)} $M_\odot$', lw=3.0)


# Set the parameters for the third plot
M = 5e5
mu = 10
e0 = 0.4
p0 = get_p_at_t(traj, 2.0 * 0.999, [M, mu, a, e0, x0, 0.0], bounds=[get_separatrix(a, e0, x0) + 0.1, 30.0])
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, charge**2/4, T=3.0, dt=10.0)

 
# Plot the third trajectory
plt.plot(p, e, linestyle=(0, (3, 1, 1, 1, 3)), label=fr"$M=${M/1e6}$\times 10^6 \, M_\odot$" + fr', $\mu=$ {int(mu)} $M_\odot$', lw=3.0)

# Set the labels and title of the plot
plt.xlabel('p',fontsize=20)
plt.ylabel('e',fontsize=20)
plt.tight_layout()
plt.legend(loc='upper right',ncol=2)
plt.xlim(1.5,12.5)
plt.ylim(-0.05,0.75)
# Save and show the plot
plt.savefig('./figures/plot_grid.pdf')
# plt.show()
