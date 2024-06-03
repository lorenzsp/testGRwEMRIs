#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
import glob
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_kerr_geo_constants_of_motion, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *

# initialize trajectory class
traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
# run trajectory
err = 1e-10
insp_kw = {
    "err": err,
    "DENSE_STEPPING": 0,
    "use_rk4": False,
    }

# get Edot function
trajELQ = EMRIInspiral(func="KerrEccentricEquatorial")
def get_Edot(M, mu, a, p0, e0, x0, charge):
    trajELQ(M, mu, a, p0, e0, x0, charge, T=1e-2, dt=10.0, **insp_kw)
    y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)
    y0 = np.array([y1, y2, y3])
    return trajELQ.inspiral_generator.integrator.get_derivatives(y0)[0] / (mu/M)

def get_delta_Edot(M, mu, a, p0, e0, x0, charge):
    
    Edot_Charge = get_Edot(M, mu, a, p0, e0, x0, charge)
    Edot_ZeroCharge = get_Edot(M, mu, a, p0, e0, x0, 0.0)
    return Edot_Charge - Edot_ZeroCharge

spin = 0.95
check = np.loadtxt(f"a0.95_xI1.000.flux")
a, p, e, xi, E, Lz, Q, pLSO, EdotInf_tot, EdotH_tot, LzdotInf_tot, LzdotH_tot, QdotInf_tot, QdotH_tot, pdotInf_tot, pdotH_tot, eccdotInf_tot, eccdotH_tot, xidotInf_tot, xidotH_tot = check.T

# Target point
target_p, target_e = 10.0, 0.4

# Calculate the Euclidean distance between each point and the target point
distances = np.sqrt((p - target_p)**2 + (e - target_e)**2)

# Get the index of the closest point
index = np.argmin(distances)

# Get the closest point
closest_p, closest_e = p[index], e[index]
scott_flux = EdotInf_tot[index]+EdotH_tot[index]
our_flux = np.abs(get_Edot(1e6,10,spin,closest_p, closest_e,1.0,0.0))
print(closest_p, closest_e, np.abs(1-scott_flux/our_flux))
# fid in p,e close to some values in the check file
