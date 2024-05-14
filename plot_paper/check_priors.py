# use bilby to create priors bilby.gw.prior.UniformInComponentsChirpMass and bilby.gw.prior.UniformInComponentsMassRatio
import numpy as np
import bilby
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio
from bilby.gw.conversion import *
import corner
import matplotlib.pyplot as plt

prior_chirp_mass = UniformInComponentsChirpMass(name='chirp_mass', minimum=1.9, maximum=2.2)
prior_mass_ratio = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1.0)

Mc = prior_chirp_mass.sample(10000)
q = prior_mass_ratio.sample(10000)
# convert from mass ratio to component masses
M = chirp_mass_and_mass_ratio_to_total_mass(Mc, q)
m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(Mc, q)
# plot corner plot
to_plot = np.vstack([m1, m2, Mc, q]).T
labels = ['m1', 'm2', 'Mc', 'q', ]

fig = corner.corner(to_plot, labels=labels,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    show_titles=False,)

plt.savefig('priors_elise.png')
# ------------------------------------------------------------
mu = 10.0
M = 1e6
Mc_max = component_masses_to_chirp_mass(M*(1+0.01), mu*(1+0.01))
Mc_min = component_masses_to_chirp_mass(M*(1-0.01), mu*(1-0.01))
Mc_true = component_masses_to_chirp_mass(M, mu)

prior_chirp_mass = UniformInComponentsChirpMass(name='chirp_mass', minimum=Mc_min, maximum=Mc_max)
prior_mass_ratio = UniformInComponentsMassRatio(name='mass_ratio', 
                                                minimum=mu*(1-0.01) / (M*(1+0.01)), 
                                                maximum=mu*(1+0.01) / (M*(1-0.01)))

def get_Mc_q_from_lnM_lnmu(lnM, lnmu):
    m1 = np.exp(lnM)
    m2 = np.exp(lnmu)
    return component_masses_to_chirp_mass(m1, m2), component_masses_to_mass_ratio(m1, m2)

def get_weights(lnM, lnmu):
    Mc,q = get_Mc_q_from_lnM_lnmu(lnM, lnmu)
    return np.exp(prior_chirp_mass.ln_prob(Mc)), np.exp(prior_mass_ratio.ln_prob(q))

Mc = prior_chirp_mass.sample(10000)
q = prior_mass_ratio.sample(10000)
# convert from mass ratio to component masses
M = chirp_mass_and_mass_ratio_to_total_mass(Mc, q)
m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(Mc, q)

# plot corner plot
to_plot = np.log(np.vstack([m1, m2, Mc, q]).T)
labels = ['ln m1', 'ln m2', 'ln Mc', 'ln q', ]

fig = corner.corner(to_plot, labels=labels,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    # truths=[np.log(M), np.log(mu), np.log(Mc_true), np.log(1e-5)],
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    show_titles=False,)

plt.savefig('priors_elise_EMRI.png')