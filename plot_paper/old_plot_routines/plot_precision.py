import glob
# from eryn.backends import HDFBackend
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
import matplotlib as mpl
import re
# import matplotlib.style as style
# style.use('tableau-colorblind10')

default_width = 5.78853 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

import matplotlib.ticker as mticker

vals = [0.000001,0.00001,0.0001,0.01,1.0]

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

g = lambda x,pos : "${}$".format(f.set_scientific('%1.10e' % x))
fmt = mticker.FuncFormatter(g)

from scipy.constants import golden
import corner
inv_golden = 1. / golden

px = 2*0.0132

mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": 'pdflatex',
    "pgf.rcfonts": False,
    "font.family": "serif",
    "figure.figsize": [246.0*px, inv_golden * 246.0*px],
  'legend.fontsize': 12,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
  'legend.title_fontsize' : 12,

# "axes.formatter.min_exponent": 1
# "axes.formatter.offset_threshold": 10
})


import matplotlib.lines as mlines

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

labels = [r'$ M [{\rm M}_\odot]$', r'$\mu [{\rm M}_\odot]$', r'$ a$', r'$ p_0 \, [M]$', r'$ e_0$', 
            r'$ D_L \, [{\rm Gpc}]$',
            r"$ \cos \theta_S$",r"$ \phi_S$",
            r"$ \cos \theta_K$",r"$ \phi_K$",
        r'$ \Phi_{\varphi 0}$', r'$ \Phi_{r 0}$',
            r"$ d$",
        ]

CORNER_KWARGS = dict(
    labels=labels,
    bins=40,
    truths=np.zeros(len(labels)),
    label_kwargs=dict(fontsize=35),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    show_titles=False,
    max_n_ticks=4,
    truth_color='k',
    labelpad=0.3,
)
def overlaid_corner(samples_list, sample_labels, name_save=None, corn_kw=None, title=None, ylim=None):
    """
    Plots multiple corners on top of each other.

    Parameters:
    - samples_list: list of numpy arrays
        List of MCMC samples for each corner plot.
    - sample_labels: list of strings
        List of labels for each set of samples.
    - name_save: string, optional
        Name of the file to save the plot. If not provided, the plot will be displayed.
    - corn_kw: dict, optional
        Additional keyword arguments to pass to the corner.corner function.
    - title: string, optional
        Title for the plot.
    - ylim: tuple, optional
        The y-axis limits for the marginalized corners.

    Returns:
    - None (if name_save is not provided) or saves the plot as a PDF file.

    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('Set1',)
    colors = [cmap(i) for i in range(n)]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Define the plot range for each dimension
    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )

    # Update corner plot keyword arguments
    corn_kw = corn_kw or {}
    corn_kw.update(range=plot_range)
    list_maxy = []
    # Create the first corner plot
    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        weights=get_normalisation_weight(len(samples_list[0]), max_len),
        **corn_kw
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))
    maxy = [axes[i, i].get_ybound()[-1] for i in range(ndim)]
    # append maxy
    list_maxy.append(maxy)
    
    # Overlay the remaining corner plots
    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **corn_kw
        )
        axes = np.array(fig.axes).reshape((ndim, ndim))
        maxy = [axes[i, i].get_ybound()[-1] for i in range(ndim)]
        # append maxy
        list_maxy.append(maxy)
    list_maxy =np.asarray(list_maxy)


    # Set y-axis limits for the marginalized corners
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes[i, i].set_ylim((0.0,np.max(list_maxy,axis=0)[i]))

    # Add legend
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=35,
        frameon=False,
        bbox_to_anchor=(0.5, ndim+1),
        loc="upper right",
        title=title,
        title_fontsize=35,
    )

    # Adjust plot layout
    plt.subplots_adjust(left=-0.1, bottom=-0.1, right=None, top=None, wspace=None, hspace=0.15)


    # Save or display the plot
    if name_save is not None:
        plt.savefig(name_save+".pdf", pad_inches=0.2, bbox_inches='tight')
    else:
        plt.show()

########################### preparation of the data #############################################
init_name = 'results_paper/mcmc_rndStart_M*_charge0.0*seed2601*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
# sort datasets by charge
datasets = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1.h5', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1.h5', ]
# create the list of injected parameters
pars_inj = ['results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.8_p8.7_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.4_e0.1_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu5.0_a0.95_p6.9_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M5e+05_mu1e+01_a0.95_p1.2e+01_e0.4_x1.0_charge0.0_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', 'results_paper/mcmc_rndStart_M1e+06_mu1e+01_a0.95_p8.3_e0.4_x1.0_charge0.0025_SNR50.0_T2.0_seed2601_nw16_nt1_injected_pars.npy', ]

print("len names", len(datasets),len(pars_inj))
cmap = plt.cm.get_cmap('Set1',)
colors = [cmap(i) for i in range(len(datasets))]#['black','red', 'royalblue']#
ls = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 3)), (0, (3, 5, 1, 5, 1))]

list_chains,labs = [], []
list_dict = []
# Scalar plot
for filename, inj_params, color in zip(datasets, pars_inj, colors):
    # Get repository name
    repo_name = inj_params.split('_injected_pars.npy')[0]
    
    # Load injected parameters
    truths = np.load(inj_params)
    
    # Load MCMC samples
    list_chains.append(np.load(repo_name + '/samples.npy'))
    # obtain median and 95% credible interval from the samples
    med = np.median(np.load(repo_name + '/samples.npy'), axis=0)
    low = np.percentile(np.load(repo_name + '/samples.npy'), 2.5, axis=0)
    high = np.percentile(np.load(repo_name + '/samples.npy'), 97.5, axis=0)

    # Parse parameters from repo_name
    params = repo_name.split('_')[3:]
    params_dict = {}
    
    for param in params:
        name_to_split = re.match(r'([a-zA-Z]+)', param).groups()[0]
        key, value = name_to_split, float(param.split(name_to_split)[1])
        params_dict[key] = value

    # labels
    label = '('

    # label += f"{params_dict.get('T')}"
    label += fr"{params_dict.get('M')/1e6}$\times 10^6$"
    if int(params_dict.get('mu'))==5:
        label += f", $\, \, \,${int(params_dict.get('mu'))}"
    else:
        label += f", {int(params_dict.get('mu'))}"
    label += f", {params_dict.get('a'):.2f}"
    label += f", {params_dict.get('e')}"
    if params_dict.get('charge') == 0.0:
        label += f", $0.0$"
    else:
        label += fr", {params_dict.get('charge')*1e3} $\times 10^{{-3}}$"
    
    label += ')'
    print(params_dict)
    # update dict with med and 95% CI
    params_dict.update({f"median":med})
    params_dict.update({f"perc2.5":low})
    params_dict.update({f"perc97.5":high})
    # append params_dict to list_dict
    list_dict.append(params_dict)
    
    labs.append(label)

########################### plot all #############################################
# create a txt and store the information in list_dict
# make a plot where on the x axis there are the parameters and on the y axis the measurement precision, each dot with different color is adifferent run of a dataset
# plt.figure()
# all_samp = np.asarray([el[:int(3e5)] for el in list_chains])
# relative_precision = np.std(all_samp,axis=1)/np.mean(all_samp,axis=1)
# relative_precision[:2] = np.std(all_samp,axis=1)[:2]
# # make a plot of the relative precision where on the x axis we have the parameters and on the y axis the relative precision
# fig, ax = plt.subplots(1,1,figsize=(246.0*px, inv_golden * 246.0*px*2))
# # plot only the intrinsic parameters
# simbols = ['o',  'X',  '^',  'd',  '*', 'P']

# indintr = np.asarray([0,1,2,3,4])
# for i in range(len(datasets)):
#     ax.plot(indintr,relative_precision[i][indintr],simbols[i], label=labs[i],ms=10,alpha=0.5)
# ax.set_yscale('log')
# ax.set_xticks(indintr)
# ax.set_xticklabels([labels[i] for i in indintr],rotation=45)
# ax.set_ylabel('Relative precision',fontsize=18)
# ax.legend(title=r'$(M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, d)$',fontsize=16,title_fontsize=18)
# plt.tight_layout()
# plt.ylim(7e-8,1e-3)
# # save the plot
# plt.savefig('./plot_paper/relative_precision.pdf')


############################################################################################################
# indintr = np.asarray([0,1,2,3,4])
for i in range(len(datasets)):
    tmp = list_chains[i][:,6:8]
    tmp[:,0] = np.arccos(tmp[:,0])
    Sigma = np.cov(tmp.T) * (180.0/(np.pi))**2 
    err_sky_loc = np.median(2*np.pi*np.sin(tmp[:,0])*np.sqrt(np.linalg.det(Sigma)) )
    #  as estimated in https://arxiv.org/pdf/2102.01708.pdf sec3.2
    errD = np.std(list_chains[i][:,5])/list_chains[i][:,5].mean()
    print(err_sky_loc,errD,err_sky_loc*np.std(list_chains[i][:,5]))
    corrcoef = np.corrcoef(list_chains[i].T)
    # plt.figure(figsize=(246.0*px*2, inv_golden * 246.0*px)); plt.plot(labels[:-1], corrcoef[-1][:-1]); plt.show()
    plt.figure(); cb = plt.imshow(corrcoef,vmin=-1, vmax=1); plt.colorbar(cb); plt.show()
    print(corrcoef[-1][:-1],"\n")

from few.trajectory.inspiral import EMRIInspiral
traj = EMRIInspiral(func="KerrEccentricEquatorialAPEX")
fig, ax = plt.subplots(1,1,figsize=(246.0*px, inv_golden * 246.0*px*2))
for i in range(len(datasets)):
    
    truths = np.load(pars_inj[i])
    inp_par = np.asarray([np.exp(truths[0]), np.exp(truths[1]), truths[2], truths[3], truths[4], 1.0, truths[-3], truths[-2], truths[-1]])
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(*inp_par[:7], T=2.0, dt=10.0)
    
    ax.loglog(Phi_phi[-1]/np.pi/2,relative_precision[i][indintr][0], simbols[i], label=labs[i],ms=10,alpha=0.5)
    
plt.legend()
plt.show()
