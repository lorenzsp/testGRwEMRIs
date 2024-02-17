import glob
from eryn.backends import HDFBackend
import numpy as np
import matplotlib.pyplot as plt
import corner
import os
from few.utils.constants import *
import matplotlib as mpl

default_width = 5.78853 # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean

# mpl.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": 'pdflatex',
#     "pgf.rcfonts": False,
#     "font.family": "serif",
#     "figure.figsize": [default_width, default_width * default_ratio],
#   'legend.fontsize': 18,
#   'xtick.labelsize': 18,
#   'ytick.labelsize': 18,
# # "axes.formatter.min_exponent": 1
# "axes.formatter.offset_threshold": 10
# })

# plt.rcParams.update({
# "text.usetex": True,
# "font.family": "sans-serif",
# "font.sans-serif": ["Helvetica"]})


import matplotlib.lines as mlines

def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

labels = [r'$\ln (M/{\rm M}_\odot$)', r'$\ln (\mu / M_{\odot})$', r'$a$', r'$p_0 \, [M]$', r'$e_0$', 
            r'$D_L \, [{\rm Gpc}]$',
            r"$\cos \theta_S$",r"$\phi_S$",
            r"$\cos \theta_K$",r"$\phi_K$",
        r'$\Phi_{\varphi 0}$', r'$\Phi_{r 0}$',
            r"$d$",
        ]

CORNER_KWARGS = dict(
    labels=labels,
    bins=40,
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

def overlaid_corner(samples_list, sample_labels, name_save=None, corn_kw=None):
    """Plots multiple corners on top of each other"""
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('Set1',)
    colors = [cmap(i) for i in range(n)]#['black','red', 'royalblue']#

    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )

    corn_kw.update(range=plot_range)
    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        weights=get_normalisation_weight(len(samples_list[0]), max_len),
        **corn_kw
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **corn_kw
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=35, frameon=False,
        bbox_to_anchor=(0.5, ndim+1), 
        loc="upper right"
    )
    
#     fig.subplots_adjust(right=1.0,top=1.0)

    plt.subplots_adjust(left=-0.1, bottom=-0.1, right=None, top=None, wspace=None, hspace=0.15)

    if name_save is not None:
        plt.savefig(name_save+".pdf", pad_inches=0.2, bbox_inches='tight')
    else:
        plt.show()


def create_folder(folder_path):
    """
    Create a folder if it doesn't exist.

    Parameters:
    - folder_path (str): The path to the folder to be created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

import healpy as hp

def hp_bin(phi, theta, nside):

    npixels = hp.nside2npix(nside)
    indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    hpx_map = np.zeros(npixels, dtype=int)
    hpx_map[idx] = counts

    return hpx_map

def get_autocorr_plot(to_check,plotname):
    n_dim  = to_check.shape[-1]
    plt.figure()
    for var in range(n_dim):
        y = to_check[:,:,var].T
        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = autocorr_gw2010(y[:, :n])
            new[i] = autocorr_new(y[:, :n])

        # Plot the comparisons
        # plt.loglog(N, gw2010, "o-", label="G&W 2010")
        plt.loglog(N, new, "o-", label=labels[var])

    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    # plt.axhline(true_tau, color="k", label="truth", zorder=-100)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(plotname+'.png')

init_name = 'results_paper/mcmc_*'
datasets = sorted(glob.glob(init_name + '.h5'))
pars_inj = sorted(glob.glob(init_name + '_injected_pars.npy'))
print("len names", len(datasets),len(pars_inj))

temp=0

samp_final = []
inj_pars = []
for filename,el in zip(datasets,pars_inj):
    print('-------------------------------------')
    file  = HDFBackend(filename)
    print(filename)
    print("acceptance:")
    print(file.get_move_info()['GaussianMove_0']['acceptance_fraction'][0])
    print(file.get_move_info()['GaussianMove_1']['acceptance_fraction'][0])
    
    # burn = int(file.iteration*0.25)
    # thin = 2
    # print("iteration ", file.iteration)
    # autocorr_time = file.get_autocorr_time(discard=burn, thin=thin)['emri']
    # print("Effective sample size",(file.iteration-burn) * file.nwalkers / np.sum(autocorr_time) )
    # print("autocorrelation", autocorr_time, "\n correlation time N/50",(file.iteration-burn)/50)
    # print("max last loglike", file.get_log_like()[-1][0])
    # print(file.get_betas()[-1])
    # # print(file.get_gelman_rubin_convergence_diagnostic(discard=burn, thin=thin, doprint=True))
    
    # mask = np.arange(file.nwalkers)
    
    # # # create directory
    # repo_name = el.split('results_intrinsic/')[-1].split('rndStart_')[-1].split('_seed')[0]
    # create_folder(repo_name)
    
    # # loglike
    # ll = file.get_log_like(discard=burn, thin=thin)[:]
    # plt.figure()
    # [plt.plot(ll[:,temp,walker],'-',label=f'{walker}') for walker in mask]
    # plt.tight_layout()
    # plt.savefig(repo_name+f'/traceplot_loglike.png', bbox_inches='tight')
    
    # # chains
    # samp = file.get_chain(discard=burn, thin=thin)['emri'][:,temp,mask,...]
    # inds = file.get_inds(discard=burn, thin=thin)['emri'][:,temp,mask,...]
    # toplot = samp[inds]
    # truths = np.load(el)

    # # check autocorrelation plot
    # get_autocorr_plot(samp[:,:,0,:],repo_name+'/autocorrelation')
    
    # # check chains
    
    # for ii in range(13):
    #     plt.figure()
    #     plt.plot(toplot[:,ii])
    #     plt.xlabel(labels[ii])
    #     plt.axhline(truths[ii],color='k')
    #     plt.tight_layout()
    #     plt.savefig(repo_name+f'/traceplot_chain{ii}.png', bbox_inches='tight')
        
    #     plt.figure()
    #     plt.hist(toplot[:,ii],bins=30,density=True)
    #     plt.axvline(truths[ii],color='k')
    #     plt.xlabel(labels[ii])
    #     plt.tight_layout()
    #     plt.savefig(repo_name+f'/posterior_chain{ii}.png', bbox_inches='tight')
    
    # # alpha bound
    # mu = np.exp(toplot[:,1])
    # d = np.abs(toplot[:,-1])
    # w = mu / np.sqrt(d)
    # y = np.sqrt(2*d)*mu*MRSUN_SI/1e3
    # plt.figure()
    # plt.hist(np.log10(y), weights=w/y, bins=np.linspace(-2.0,0.5,num=40), density=True)
    # plt.tight_layout()
    # plt.xlabel(r'$\log_{10} [\sqrt{\alpha} / {\rm km} ]$',size=22)
    # vpos = 0.8
    # plt.axvline(vpos,color='k',linestyle=':',label='Current bound')
    # vpos = np.log10(0.4 * np.sqrt( 16 * np.pi**0.5 ))
    # plt.axvline(vpos,color='r',linestyle=':',label='Best bound from 3G')
    # # text_position = (vpos - 0.1, vpos)  # Adjust the position as needed
    # # plt.text(*text_position, 'Current bound', verticalalignment='center', fontsize=18, rotation='vertical')
    # # legend = plt.legend(title=r'$(T [{\rm yr}], M \, [{\rm M}_\odot], \mu \, [{\rm M}_\odot], a, e_0, T [{\rm yr}])$',framealpha=1.0,ncol=2,loc='upper left',fontsize=12)
    # # legend.get_title().set_fontsize('12')
    # plt.legend()
    # plt.savefig(repo_name+f'/alpha_bound.png', bbox_inches='tight')
    

    # CORNER_KWARGS["truths"] = truths
    
    # overlaid_corner([toplot], [''], name_save=repo_name + f'/corner_{temp}', corn_kw=CORNER_KWARGS)
    # np.save(repo_name + '/samples',toplot)
    # # plt.figure(); corner.corner(toplot, truths=truths); plt.tight_layout(); plt.savefig(repo_name + '/corner.png')
    
    # plt.close()