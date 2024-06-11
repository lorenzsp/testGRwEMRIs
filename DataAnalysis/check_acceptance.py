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
            r"$\Lambda$",
            "loglike"
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

init_name = './results/MCMC_noise0.0_M3.6e+04_mu3.6_a0.95_p2.5e+01_e0.4_x1.0_charge0.0_SNR11.0_T0.5_seed2601_nw26_nt1*'
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
    print("iteration",file.iteration)
    print("swaps:",file.swaps_accepted/file.iteration)
    print("acceptance:")
    print(file.get_move_info())
    
    burn, thin = int(file.iteration*0.2), 1
    # burn,thin = file.get_autocorr_thin_burn()
    autocorr_time = file.get_autocorr_time(discard=burn, thin=thin)['emri']
    print("autocorrelation", autocorr_time, "\n correlation time N/50",(file.iteration-burn)/50)

    # # ------ create dir ----------
    repo_name = el.split('_injected_pars.npy')[0]
    create_folder(repo_name)
    
    # --- select based on loglike ---
    ll = file.get_log_like(discard=burn, thin=thin)[:,temp]
    # mask based on median like
    # mask = np.delete(np.arange(file.nwalkers),[4,9])
    # mask = (ll[-1]<10.0)
    mask = (ll[-1]>np.max(ll[-1])-20)
    print("maximum likelihood",ll)
    
    # ------ Import samples ----------
    truths = np.load(el)
    maxind = int(file.iteration*file.nwalkers*file.ntemps*0.95)
    samp = file.get_chain(discard=burn, thin=thin)['emri'][:,temp,mask,...]
    inds = file.get_inds(discard=burn, thin=thin)['emri'][:,temp,mask,...]
    toplot = samp[inds]
    ll_plot = file.get_log_like(discard=burn, thin=thin)[:,temp,mask].flatten()
    
    # ------ Covariance Evolution ----------
    samp_cov = np.cov(toplot,rowvar=False) * 2.38**2 / 13
    # # create a plot to investigate the stability of the covariance matrix as a function of the iteration number
    it_cov_ev = range(100, samp.shape[0], 100)
    cov_evolution = np.asarray([np.diag(np.cov(samp[:i].reshape(-1, samp.shape[-1]), rowvar=False) * 2.38**2 / 13 ) for i in it_cov_ev])
    # normalize cov_evolution by the first element
    # curent_cov = np.diag(np.load(repo_name+'_covariance.npy'))
    cov_evolution /= cov_evolution[0]

    plt.figure()
    [plt.plot(it_cov_ev,cov_evolution[:,ii],label=labels[ii]) for ii in range(cov_evolution.shape[1])]
    plt.xlabel('iteration')
    plt.ylabel('normalized diagonal element of covariance matrix')
    plt.legend()
    plt.tight_layout()
    plt.savefig(repo_name+'/covariance_trace.png')
    
    # np.save(repo_name+'_covariance.npy', samp_cov) 
    # np.save(repo_name + '_samples',toplot)
    
    # ------ Log like ----------
    # detCov = np.linalg.det(np.cov(toplot,rowvar=False))
    ndim = toplot.shape[-1]
    exp_loglike = -0.5*(ndim)
    ll = file.get_log_like(discard=burn, thin=thin)[:]
    plt.figure()
    [plt.plot(ll[:,temp,walker],'-',label=f'{walker}') for walker in np.arange(file.nwalkers)]
    # plt.axhline(exp_loglike,color='k',linestyle='--',label='Expected loglike')
    plt.tight_layout()
    # plt.legend()
    plt.savefig(repo_name+f'/traceplot_loglike.png', bbox_inches='tight')
    
    
    # # ------ autocorrelation plot ----------
    get_autocorr_plot(samp[:,:,0,:],repo_name+'/autocorrelation')
    
    # ------ trace plot ----------
    # check chains
    for ii in range(samp.shape[-1]):
        plt.figure()
        plt.plot(samp[:,:,0,ii])
        plt.xlabel(labels[ii])
        plt.axhline(truths[ii],color='k')
        plt.tight_layout()
        plt.savefig(repo_name+f'/traceplot_chain{ii}.png', bbox_inches='tight')
    
    CORNER_KWARGS["truths"] = np.append(truths,0.0)
    
    overlaid_corner([np.hstack((toplot,ll_plot[:,None]))], [''], name_save=repo_name + f'/corner_{temp}', corn_kw=CORNER_KWARGS)
    np.save(repo_name + '/samples',toplot)
    

    plt.close()