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

mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": 'pdflatex',
    "pgf.rcfonts": False,
    "font.family": "serif",
    "figure.figsize": [default_width, default_width * default_ratio],
  'legend.fontsize': 18,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
# "axes.formatter.min_exponent": 1
"axes.formatter.offset_threshold": 10
})

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
        plt.loglog(N, new, "o-", label=f"new var{var}")

    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    # plt.axhline(true_tau, color="k", label="truth", zorder=-100)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(plotname+'.png')

init_name = 'final_results/*charge*'
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
    # print(file.get_move_info())
    
    burn = int(file.iteration*0.30)
    thin = 1
    autocorr_time = file.get_autocorr_time(discard=burn, thin=thin)['emri']
    print("iteration ", file.iteration)
    print("Effective sample size",(file.iteration-burn) * file.nwalkers / np.sum(autocorr_time) )
    print("autocorrelation", autocorr_time, "\n correlation time N/50",(file.iteration-burn)/50)
    # print(file.get_gelman_rubin_convergence_diagnostic(discard=burn, thin=thin, doprint=True))
    # mask = np.arange(file.nwalkers)
    
    # # create directory
    # repo_name = el.split('results_mcmc/')[-1].split('new_')[-1].split('_seed')[0]
    # create_folder(repo_name)
    
    # # loglike
    # ll = file.get_log_like(discard=burn, thin=thin)[:-1000]
    # plt.figure()
    # [plt.plot(ll[:,temp,walker],'-',label=f'{walker}') for walker in mask]
    # plt.tight_layout()
    # plt.savefig(repo_name+f'/traceplot_loglike.png')
    
    # # chains
    # samp = file.get_chain(discard=burn, thin=thin)['emri'][:-1000,temp,mask,...]
    # inds = file.get_inds(discard=burn, thin=thin)['emri'][:-1000,temp,mask,...]
    # toplot = samp[inds]

    # # check autocorrelation plot
    # get_autocorr_plot(samp[:,:,0,:],repo_name+'/autocorrelation')
    
    # # check chains
    # truths = np.load(el)
    # for ii in range(12):
    #     plt.figure()
    #     plt.title(repo_name+f' variable {ii}')
    #     plt.plot(toplot[:,ii])
    #     plt.axhline(truths[ii],color='k')
    #     plt.tight_layout()
    #     plt.savefig(repo_name+f'/traceplot_chain{ii}.png')
        
    #     plt.figure()
    #     plt.title(repo_name+f' variable {ii}')
    #     plt.hist(toplot[:,ii],bins=30,density=True)
    #     plt.axvline(truths[ii],color='k')
    #     plt.tight_layout()
    #     plt.savefig(repo_name+f'/posterior_chain{ii}.png')
    
    # # alpha bound
    # mu = np.exp(toplot[:,1])
    # d = toplot[:,-1]
    # w = mu / np.sqrt(d)
    # y = np.sqrt(2*d)*mu*MRSUN_SI/1e3
    # plt.figure()
    # plt.hist(np.log10(y),weights=w/y, bins=40, density=True)
    # plt.tight_layout()
    # plt.savefig(repo_name+f'/alpha_bound.png')
    
    # plt.figure(); corner.corner(toplot, truths=truths); plt.tight_layout(); plt.savefig(repo_name + '/corner.png')
    
    # plt.close()