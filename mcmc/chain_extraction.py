import numpy as np
import matplotlib.pyplot as plt
import h5py
from lisatools.sampling.utility import ModifiedHDFBackend
import corner
import argparse
parser = argparse.ArgumentParser(description='MCMC few')
parser.add_argument('-filename','--filename', help='File from mcmc', required=True, type=str)
# parser.add_argument('-outputname','--outputname', help='Output File', required=True, type=str)
args = vars(parser.parse_args())

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


# import chains
# filename ='m_n_snr_50_no_noise_1000000.0_50.0_0.9_15.482608152920559_0.0_2.0_0.0_T4.0.h5'
filename= args['filename']
back = ModifiedHDFBackend(filename)
chains = back.get_chain()
loglike = back.get_log_prob()
# number of variables
n = chains.shape[2]
n_dim = chains.shape[-1]
temp =0

burnin = 1000
# select chains
good_chains = []
plt.figure()
for i in range(0,n):
    if np.mean(loglike[burnin:,temp,i])>-10.0:
        print(i, np.mean(loglike[burnin:,temp,i]))
        good_chains.append(i)
        plt.plot(loglike[burnin:,temp,i],label=str(i))
plt.savefig(filename + "loglike_chains.pdf" )

samp = np.array([chains[burnin:,temp,good_chains,variable].flatten() for variable in range(n_dim)]).T

np.save(filename + "_samples", samp)

# check autocorrelation
plt.figure()
for var in range(n_dim):
    y = chains[burnin:,temp,good_chains,var].T

    print(y.shape)
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
plt.legend(fontsize=14);
plt.savefig(filename+ "autocorrelation.pdf")

# figure = corner.corner(samp)
# plt.savefig('corner.pdf',bbox_inches='tight')

name_txt ='lnmu_charge_'+filename+'.dat' 
with open(name_txt, 'w') as f:
    for i in range(samp.shape[0]):
        f.write(str(samp[i,1]) + '\t' + str(samp[i,-1]) + '\n')