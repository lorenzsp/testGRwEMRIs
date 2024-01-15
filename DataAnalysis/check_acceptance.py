import glob
from eryn.backends import HDFBackend

datasets = sorted(glob.glob('results_mcmc/MCMC_new*nw16_nt3.h5'))

for filename in datasets:
    file  = HDFBackend(filename)
    print(filename)
    print(file.get_move_info())
    burn = int(file.iteration*0.25)
    thin = 1
    print("autocorrelation",file.get_autocorr_time(discard=burn, thin=thin), "\n correlation time N/50",(file.iteration-burn)/50)