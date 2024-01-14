import glob
from eryn.backends import HDFBackend

datasets = sorted(glob.glob('results_mcmc/MCMC_new*nw16_nt3.h5'))

for filename in datasets:
    file  = HDFBackend(filename)
    print(filename)
    print(file.get_move_info())