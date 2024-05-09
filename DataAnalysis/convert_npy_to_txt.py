import glob
import numpy as np
import subprocess
from few.utils.constants import *
import zipfile
import os

# Get a list of all samples.npy files in the results/*/*.npy directories
npy_files = glob.glob('paper_runs/*/*.npy')

list_txt = []
# Convert each file to a .txt file
for npy_file in npy_files:
    # Load the numpy array
    array = np.load(npy_file)

    # Select samples
    Lambda = array[:,-1]
    array = array[Lambda>0.0]
    Lambda = array[:,-1]
    mu = np.exp(array[:,1])
    
    sqrt_alpha = 2*np.sqrt(2)*mu*MRSUN_SI/1e3*Lambda**(1/4)
    weights = mu * Lambda**(-3/4) * np.sqrt(2)/4
    
    charge = np.sqrt(4 * Lambda)
    weights_ch=1/np.sqrt(Lambda)
    
    headers = ' '.join([
    'ln M',
    'ln mu',
    'a',
    'p0',
    'e0',
    'DL',
    'costhetaS',
    'phiS',
    'costhetaK',
    'phiK',
    'Phivarphi0',
    'Phir0',
    'Lambda',
    'charge',
    'charge_weights',
    'sqrt_alpha',
    'sqrt_alpha_weights'
    ])

    array_to_save = np.hstack((array,charge[:,None],weights_ch[:,None],sqrt_alpha[:,None],weights[:,None]))

    # Write the array to a .txt file
    np.savetxt(npy_file.replace('.npy', '.txt'), array_to_save, header=headers)
    
    list_txt += [npy_file.replace('.npy', '.txt')]
    

# Run the zip command
subprocess.run(['zip', '-r', 'samples.zip', *list_txt])

for npy_file in npy_files:
    # now remove the .txt file
    subprocess.run(['rm', npy_file.replace('.npy', '.txt')])