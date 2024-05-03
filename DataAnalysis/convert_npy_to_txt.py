import glob
import numpy as np
import subprocess

# Get a list of all samples.npy files in the results/*/*.npy directories
npy_files = glob.glob('results/*/*.npy')

# Convert each file to a .txt file
for npy_file in npy_files:
    # Load the numpy array
    array = np.load(npy_file)

    # Convert the array to float32
    # array = array.astype('float32')

    # Write the array to a .txt file
    np.savetxt(npy_file.replace('.npy', '.txt'), array)

# Run the zip command
subprocess.run(['zip', '-r', 'results.zip', 'results/'])