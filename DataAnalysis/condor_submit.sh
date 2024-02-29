#!/bin/bash
# run the following before running this: chmod +x mcmc.py
# then: bash condor_submit.sh

# Assign variables to each parameter
Tobs=2
dt=10.0
M=1e6
mu=1e1
a=0.95
p0=13.0
e0=0.4
x0=1.0
charge=0.0
nwalkers=32
ntemps=1
nsteps=500000

# ---------------------------------------------------------
dev=0
# Create a dynamic output filename
output_filename="output_lakshmi_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
condor_submit -a "Tobs=$Tobs" -a "dt=$dt" -a "M=$M" -a "mu=$mu" -a "a=$a" -a "p0=$p0" -a "e0=$e0" -a "x0=$x0" -a "charge=$charge" -a "nwalkers=$nwalkers" -a "ntemps=$ntemps" -a "nsteps=$nsteps" -a "dev=$dev" -a "output_filename=$output_filename" submit_file.submit

# # ---------------------------------------------------------
# dev=7
# a=0.80
# # Create a dynamic output filename
# output_filename="output_lakshmi_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# condor_submit -a "Tobs=$Tobs" -a "dt=$dt" -a "M=$M" -a "mu=$mu" -a "a=$a" -a "p0=$p0" -a "e0=$e0" -a "x0=$x0" -a "charge=$charge" -a "nwalkers=$nwalkers" -a "ntemps=$ntemps" -a "nsteps=$nsteps" -a "dev=$dev" -a "output_filename=$output_filename" submit_file.submit

a=0.95
# ---------------------------------------------------------
dev=4
e0=0.2
# Create a dynamic output filename
output_filename="output_lakshmi_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
condor_submit -a "Tobs=$Tobs" -a "dt=$dt" -a "M=$M" -a "mu=$mu" -a "a=$a" -a "p0=$p0" -a "e0=$e0" -a "x0=$x0" -a "charge=$charge" -a "nwalkers=$nwalkers" -a "ntemps=$ntemps" -a "nsteps=$nsteps" -a "dev=$dev" -a "output_filename=$output_filename" submit_file.submit

e0=0.4

# ---------------------------------------------------------
dev=5
Tobs=2
M=7e5
# Create a dynamic output filename
output_filename="output_lakshmi_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
condor_submit -a "Tobs=$Tobs" -a "dt=$dt" -a "M=$M" -a "mu=$mu" -a "a=$a" -a "p0=$p0" -a "e0=$e0" -a "x0=$x0" -a "charge=$charge" -a "nwalkers=$nwalkers" -a "ntemps=$ntemps" -a "nsteps=$nsteps" -a "dev=$dev" -a "output_filename=$output_filename" submit_file.submit

# ---------------------------------------------------------
dev=6
Tobs=2
M=1e6
mu=5
# Create a dynamic output filename
output_filename="output_lakshmi_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
condor_submit -a "Tobs=$Tobs" -a "dt=$dt" -a "M=$M" -a "mu=$mu" -a "a=$a" -a "p0=$p0" -a "e0=$e0" -a "x0=$x0" -a "charge=$charge" -a "nwalkers=$nwalkers" -a "ntemps=$ntemps" -a "nsteps=$nsteps" -a "dev=$dev" -a "output_filename=$output_filename" submit_file.submit
