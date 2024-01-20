#!/bin/bash

# Assign variables to each parameter
Tobs=2
dt=10.0
M=0.5e6
mu=5.0
a=0.95
p0=13.0
e0=0.4
x0=1.0
charge=0.001
# 0.003651653199713658
nwalkers=8
ntemps=3
nsteps=500000

# ---------------------------------------------------------
dev=1
# Create a dynamic output filename
output_filename="BFoutput_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
nohup python mcmc_BF.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps > $output_filename &