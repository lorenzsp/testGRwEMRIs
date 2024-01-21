#!/bin/bash

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
nwalkers=16
ntemps=3
nsteps=500000

# # ---------------------------------------------------------
# dev=0
# M=5e5
# mu=5.0
# # sqrt_alpha = 0.4 * np.sqrt( 16 * np.pi**0.5 )
# # d = 0.04162022533664328 = (sqrt_alpha/(5*MRSUN_SI/1e3))**2 / 2
# charge=0.04162022533664328
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps > $output_filename &

# # # ---------------------------------------------------------
# dev=1
# M=5e5
# mu=10.0
# # d = 0.01040505633416082 = (sqrt_alpha/(10*MRSUN_SI/1e3))**2 / 2
# charge=0.01040505633416082
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps > $output_filename &

# # ---------------------------------------------------------
# dev=2
# M=5e5
# mu=5.0
# charge=0.0
# # Create a dynamic output filename
# output_filename="output_noplunge_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps > $output_filename &
