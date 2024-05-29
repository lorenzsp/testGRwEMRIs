#!/bin/bash

# Assign variables to each parameter
Tobs=2.0
dt=10.0
M=1e6
mu=3.6
a=0.95
p0=13.0
e0=0.4
x0=1.0
charge=0.0
nwalkers=26
ntemps=1
nsteps=500000
noise=0.0
outname=MCMC
# ---------------------------------------------------------
dev=7
# Create a dynamic output filename
output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise > $output_filename &

# # ---------------------------------------------------------
# dev=1
# a=0.80
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise > $output_filename &

# a=0.95
# # ---------------------------------------------------------
# dev=2
# e0=0.2
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise > $output_filename &

# e0=0.4

# # ---------------------------------------------------------
# dev=3
# M=5e5
# mu=10.0
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise > $output_filename &

# # ---------------------------------------------------------
# dev=5

# M=0.5e6
# mu=5
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise > $output_filename &

# # ---------------------------------------------------------
# Tobs=1.0
# dt=2.5
# M=1e5
# mu=5
# a=0.95
# p0=13.0
# e0=0.4
# x0=1.0
# charge=0.0
# nwalkers=26
# ntemps=1
# nsteps=500000
# noise=0.0
# outname=MCMC

# dev=7
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise -Tplunge 1.0 > $output_filename &
# ----------------------- Vacuum ----------------------------------
# Tobs=2.0
# dt=10
# M=1e6
# mu=10.0
# a=0.95
# p0=13.0
# e0=0.4
# x0=1.0
# charge=0.0
# nwalkers=26
# ntemps=1
# nsteps=500000
# noise=0.0
# outname=vacuumMCMC

# dev=0
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise -Tplunge 2.0 -vacuum 1 > $output_filename &

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # # ----------------------- bias ----------------------------------
# Tobs=0.5
# dt=2.5
# M=1e5
# mu=5
# a=0.95
# p0=13.0
# e0=0.4
# x0=1.0
# nwalkers=26
# ntemps=1
# nsteps=500000

# charge=0.025
# dev=1
# outname=bias
# # Create a dynamic output filename
# output_filename="output_bias_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname bias -noise $noise -vacuum 1 -Tplunge 0.5 > $output_filename &

# # ------------------ Non Zero Charge ---------------------------------------
# # Assign variables to each parameter
# outname=devGR
# dev=0

# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise -vacuum 0 -Tplunge 0.5 > $output_filename &
