#!/bin/bash

# Assign variables to each parameter
Tobs=2.0
dt=10.0
M=1e6
mu=1e1
a=0.95
p0=13.0
e0=0.4
x0=1.0
charge=0.0
nwalkers=16
ntemps=1
nsteps=500000

# # ------------------ Zero Noise---------------------------------------
# dev=1
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &

# # ---------------------------------------------------------
# dev=1
# a=0.80
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &

# a=0.95
# # ---------------------------------------------------------
# dev=2
# e0=0.2
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &

# e0=0.4

# # ---------------------------------------------------------
# dev=2
# M=5e5
# mu=10.0
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &

# # ---------------------------------------------------------
# dev=3
# M=1e6
# mu=5
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &

# # ---------------------------------------------------------
# dev=3
# M=0.5e6
# mu=5
# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &

# ----------------------- bias ----------------------------------
Tobs=2.0
dt=10.0
M=1e6
mu=1e1
a=0.95
p0=13.0
e0=0.4
x0=1.0
charge=0.0025
dev=3
# Create a dynamic output filename
output_filename="output_bias_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# Execute the Python command and redirect output to the dynamic filename
nohup python bias.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname bias > $output_filename &
# # ----------------------- ----------------------------------
# Tobs=2.0
# dt=10.0
# M=1e6
# mu=1e1
# a=0.95
# p0=13.0
# e0=0.4
# x0=1.0
# charge=0.01
# dev=3
# # Create a dynamic output filename
# output_filename="output_bias_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python bias.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname bias > $output_filename &

# # ------------------ Non Zero Charge ---------------------------------------
# # Assign variables to each parameter
# Tobs=2.0
# dt=10.0
# M=1e6
# mu=1e1
# a=0.95
# p0=13.0
# e0=0.4
# x0=1.0
# # 0.4 extremal bound from Fig 21 https://arxiv.org/pdf/2010.09010.pdf
# # d = 0.01040505633416082 = (sqrt_alpha/(10*MRSUN_SI/1e3))**2 / 2
# charge=0.0025
# nwalkers=16
# ntemps=1
# nsteps=500000
# dev=7

# # Create a dynamic output filename
# output_filename="output_Tobs${Tobs}_dt${dt}_M${M}_mu${mu}_a${a}_p0${p0}_e0${e0}_x0${x0}_charge${charge}_dev${dev}_nwalkers${nwalkers}_ntemps${ntemps}_nsteps${nsteps}.txt"

# # Execute the Python command and redirect output to the dynamic filename
# nohup python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname mcmc > $output_filename &
