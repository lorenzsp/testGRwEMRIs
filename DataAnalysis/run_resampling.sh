#!/bin/bash
# print process id
echo $$
# nohup bash run_resampling.sh > out.out &
# Assign variables to each parameter
Tobs=2.0
dt=10.0
M=1e6
mu=10.0
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
dev=3
# ---------------------------------------------------------

# Execute the Python command and redirect output to the dynamic filename
python resampling.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise

# # ---------------------------------------------------------

a=0.80

# Execute the Python command and redirect output to the dynamic filename
python resampling.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise

a=0.95
# # ---------------------------------------------------------
e0=0.2

# Execute the Python command and redirect output to the dynamic filename
python resampling.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise

e0=0.4
# # ---------------------------------------------------------

M=5e5
mu=10.0

# Execute the Python command and redirect output to the dynamic filename
python resampling.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise

# ---------------------------------------------------------
M=0.5e6
mu=5

# Execute the Python command and redirect output to the dynamic filename
python resampling.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise

# ---------------------------------------------------------
Tobs=0.5
dt=2.5
M=1e5
mu=5
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

# Execute the Python command and redirect output to the dynamic filename
python resampling.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps -outname $outname -noise $noise -Tplunge 0.5
