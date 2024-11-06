#!/bin/bash

# source 1
nohup python search.py -delta 1e-1 -Tobs 0.50 -dt 10.0 -M 1e6 -mu 5.0 -a 0.95 -p0 13.0 -e0 0.35 -x0 1.0 -dev 7 -nwalkers 64 -nsteps 500000 -outname test -SNR 20 -noise 1.0 -window_duration 86400 > out20.out &
# nohup python search.py -dev 6 -nwalkers 32 -nsteps 500000 -outname search25 -SNR 25 -delta 1e-1 -Tobs 0.50 -dt 10.0 -M 1e6 -mu 5.0 -a 0.95 -p0 13.0 -e0 0.35 -x0 1.0 -noise 1.0 -window_duration 86400 > out25.out &
# nohup python search.py -dev 7 -nwalkers 32 -nsteps 500000 -outname search15 -SNR 15 -delta 1e-1 -Tobs 0.50 -dt 10.0 -M 1e6 -mu 5.0 -a 0.95 -p0 13.0 -e0 0.35 -x0 1.0 -noise 1.0 -window_duration 86400 > out15.out &