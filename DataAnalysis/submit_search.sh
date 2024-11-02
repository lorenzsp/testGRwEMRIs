#!/bin/bash

# Define the array of window_duration values
window_durations=(86400)

# Define the array of GPU devices
gpus=(6)

# Loop over each window_duration value and submit the job
for i in "${!window_durations[@]}"; do
    window_duration=${window_durations[$i]}
    gpu=${gpus[$i]}
    nohup python search.py -delta 1e-1 -Tobs 0.5 -dt 10.0 -M 1e6 -mu 10.0 -a 0.95 -p0 13.0 -e0 0.4 -x0 1.0 -dev $gpu -nwalkers 16 -nsteps 10000 -outname search -SNR 30.0 -noise 1.0 -window_duration $window_duration > out_abs_$window_duration.out &
done
