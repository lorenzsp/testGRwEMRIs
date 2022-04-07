# Example code to interpolate the SSF eccentric orbit fluxes

This code is identical to the code in FEW so all the code to load and interpolate the data can be reused. The only difference it the scaling applied to Edot and Ldot

compile with:  g++ -o InterpolateEccFluxes InterpolateEccFluxes.cc Interpolant.cc -lgsl -lm

Run ./InterpolateEccFluxes