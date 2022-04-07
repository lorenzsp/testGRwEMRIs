# Eccentric orbit flux data

## Flux data

SSF_ecc_flux.dat contains data for the scalar field fluxes for eccentric orbits. The format of the file is:

p e Edot 1-<Ft/ut>/Edot Ldot  1-<Fphi/ut>/Ldot

where Ft = F_t, Fphi = F_phi, ut = u^t, Edot is the sum of the infinity and horizon energy flux,  Ldot is the sum of the infinity and horizon angular momentym flux. The 4th and 6th columns thus provide a measure of how well the balance law holds and provides a good (relative) error on the fluxes.

The grid in the data file can be made rectangular by defining y = p - 6 -2e

## Scaled flux data

SFF_ecc_flux_scaled.dat contains data on the y=p-6-2e,e grid and with the flux data scaled such that

EdotScaled = Edot p^4
LdotScaled = Ldot p^{5/2}

The Mathematica notebook ProcesEccFluxData.nb computes the scaled flux data file.