using OptimalControl
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
using MINPACK
using ForwardDiff
using LinearAlgebra

## Problem definition

Tmax = 60                                  # Maximum thrust (Newtons)
cTmax = 3600^2 / 1e6;
T = Tmax * cTmax;     # Conversion from Newtons to kg x Mm / h²
mass0 = 1500                               # Initial mass of the spacecraft
β = 1.42e-02                               # Engine specific impulsion
μ = 5165.8620912                           # Earth gravitation constant
P0 = 11.625                                # Initial semilatus rectum
ex0, ey0 = 0.75, 0                         # Initial eccentricity
hx0, hy0 = 6.12e-2, 0                      # Initial ascending node and inclination
L0 = π                                     # Initial longitude
Pf = 42.165                                # Final semilatus rectum
exf, eyf = 0, 0                            # Final eccentricity
hxf, hyf = 0, 0                            # Final ascending node and inclination   

