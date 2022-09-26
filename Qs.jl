cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DelimitedFiles
using StaticArrays
using LinearAlgebra
using Random
using Distributions
using FFTW

#
N_config=100
#longitudinal layers
Ny=100
# Transverse lattice size in one direction
#N=64
# lattice spacing
a=1 ./N
# infra regulator m
m=0.5
#
gμ=1
#
Nc=3
