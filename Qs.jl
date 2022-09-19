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

load("Data/Wilson_JLD/V_1_64.jld2","FWL_64")
load("Data/Wilson_JLD/V_1_128.jld2","FWL_128")
load("Data/Wilson_JLD/V_1_256.jld2","FWL_128")
load("Data/Wilson_JLD/V_1_512.jld2","FWL_512")

# Create space in Memory to store the data for dipole
D_64=randn(ComplexF64, (100,64))
D_128=randn(ComplexF64, (100,64))
D_256=randn(ComplexF64, (100,256))
D_512=randn(ComplexF64, (100,512))



function dipole(N,data)

    # define the correct wave number
    wave_number=fftfreq(N,2π)

    # Calculate magnitude of lattice momentum square for later use

    K2=zeros(N,N)
    for i in 1:N, j in 1:N
        K2[i,j]=2(2-cos(wave_number[i])-cos(wave_number[j]))/a^2
    end

    #calculate fft and ifft plans for later use
    rho=randn(ComplexF64, (N,N))
    fft_p=plan_fft(rho; flags=FFTW.MEASURE, timelimit=Inf)
    ifft_p=plan_ifft(rho; flags=FFTW.MEASURE, timelimit=Inf)

    






end
