cd(@__DIR__)


using DelimitedFiles
using StaticArrays
using LinearAlgebra
using Random
using Distributions
using FFTW
using JLD2
using FileIO



# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
include("SU3.jl")


N_config=parse(Float64,ARGS[1])
N=parse(Float64,ARGS[2])

#longitudinal layers
Ny=100
# Transverse lattice size in one direction
#N=64
# lattice spacing
a=1/N
# infra regulator m
m=0.5
#
gμ=1
#
Nc=3

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

function V()

    ρₓ=rand(Normal(0,gμ/sqrt(Ny)),Ny,N,N,8) # draw the color charge density from N(0,1)
                                    # for each layer, each point, and each color

    ρₖ=randn(ComplexF64, (Ny,N,N,8)) # create a random complex matrix to store the data from FFT

    Aₖ=randn(ComplexF64, (Ny,N,N,8)) # create a random complex matrix to store the data for momemtum space field for fixed color
    Aₓ=similar(Aₖ)                   # same thing for coordinate space

    Vₓ=randn(ComplexF64, (N,N,3,3))
    #calculate ρₖ
    for i in 1:Ny, a in 1:8
            ρₖ[i,:,:,a]=fft(ρₓ[i,:,:,a])
    end
    #calculate A(k) for fixed color
    for n in 1:N, l in 1:N
        Aₖ[:,n,l,:]=ρₖ[:,n,l,:]/(K2[n,l]+m^2)
    end
    # calculate A(x). Note that A(x) is real
    for i in 1:Ny, a in 1:8
        Aₓ[i,:,:,a]=real(ifft(Aₖ[i,:,:,a]))
    end
    # taking the product of all layers
    for n in 1:N, l in 1:N
         Vₓ[n,l,:,:]=exp(sum(-im*Aₓ[1,n,l,a]*t[a] for a in 1:8))
         for i in 2:Ny
             Vₓ[n,l,:,:]=exp(sum(-im*Aₓ[i,n,l,a]*t[a] for a in 1:8))*Vₓ[n,l,:,:]
         end
    end

    Vₓ
end
# using fft and ifft plans, keep it for later testing

F_Wilson_line=V()

#reshape the data

F_data=randn(ComplexF64, (N^2,9))




println(F_data)

flush(stdout)
