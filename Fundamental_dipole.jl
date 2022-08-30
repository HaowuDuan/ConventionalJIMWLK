cd(@__DIR__)
using Pkg
Pkg.activate(".")


using DelimitedFiles
using StaticArrays
using LinearAlgebra
using LaTeXStrings
using Random
using Distributions
using FFTW
using Plots


# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
include("SU3.jl")


#longitudinal layers
Ny=100
# Transverse lattice size in one direction
N=64
# lattice spacing
a=1
# infra regulator m
m=2/N
#
gμ=1

# define the correct wave number
wave_number=fftfreq(N,2π)

# Calculate magnitude of lattice momentum square for later use

K2=zeros(N,N)
for i in 1:N, j in 1:N
    K2[i,j]=2(2-cos(wave_number[i])-cos(wave_number[j]))
end

#calculate fft and ifft plans for later use
rho=randn(ComplexF32, (N,N))
fft_p=plan_fft(rho; flags=FFTW.MEASURE, timelimit=Inf)
ifft_p=plan_ifft(rho; flags=FFTW.MEASURE, timelimit=Inf)

function V()

    ρₓ=rand(Normal(0,gμ/sqrt(Ny)),Ny,N,N,8) # draw the color charge density from N(0,1)
                                    # for each layer, each point, and each color

    ρₖ=randn(ComplexF32, (Ny,N,N,8)) # create a random complex matrix to store the data from FFT

    Aₖ=randn(ComplexF32, (Ny,N,N,8)) # create a random complex matrix to store the data for momemtum space field for fixed color
    Aₓ=similar(Aₖ)                   # same thing for coordinate space

    Vₓ=randn(ComplexF32, (N,N,3,3))
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
    # taking the product of all layer
    for n in 1:N, l in 1:N
         Vₓ[n,l,:,:]=exp(sum(-im*Aₓ[1,n,l,a]*t[a] for a in 1:8))
         for i in 2:Ny
             Vₓ[n,l,:,:]=exp(sum(-im*Aₓ[i,n,l,a]*t[a] for a in 1:8))*Vₓ[n,l,:,:]
         end
    end

    @time Vₓ
end
# using fft and ifft plans, keep it for later testing
function V_plan()

    ρₓ=rand(Normal(0,gμ/sqrt(Ny)),Ny,N,N,8) # draw the color charge density from N(0,1)
                                    # for each layer, each point, and each color

    ρₖ=randn(ComplexF32, (Ny,N,N,8)) # create a random complex matrix to store the data from FFT

    Aₖ=randn(ComplexF32, (Ny,N,N,8)) # create a random complex matrix to store the data for momemtum space field for fixed color
    Aₓ=similar(Aₖ)                   # same thing for coordinate space

    Vₓ=randn(ComplexF32, (N,N,3,3))
    #calculate ρₖ
    for i in 1:Ny, a in 1:8
            ρₖ[i,:,:,a]=fft_p*ρₓ[i,:,:,a]
    end
    #calculate A(k) for fixed color
    for n in 1:N, l in 1:N
        Aₖ[:,n,l,:]=ρₖ[:,n,l,:]/(K2[n,l]+m^2)
    end
    # calculate A(x). Note that A(x) is real
    for i in 1:Ny, a in 1:8
        Aₓ[i,:,:,a]=real(ifft_p*Aₖ[i,:,:,a])
    end
    # taking the product of all layer
    for n in 1:N, l in 1:N
         Vₓ[n,l,:,:]=exp(sum(-im*Aₓ[1,n,l,a]*t[a] for a in 1:8))
         for i in 2:Ny
             Vₓ[n,l,:,:]=exp(sum(-im*Aₓ[i,n,l,a]*t[a] for a in 1:8))*Vₓ[n,l,:,:]
         end
    end

    @time  Vₓ
end
