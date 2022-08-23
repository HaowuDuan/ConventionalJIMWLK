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

#pre-defined functions
# SU(3) generators
include("SU3.jl")

# The parameters

#longitudinal layers
Ny=6
# Transverse lattice size in one direction
N=64
# lattice spacing
a=0.1
# infra regulator m
m=0.01
# Saturation momentum for fixed coupling
Qs=1
#the coupling constant g
g_s=0.15
# the strong coupling constant
αs=g_s^2/(4*pi)
# color charge density mu^2
μ2=1

rng=Random.seed!(123)

function V(Ny,μ2,N,m,g_s,a)
    ρ_x=rand(Normal(0,sqrt(μ2)),Ny,N,N,8)
    ρ_k=randn(rng, ComplexF64, (Ny,N,N,8))


    V_momentum=randn(rng, ComplexF64, (Ny,N,N,3,3))

    for i in 1:Ny
        for a in 8
            ρ_k[i,:,:,a]=fftshift(fft(ρ_x[i,:,:,a]))
        end
    end

    for i in 1:Ny
        for kx in 1:N
            for ky in 1:N
                  V_momentum[i,kx,ky,:,:]=exp.(-im*g_s*sum(ρ_k[i,kx,ky,j]*t[j] for j in 1:8)/((sin(kx*2pi*a/(N*2)-a*pi)^2+sin(ky*2pi*a/(N*2)-a*pi)^2-a*pi)*4/a^2+m^2))
            end
         end
     end

    V_momentum
end

# Given the result of fundamental wilson_line, we define the fundamental dipole as
function dipole_f_momentum(Ny,μ2,N,m,g_s,a)
        W=V(Ny,μ2,N,m,g_s,a)
        dipole=randn(rng, ComplexF64, (N,N))
        for kx in 1:N
            for ky in 1:N
               dipole[kx,ky]=prod(tr(transpose(conj(W[i,kx,ky,:,:]))*W[i,kx,ky,:,:]) for i in 1:Ny)
           end
       end

   @time   dipole
end


tmp=V(Ny,μ2,N,m,g_s,a)


data_dipole_f=randn(rng, ComplexF64, (100,N,N))

for i in 1:100
    tmp=dipole_f_momentum(1,μ2,N,m,g_s,a)
    data_dipole_f[i,:,:]=tmp
end

data_dipole=sum(data_dipole_f[i,:,:] for i in 1:100)/100



scatter(1:64 , data_dipole[32,:])
