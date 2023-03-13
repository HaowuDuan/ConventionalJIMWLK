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
using Statistics
using JLD2
using FileIO
using ProgressMeter
using BenchmarkTools

# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
#now updated to include the identity matrix t[9]=I/3
include("SU3.jl")
#longitudinal layers
const Ny=50
# Transverse lattice size in one direction
const N=64*4
#lattice size
const L=32
# lattice spacing
const a=L/N
# infra regulator m
const m2=(0.0002)^2#1/L
#
const gμ=1
#
const Nc=3
const Ng=Nc^2-1


# number of configurations
#N_config=100
# anything will be updated with new N

#For testing only, set seed to 1234



begin
    # define the correct wave number
    wave_number=fftfreq(N,2π)

    # Calculate magnitude of lattice momentum square for later use
    K2=zeros(N,N)
    for i in 1:N, j in 1:N
        K2[i,j]=2(2-cos(wave_number[i])-cos(wave_number[j]))/a^2
    end

    #calculate fft and ifft plans for later use
    rho=randn(ComplexF32, (N,N))
    fft_p=plan_fft(rho; flags=FFTW.MEASURE, timelimit=Inf)
    ifft_p=plan_ifft(rho; flags=FFTW.MEASURE, timelimit=Inf)

    seed=1234#abs(rand(Int))
    rng=MersenneTwister(seed)

end


#functions to compute fundamental wilson line for a single configuration
#function to compute rho_k
function rho_k()
    # set the seed for testing purpose
    # rng = MersenneTwister()

    fft_p*rand(rng,Normal(0,gμ/(sqrt(Ny)*a)),N,N)# draw the color charge density from N(0,1)                                # for each layer, each point, and each color
end


#function to compute field A(x) for a fixed color
function Field(ρ)

        Threads.@threads for l in 1:N
            for n in 1:N
            ρ[n,l]=ρ[n,l]/(K2[n,l].+m2)
            end
        end
        # This is the problem !!!!!!!!
        ρ[1,1] = 0.0im

      real.(ifft_p*ρ)
end

#function to compute a single layer wilson line
function Wilson_line()
     Vₓ_arg=zeros(ComplexF32, (N,N,Nc,Nc))
     Vₓ=zeros(ComplexF32, (N,N,Nc,Nc))


     for a in 1:Ng
         ρ=rho_k()
         # This is the problem 2
         A_tmp=Field(ρ)
         for l in 1:N, n in 1:N
               Vₓ_arg[n,l,:,:]=A_tmp[n,l]*t[a]+Vₓ_arg[n,l,:,:]
         end
     end


     for l in 1:N, n in 1:N
         V_tmp= Vₓ_arg[n,l,:,:]
         Vₓ[n,l,:,:].= exp(1.0im*V_tmp)
     end
      Vₓ
end

function Wilson_line_Ny()
    p=Progress(Ny)
     V=Wilson_line()
     Threads.@threads for ny in 1:Ny-1
              W=Wilson_line()
           for l in 1:N, n in 1:N
              V[n,l,:,:]=W[n,l,:,:]* V[n,l,:,:]
            end
            next!(p)
        end
      V
end


# momentum space Wilson_line
function V_k!(V)
    for a in 1:Nc, b in 1:Nc
        V_tmp=@view V[:,:,a,b]
        V[:,:,a,b]=fft_p*V_tmp
    end
    return V
end


#Momentum space dipole
function Dk(V)
       V_k!(V)
       D_k=zeros(ComplexF32,N,N)

       for i in 1:N, j in 1:N

           D_k[i,j]=tr(V[i,j,:,:]*adjoint(V[i,j,:,:]))/Nc

       end

   return D_k
end

# coordinate space dipole
function Dr(V)
     D_k= Dk(V)
     #D_x= zeros(Float32,N,N)

     D_x= ifft(D_k)/N^2

     D_r= zeros(Float32,Int(N/2))
     N_r= zeros(Float32,Int(N/2))



     for i in 1: Int(N/2), j in 1:Int(N/2)
          r=floor(Int,sqrt(i^2+j^2))+1
         if r<Int(N/2)
             D_r[r]= D_r[r]+ real(D_x[i,j])
             N_r[r]= N_r[r]+1
         end
     end

     for r in 1:Int(N/2)
         D_tmp=D_r[r]
         D_r[r]=D_tmp/N_r[r]
     end

     return  D_r
end



data_dipole=zeros(Float32,Int(N/2),2)

W_path_test=Wilson_line_Ny()

save("/Users/hwd/Desktop/dijet/W_path_test.jld2", "haowu_data" ,W_path_test)

D_tmp=Dr(W_path_test)

for i in 1:Int(N/2)
    data_dipole[i,1]=i
    data_dipole[i,2]=D_tmp[i]
end


plot(data_dipole[3:100,1],data_dipole[3:100,2],label="haowu data")


data_v=load("wilson_line.jld2", "V_data")

D_v=Dr(data_v)

dipole_v=zeros(Float32,Int(N/2),2)


for i in 1:Int(N/2)
    dipole_v[i,1]=i
    dipole_v[i,2]=D_v[i]
end

plot!(dipole_v[3:100,1],dipole_v[3:100,2],label="vladi dipole data")



function BOOTSTRAP(data,N_of_config)
     bootstrap=zeros(N_of_config)
     for i in 1:N_of_config
     BS_sample = similar(bootstrap)
     for j in 1:N_of_config
     index=rand(1:N_of_config)
     BS_sample[j]=data[index]
     end
     bootstrap[i]=mean(BS_sample)
     end

     bootstrap_MEAN = mean(bootstrap)
     bootstrap_STD = std(bootstrap)
     bootstrap_MEAN,bootstrap_STD
end
