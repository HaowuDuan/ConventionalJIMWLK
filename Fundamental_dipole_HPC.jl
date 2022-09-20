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

# lattice spacing
a=32/N
# infra regulator m
m=1/32
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
rho=randn(ComplexF32, (N,N))
fft_p=plan_fft(rho; flags=FFTW.MEASURE, timelimit=Inf)
ifft_p=plan_ifft(rho; flags=FFTW.MEASURE, timelimit=Inf)

# functions to compute fundamental wilson line for a single configuration

    #function to compute rho_k
function rho_k()
        fft(rand(Normal(0,gμ/sqrt(Ny)),N,N))# draw the color charge density from N(0,1)
                                                # for each layer, each point, and each color
end

    #function to compute field A(x) for a fixed color
function Field()
            ρₖ=rho_k()
            A_k=similar(ρₖ)
            for n in 1:N, l in 1:N
                A_k[n,l]=ρₖ[n,l]/(K2[n,l].+m^2)
            end
            real(ifft(A_k))
end

    #function to compute a single layer wilson line
function Wilson_line()
         Vₓ_arg=randn(ComplexF32, (N,N,3,3))
         Vₓ=randn(ComplexF32, (N,N,3,3))
         @Threads.threads for a in 1:8
             tmp_A=Field()
              if a ==1
                for n in 1:N, l in 1:N
                    Vₓ_arg[n,l,:,:]=-im*tmp_A[n,l]*t[a]
                end
             else
                 for n in 1:N, l in 1:N
                   Vₓ_arg[n,l,:,:]=-im*tmp_A[n,l]*t[a]+Vₓ_arg[n,l,:,:]
                 end
             end
         end

         for n in 1:N, l in 1:N
             Vₓ[n,l,:,:]=exp(Vₓ_arg[n,l,:,:])
         end
         Vₓ
end


    #taking product for multiple layers
function Wilson_line_Ny()
        V=Wilson_line()
        Threads.@threads for ny in 2:Ny
                  W=Wilson_line()
               for n in 1:N, l in 1:N
                  V[n,l,:,:]=W[n,l,:,:]*V[n,l,:,:]
                end
            end
          V
end

dipole_tmp=zeros(N,N,N,N)
r_size=40*(N÷64)
dipole_r_tmp=zeros(r_size,2)
# for data storage
data_dipole_64=zeros(100,r_size,2)


function r(x,y)
    x=@SVector[x[1],x[2]]
    y=@SVector[y[1],y[2]]

    r=sqrt(dot(x-y,x-y))÷1
    Int(r)
end

function D_xy()
     V_tmp=Wilson_line_Ny()
      for x1 in 1:N,x2 in 1:N,y1 in 1:N,y2 in 1:N
            dipole_tmp[x1,x2,y1,y2]=real(tr(conj(transpose(V_tmp[x1,x2,:,:]))*V_tmp[y1,y2,:,:]))/Nc
      end
      dipole_tmp
end

function D_r()
    dipole_tmp=D_xy()
    Threads.@threads for x1 in 1:N
        for x2 in 1:N,y1 in 1:N,y2 in 1:N
          x=[x1,x2]
          y=[y1,y2]
          i=r(x,y)

          if i<r_size
               dipole_r_tmp[i+1,2]=dipole_r_tmp[i+1,2]+dipole_tmp[x1,x2,y1,y2]
               dipole_r_tmp[i+1,1]=dipole_r_tmp[i+1,1]+1
          end
       end
    end

    Threads.@threads for i in 1:r_size
            dipole_r_tmp[i,2]=dipole_r_tmp[i,2]/dipole_r_tmp[i,1]
            dipole_r_tmp[i,1]=i-1
    end
    dipole_r_tmp
end

D_data=D_r()

println(D_data)

flush(stdout)
