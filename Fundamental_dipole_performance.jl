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
# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
include("SU3.jl")



#longitudinal layers
Ny=100
# Transverse lattice size in one direction
N=128
# lattice spacing
a=32/N
# infra regulator m
m=1/32
#
gμ=1

#
Nc=3
# number of configurations
N_config=100
# anything will be updated with new N
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
end

# functions to compute fundamental wilson line for a single configuration
begin
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
         for a in 1:8
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
         for ny in 2:Ny
                  W=Wilson_line()
               for n in 1:N, l in 1:N
                  V[n,l,:,:]=W[n,l,:,:]*V[n,l,:,:]
                end
            end
          V
    end
end
# functions to compute dipole correlators for a single configuration
begin

    dipole_tmp=zeros(N,N,N,N)
    r_size=40*(N÷64)
    dipole_r_tmp=zeros(r_size,2)
    # for data storage
    #data_dipole_64=zeros(100,r_size,2)
    data_dipole_128=zeros(100,r_size,2)

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
         for x1 in 1:N
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

         for i in 1:r_size
                dipole_r_tmp[i,2]=dipole_r_tmp[i,2]/dipole_r_tmp[i,1]
                dipole_r_tmp[i,1]=i-1
        end
        dipole_r_tmp
   end
end



# now we compute the dipoles for different N
p = Progress(N_config)

 for i in 1:10
      data_tmp=D_r()
      data_dipole_64[i,:,:]=data_tmp[:,:]
     # next!(p)
end

for i in 1:10
     data_tmp=D_r()
     data_dipole_128[i,:,:]=data_tmp[:,:]
    # next!(p)
end

scatter()

scatter(data_dipole_64[8,:,1],data_dipole_64[8,:,2])
scatter(data_dipole_64[1,:,1],data_dipole_64[1,:,2])





function bootstrap_arr(db,M)
     bs=[]
     for i in 1:M
     dbBS = []
     for j in 1:length(db)
     idx=rand(1:length(db))
     push!(dbBS, db[idx])
     end
     push!(bs,mean(dbBS))
     end

     MEAN = mean(bs)
     STD = std(bs)
     (MEAN,STD)
end
