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
#
Nc=3
# define the correct wave number
wave_number=fftfreq(N,2π)
# number of configurations
N_config=10
# Calculate magnitude of lattice momentum square for later use
K2=zeros(N,N)
for i in 1:N, j in 1:N
    K2[i,j]=2(2-cos(wave_number[i])-cos(wave_number[j]))
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

# the next step is to compute coordinate space dipole
# first, we define the lattice distance r, the input is the lattice size
function r(x,y)
    x=@SVector[x[1],x[2]]
    y=@SVector[y[1],y[2]]

    r=sqrt(dot(x-y,x-y))÷1
    Int(r)
end

dipole=zeros(N,N,N,N)
V_test=V()
for x1 in 1:N,x2 in 1:N,y1 in 1:N,y2 in 1:N
    dipole[x1,x2,y1,y2]=real(tr(conj(transpose(V_test[x1,x2,:,:]))*V_test[y1,y2,:,:]))/Nc
end

# calculate d(r) from single configuraton

dipole_r=zeros(2,40)

for x1 in 1:N,x2 in 1:N,y1 in 1:N,y2 in 1:N
    x=[x1,x2]
    y=[y1,y2]
    i=r(x,y)

    if i<40
       dipole_r[1,i]=dipole_r[1,i]+dipole[x1,x2,y1,y2]
       dipole_r[2,i]=dipole_r[2,i]+1
    end
end

scatter(1:40,dipole_r[1,:]./dipole_r[2,:])

# do the calculation in momentnum space, namely do fft on the V_x to get V_k
Vₖ=fft(V_test)
# in momentum space, the dipole is localized, define dipole_k
dipole_k=zeros(2,31)
momentum=LinRange(0,30,31)

for k1 in 1:N,k2 in 1:N

    i=Int(sqrt(K2[k1,k2])÷wave_number[2])

    if i<30
       dipole_k[1,i+1]=dipole_k[1,i+1]+real(tr(conj(transpose(Vₖ[k1,k2,:,:]))*Vₖ[k1,k2,:,:]))/Nc
       dipole_k[2,i+1]=dipole_k[2,i+1]+1
    end
end

scatter(momentum,dipole_k[1,:]./(dipole_k[2,:]*N^2)*wave_number[2]^2 .*(momentum[:]).^2 )



# extrac the same dipole_k and dipole_k but from multiple configurations.
# created matrices to store data for each configuration in order to extract error bar using bootstrap

dipole_k=zeros(2,31,N_config)
dipole_r=zeros(2,31,N_config)

Threads.@threads @time for j in 1:N_config
      #create a configuration and also prepare the momentum space Wilson line
      Vₓ_tmp=V()
      Vₖ_tmp=fft(Vₓ_tmp)
      # Define temporary matrix to store inter-mediate step data
      dipole_tmp=zeros(N,N,N,N)
      dipole_r_tmp=zeros(2,31)
      dipole_k_tmp=zeros(2,31)


      for x1 in 1:N,x2 in 1:N,y1 in 1:N,y2 in 1:N
          dipole_tmp[x1,x2,y1,y2]=real(tr(conj(transpose(Vₓ_tmp[x1,x2,:,:]))*Vₓ_tmp[y1,y2,:,:]))/Nc
      end
      # calculate coordinate space dipole for a single configuration
      for x1 in 1:N,x2 in 1:N,y1 in 1:N,y2 in 1:N
          x=[x1,x2]
          y=[y1,y2]
          i=r(x,y)

          if i<30
             dipole_r_tmp[1,i+1]=dipole_r_tmp[1,i+1]+dipole_tmp[x1,x2,y1,y2]
             dipole_r_tmp[2,i+1]=dipole_r_tmp[2,i+1]+1
          end

      end

      # calculate momentum space dipole for a single configuration
      for k1 in 1:N,k2 in 1:N

          i=Int(sqrt(K2[k1,k2])÷wave_number[2])

          if i<30
             dipole_k_tmp[1,i+1]=dipole_k_tmp[1,i+1]+real(tr(conj(transpose(Vₖ_tmp[k1,k2,:,:]))*Vₖ_tmp[k1,k2,:,:]))/Nc
             dipole_k_tmp[2,i+1]=dipole_k_tmp[2,i+1]+1
          end
      end

      #
       dipole_k[1,:,j]=dipole_k_tmp[1,:]./dipole_k_tmp[2,:]
       dipole_r[1,:,j]=dipole_r_tmp[1,:]./dipole_r_tmp[2,:]
end


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
