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
using GellMannMatrices

# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
#now updated to include the identity matrix t[9]=I/3
include("SU3.jl")
#longitudinal layers
Ny=50
# Transverse lattice size in one direction
N=64*4
#lattice size
L=32
# lattice spacing
a=L/N
# infra regulator m
m2=(0.001)^2#1/L
#
gμ=1

#
Nc=3
Ng=Nc^2-1
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
end


# functions to compute fundamental wilson line for a single configuration
#function to compute rho_k
function rho_k()
    # set the seed for testing purpose
    # rng = MersenneTwister()
    seed=1234
    rng=MersenneTwister(seed)

    fft_p*rand(rng,Normal(0,gμ/(sqrt(Ny)*a)),N,N)# draw the color charge density from N(0,1)                                # for each layer, each point, and each color
end

rho_test=@time rho_k()

rho_test
#function to compute field A(x) for a fixed color
function compute_field!(rhok)
# Modifies the argument to return the field
    Threads.@threads for j in 1:N
        for i in 1:N
            @inbounds rhok[i,j] = a^2*rhok[i,j] / (a^2 * m2 + 4.0 * sin(π*(i-1)/N)^2 + 4.0 * sin(π*(j-1)/N)^2)
            # factor of a^2 was removed to account for the normalization of ifft next
            # ifft computes sum / (lenfth of array) for each dimension
        end
    end
    rhok[1,1] = 0.0im
    ifft!(rhok)
end

function Field!(ρ)
        #A_k=similar(ρ)
        Threads.@threads for l in 1:N
            for n in 1:N
            @inbounds ρ[n,l]=a^2*ρ[n,l]/(a^2 * m2 + 4.0 * sin(π*(n-1)/N)^2 + 4.0 * sin(π*(l-1)/N)^2) #(K2[n,l].+m2)#(2(2-cos(wave_number[n])-cos(wave_number[l]))/a^2 .+m^2 )
            end
        end
        # This is the problem 1
        ρ[1,1] = 0.0im

        #real.(ifft!(ρ))
        ifft!(ρ)
end


BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

@benchmark Field!(rho_test)

@benchmark compute_field!(rho_test)
#@btime Field()
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

W_test=@time Wilson_line()

#taking product for multiple layers
function Wilson_line_Ny()
    V=Wilson_line()
     Threads.@threads for ny in 1:Ny-1
         display(ny)
              W=Wilson_line()
           for l in 1:N, n in 1:N
              V[n,l,:,:]=W[n,l,:,:]* V[n,l,:,:]
            end
        end
      V
end

# momentum space Wilson_line
function V_k_a(V)
    V_k_a=zeros(ComplexF32,Ng,N,N)
    V_tmp_a=zeros(ComplexF32,N,N)
    for a in 1:Ng
        for i in 1:Nc, j in 1:Nc
           V_tmp_a[i,j]=2*tr(V[i,j,:,:]*t[a])
        end

         V_k_a[a,:,:]=fft_p*V_tmp_a
    end
    return V_k_a
end
#Momentum space dipole
function Dk_a(V)
       Vk=V_k_a(V)
       #Vk_prime=zeros(ComplexF32,N,N)
       D_k=zeros(ComplexF32,N,N)

       for j in 1:N, i in 1:N
            #Vk_tmp=sum(2*t[a]*tr(Vk[i,j,:,:]*t[a]) for a in 1:Ng)
            #Vk_prime[i,j,:,:]=Vk_tmp
            D_k[i,j]=sum(Vk[a,i,j]*conj(Vk[a,i,j]) for a in 1:Ng )/2Nc
       end

   return D_k
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
       D_k=zeros(Float32,N,N)

       for i in 1:N, j in 1:N

           D_k[i,j]=real(tr(V[i,j,:,:]*adjoint(V[i,j,:,:])))/Nc

       end

   return D_k
end


# coordinate space dipole
function Dr(V)
     D_k= Dk(V)
     D_x= zeros(Float32,N,N)

     D_x= ifft(D_k)*N^2

     D_r= zeros(Float32,Int(N/2))
     N_r= zeros(Int(N/2))



     for i in 1:N, j in 1:N
         r=Int(sqrt(i^2+j^b))+1

         D_r[r]= D_r[r]+D_x[i,j]
         N_r[r]= N_r[r]+1
     end

     for r in 1:N/2
         D_tmp=D_r[r]
         D_r[r]=D_tmp/N_r[r]
     end

     return
end

V_test=@time Wilson_line_Ny()

data_dipole=zeros(Float32,Int(N/2),2)

D_tmp=Dr(V_test)

for i in 1:N/2
    data_dipole[i,1]=i
    data_dipole[i,2]=
end


data_v=load("wilson_line.jld2", "V_data")

D_v=Dr(data_v)

dipole_v=zeros(Float32,Int(N/2),2)


for i in 1:Int(N/2)
    dipole_v[i,1]=i
    dipole_v[i,2]=D_v[i]
end

plot(dipole_v[3:100,1],dipole_v[3:100,2],label="vladi dipole data")


(r,s)=Dr_prime(data_v)

plot(r,s,label="vladi dipole data")



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


#=

function D_r()
         dipole_tmp=D_xy()
         dipole_r_tmp=zeros(r_size,2)
         Threads.@threads for y2 in 1:N
             for y1 in 1:N,x2 in 1:N,x1 in 1:N
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



p = Progress(1)

for i in 1:N_config
      data_tmp=D_r()
      data_dipole_64[i,:,:]=data_tmp[:,:]
      next!(p)
end


data_test=D_r()
data_test20=D_r()
scatter(data_test[:,1],data_test[:,2])


scatter!(data_dipole_64[80,:,1],data_dipole_64[62,:,2])
scatter(data_dipole_64[1,:,1],data_dipole_64[1,:,2])

data_dipole_64_BS=zeros(40,3)

for n in 1:40
    data_dipole_64_BS[n,1]=n-1
    data_tmp=data_dipole_64[:,n,2]
    tmp=BOOTSTRAP(data_tmp,N_config)

    data_dipole_64_BS[n,2]=tmp[1]
    data_dipole_64_BS[n,3]=tmp[2]
end


scatter(data_dipole_64_BS[:,1],data_dipole_64_BS[:,2],yerr=data_dipole_64_BS[:,3])

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


writedlm("Data/fundamental_dipole_64_BS.dat",data_dipole_64_BS)

save("Data/fundamental_dipole_64_raw.jld2", "D_64", data_dipole_64)

# N=128
data_dipole_128=zeros(100,r_size*2,2)
for i in 1:100
    data_dipole_128[i,:,:]=readdlm("Data/data/Dipole_$(i)_128.dat",Float64)
end

data_dipole_128_BS=zeros(80,3)


for n in 1:80
    data_dipole_128_BS[n,1]=n-1
    data_tmp=data_dipole_128[:,n,2]
    tmp=BOOTSTRAP(data_tmp,N_config)

    data_dipole_128_BS[n,2]=tmp[1]
    data_dipole_128_BS[n,3]=tmp[2]
end

scatter(data_dipole_64_BS[:,1],data_dipole_64_BS[:,2],yerr=data_dipole_64_BS[:,3], label="N=64")
scatter!(data_dipole_128_BS[:,1].*0.5,data_dipole_128_BS[:,2],yerr=data_dipole_128_BS[:,3],label="N=128")



writedlm("Data/fundamental_dipole_128_BS.dat",data_dipole_128_BS)

save("Data/fundamental_dipole_128_raw.jld2", "D_128", data_dipole_128)


f(x)=exp(-x^2/(10^2)/2)
y=LinRange(0,40,100)
Y=similar(y)
Y.=f.(y)
plot!(y,Y,label="y=exp(-(x/10)^2/2)")



# cross check
G=gellmann(Nc,skip_identity=false)

G[9]


=#
