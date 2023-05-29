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
const m2=(0.05)^2#1/L
#
const gμ=1
#
const Nc=3
const Ng=Nc^2-1


# number of configurations
N_config=10
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

    seed=abs(rand(Int))
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
        # This is the problem !!!!!!!!, okie, I know why. Things blows up when m^2 is too small.
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
         V_tmp=@view Vₓ_arg[n,l,:,:]
         Vₓ[n,l,:,:].= exp(1.0im*V_tmp)
     end
      Vₓ
end

function Wilson_line_Ny()
    # p=Progress(Ny)
     V=Wilson_line()
     Threads.@threads for ny in 1:Ny-1
              W=Wilson_line()
           for l in 1:N, n in 1:N
              V[n,l,:,:]=W[n,l,:,:]* V[n,l,:,:]
            end
            #next!(p)
        end
      V
end

# momentum space Wilson_line
function V_k(V)
    Vk=zeros(ComplexF32,N,N,Nc,Nc)
    for b in 1:Nc, a in 1:Nc
        Vk[:,:,a,b]=fft_p*@view V[:,:,a,b]
    end
    return Vk
end

#Momentum space dipole
function Dk(V)
       Vk=V_k(V)
       D_k=zeros(Float32,N,N)

       for j in 1:N, i in 1:N
           D_k[i,j]=@views real(tr(Vk[i,j,:,:]*adjoint(Vk[i,j,:,:])))/Nc
       end
   return D_k
end

#=
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
         D_r[r]=D_tmp/(N_r[r]+1e-6)
     end

     return  (collect(1:Int(N))*a,D_r)
end
=#

function Dr_prime(V,Δ)
     N_step=Int(N/Δ)
     Δ_size=Δ*a

     D_k= Dk(V)
     #D_x= zeros(Float32,N,N)

     D_x= ifft(D_k)/N^2

     D_r= zeros(Float32,N_step)
     N_r= zeros(Float32,N_step)

     for i in 1: Int(N/2), j in 1:Int(N/2)
             r_index=floor(Int,sqrt(((i-1)*a)^2+((j-1)*a)^2)/Δ_size)+1
         if  r_index< N_step
             D_r[r_index]= D_r[r_index]+ real(D_x[i,j])
             N_r[r_index]= N_r[r_index]+1
         end
     end

     for r in 1:Int(N_step)
         D_tmp=D_r[r]
         D_r[r]=D_tmp/(N_r[r]+1e-6)
     end

     return  (collect(1:N_step)*Δ_size,D_r)
end

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
     (bootstrap_MEAN,bootstrap_STD)
end


function Sat_Mom(data_diople)
          for i in 1:length(data_diople[:,1])
                  δ=data_diople[i,2]-exp(-0.5)
                 if δ<=0
                      Qs= sqrt(2)/data_diople[i,1]
                      Rs = data_diople[i,1]
                     break
                 end
          end
    return (Qs,R)
end

function La_mod(n,N)
          if n <= N
              return n
          elseif n > N
              return 1
          end
end

function Gluon_field(V)
    A=zeros(ComplexF32, (N,N,2,Ng))
    Ax_tmp=zeros(ComplexF32, (Nc,Nc))
    Ay_tmp=zeros(ComplexF32, (Nc,Nc))
    for nx in 1:N
        for ny in 1:N
           V_tmp=@view V[nx,ny,:,:]
           V_xp=@view V[mod(nx,N)+1,ny,:,:]
           V_xm=@view V[mod(nx-2,N)+1,ny,:,:]
           V_yp=@view V[nx,mod(ny,N)+1,:,:]
           V_ym=@view V[nx,mod(ny-2,N)+1,:,:]



            Ax_tmp.= adjoint(V_tmp)*(V_xp-V_tmp)/(1im*a)
            Ay_tmp.= adjoint(V_tmp)*(V_yp-V_tmp)/(1im*a)

            for a in 1:Ng
                A[nx,ny,1,a]=2tr(Ax_tmp*t[a])
                A[nx,ny,2,a]=2tr(Ay_tmp*t[a])
            end

        end
    end
    A
end

function Gluon_field_k(A_field)
    A_k=similar(A_field)
    for a in 1:Ng
        A_k[:,:,1,a]=fft_p*A_field[:,:,1,a]
        A_k[:,:,2,a]=fft_p*A_field[:,:,2,a]
    end
    return A_k
end


function xG_ij(A_field)
      Ak= Gluon_field_k(A_field)
      xG_k=zeros(ComplexF32, (N,N,2,2))
      xG_r=zeros(ComplexF32, (N,N,2,2))
      for ny in 1:N
          for nx in 1:N

               xG_k[nx,ny,1,1]=sum(Ak[nx,ny,1,c]*conj(Ak[nx,ny,1,c]) for c in 1:Ng)
               xG_k[nx,ny,1,2]=sum(Ak[nx,ny,1,c]*conj(Ak[nx,ny,2,c]) for c in 1:Ng)
               xG_k[nx,ny,2,2]=sum(Ak[nx,ny,2,c]*conj(Ak[nx,ny,2,c]) for c in 1:Ng)
               xG_k[nx,ny,2,1]=sum(Ak[nx,ny,2,c]*conj(Ak[nx,ny,1,c]) for c in 1:Ng)
          end
      end

      for i in 1:2
           for j in 1:2
               xG_r[:,:,j,i]= ifft_p*xG_k[:,:,j,i]
           end
      end
      return real(xG_r./N^2)
end


function xGh_r(xg_ij)
    size=floor(Int,N/2)
    data_xG=zeros(Float32,size)
    data_xh=zeros(Float32,size)


    for i in 1:size
        data_xG[i]=8*pi*(xg_ij[i,1,1,1]+xg_ij[i,1,2,2])
        data_xh[i]=8*pi*(xg_ij[i,1,1,1]-xg_ij[i,1,2,2])
    end

    return  (collect(1:size)*a,data_xG,data_xh)
end


function xGh(xg_ij)
    N_step=floor(Int,N/2)
    Δ_size=2*a

    xg_r= zeros(Float32,N_step)
    xh_r= zeros(Float32,N_step)
    N_r= zeros(Float32,N_step)

    for i in 1: Int(N/2), j in 1:Int(N/2)


            xg_tmp= xg_ij[i,j,1,1]+xg_ij[i,j,2,2]
            xh_tmp= -(xg_ij[i,j,1,1]+xg_ij[i,j,2,2])+2*i^2*xg_ij[i,j,1,1]/(i^2+j^2)+2*j^2*xg_ij[i,j,2,2]/(i^2+j^2)+2*i*j*(xg_ij[i,j,1,2]+xg_ij[i,j,2,1])/(i^2+j^2)
            r_index=floor(Int,sqrt(((i-1)*a)^2+((j-1)*a)^2)/Δ_size)+1
        if  r_index< N_step
            xg_r[r_index]=xg_r[r_index]+xg_tmp
            xh_r[r_index]=xh_r[r_index]+xh_tmp
            N_r[r_index]= N_r[r_index]+1
        end
    end

    for r in 1:Int(N_step)
        xG_tmp= xg_r[r]
        xg_r[r]=xG_tmp/(N_r[r]+1e-6)

        xh_tmp= xh_r[r]
        xh_r[r]=xh_tmp/(N_r[r]+1e-6)

    end

    return  (collect(1:N_step)*Δ_size,xg_r,xh_r)
end

v_test=Wilson_line_Ny()
A_test=Gluon_field(v_test)
xg_test=xG_ij(A_test)

(r_d,dip)=Dr_prime(v_test,2)

(r_t,xG_t,xh_t)=xG(xg_test)

(r_t_2,xG_t_2,xh_t_2)=xGh(xg_test)
plot(r_t_2,xG_t_2)


plot(r_t,-xh_t)
plot(r_t,xG_t)
plot(r_d,dip)



data_xgh=zeros(20,128,3)

P_gh=Progress(20)
for i in 1:20
    V_tmp=Wilson_line_Ny()
    A_tmp=Gluon_field(V_tmp)

    xg_ij_tmp=xG_ij(A_tmp)

    (r_tmp,xg_tmp,xh_tmp)=xG(xg_ij_tmp)

    data_xgh[i,:,1]=r_tmp

    data_xgh[i,:,2]=xg_tmp

    data_xgh[i,:,3]=xh_tmp

    next!(P_gh)
end

data_xg=zeros(128)

data_xh=zeros(128)

for i in 1:128
    data_xg[i]=mean(data_xgh[:,i,2])

    data_xh[i]=mean(data_xgh[:,i,3])
end


plot(data_xgh[1,1:50,1],data_xg[1:50],label="xG")

plot(data_xgh[1,1:50,1],data_xh[1:50],label="xh")
