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
    return (Qs,Rs)
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
    for nx in 1:N
        for ny in 1:N

            Ax_tmp=conj(V[nx,ny,:,:])*(conj(V[La_mod(nx+1,N),ny,:,:])-conj(V[nx,ny,:,:]))/(1im*a)
            Ay_tmp=conj(V[nx,ny,:,:])*(conj(V[nx,La_mod(ny+1,N),:,:])-conj(V[nx,ny,:,:]))/(1im*a)

            for a in 1:Ng
                A[nx,ny,1,a]=2tr(Ax_tmp*t[a])
                A[nx,ny,2,a]=2tr(Ay_tmp*t[a])
            end

        end
    end
    A
end

V_test=Wilson_line_Ny()
A_test=Gluon_field(V_test)


function Gluon_field_k(A_field)
    A_k=similar(A_field)
    for a in 1:Ng
        A_k[:,:,1,a]=fft_p*A_field[:,:,1,a]
        A_k[:,:,2,a]=fft_p*A_field[:,:,2,a]
    end
    return A_k
end

typeof(A_test)
A_test_k=Gluon_field_k(A_test)

function xG_ij(A_field)
      Ak= Gluon_field_k(A_field)
      xG_k=zeros(ComplexF32, (N,N,2,2))
      xG_r=zeros(ComplexF32, (N,N,2,2))
      for ny in 1:N
          for nx in 1:N

               xG_k[nx,ny,1,1]=sum(Ak[nx,ny,1,c]*Ak[nx,ny,1,c] for c in 1:Ng)
               xG_k[nx,ny,1,2]=sum(Ak[nx,ny,1,c]*Ak[nx,ny,2,c] for c in 1:Ng)
               xG_k[nx,ny,2,2]=sum(Ak[nx,ny,2,c]*Ak[nx,ny,2,c] for c in 1:Ng)
               xG_k[nx,ny,2,1]=sum(Ak[nx,ny,2,c]*Ak[nx,ny,1,c] for c in 1:Ng)
          end
      end

      for i in 1:2
           for j in 1:2
               xG_r[:,:,j,i]= ifft_p*xG_k[:,:,j,i]
           end
      end
      return real(xG_r./N^2)
end

xg_test=xG_ij(A_test)

function xGh(xg_ij)
    size=floor(Int,N/2)
    data_xG=zeros(Float32,size)
    data_xh=zeros(Float32,size)


    for i in 1:size
        data_xG[i]=8*pi*(xg_ij[i,1,1,1]+xg_ij[i,1,2,2])
        data_xh[i]=8*pi*(xg_ij[i,1,1,1]-xg_ij[i,1,2,2])
    end

    return  (collect(1:size)*a,data_xG,data_xh)
end

(r_t,xG_t,xh_t)=xGh(xg_test)

scatter(r_t,xh_t)



function xG(xg_ij)
    N_step=floor(Int,N/2)
    Δ_size=2*a

    xg_r= zeros(Float32,N_step)
    xh_r= zeros(Float32,N_step)
    N_r= zeros(Float32,N_step)

    for i in 1: Int(N/2), j in 1:Int(N/2)
            xg_tmp=xg_ij[i,j,1,1]+xg_ij[i,j,2,2]
            xh_tmp=-(xg_ij[i,j,1,1]+xg_ij[i,j,2,2])+2*i^2*xg_ij[i,j,1,1]/(i^2+j^2)+2*j^2*xg_ij[i,j,2,2]/(i^2+j^2)+2*i*j*(xg_ij[i,j,1,2]+xg_ij[i,j,2,1])/(i^2+j^2)
            r_index=floor(Int,sqrt(((i-1)*a)^2+((j-1)*a)^2)/Δ_size)+1
        if  r_index< N_step
            xg_r[r_index]=xg_r[r_index]+xg_tmp
            xh_r[r_index]=xh_r[r_index]+xh_tmp
            N_r[r_index]= N_r[r_index]+1
        end
    end

    for r in 1:Int(N_step)
        xG_tmp=xg_r[r]
        xg_r[r]=xG_tmp/(N_r[r]+1e-6)

        xh_tmp=xh_r[r]
        xh_r[r]=xh_tmp/(N_r[r]+1e-6)

    end

    return  (collect(1:N_step)*Δ_size,xg_r,xh_r)
end


(r_t,xG_t,xh_t)=xG(xg_test)

plot(r_t,xh_t*(8π))

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


scatter(data_xgh[1,:,1],data_xg)

scatter(data_xgh[1,:,1],data_xh)






data_dipole=zeros(128,2,N_config)
P=Progress(N_config)
Threads.@threads for i in 1:N_config
                 W_path_tmp=Wilson_line_Ny()
                (r_tmp, D_tmp)=Dr_prime(W_path_tmp,2)

                 data_dipole[:,1,i]=r_tmp
                 data_dipole[:,2,i]=D_tmp
                 next!(P)
end













#=
data_dipole=zeros(Float32,Int(N/2),2)

W_path_test=zeros(ComplexF32,N,N,Nc,Nc)
W_path_test=Wilson_line_Ny()
#save("/Users/hwd/Desktop/dijet/W_path_test.jld2", "haowu_data" ,W_path_test)
(r,D_tmp)=Dr_prime(W_path_test,2)
(r4,D_tmp4)=Dr_prime(W_path_test,4)
plot(r,D_tmp,label="haowu data, Delta=2")
plot!(r4,D_tmp4,label="haowu data, Delta=4")

(r1,D_tmp1)=Dr(W_path_test)
plot!(r,D_tmp1,label="haowu data, Delta=1")


test_dat=zeros(128,2,3)

P=Progress(3)
Threads.@threads for i in 1:3
                 W_path_tmp=Wilson_line_Ny()
                (r_tmp, D_tmp)=Dr_prime(W_path_tmp,2)

                 test_dat[:,1,i]=r_tmp
                 test_dat[:,2,i]=D_tmp
                 next!(P)
end

plot(test_dat[:,1,1],test_dat[:,2,1])
plot!(test_dat[:,1,2],test_dat[:,2,2])
plot!(test_dat[:,1,3],test_dat[:,2,3])
=#
data_dipole=zeros(128,2,N_config)
P=Progress(N_config)
Threads.@threads for i in 1:N_config
                 W_path_tmp=Wilson_line_Ny()
                (r_tmp, D_tmp)=Dr_prime(W_path_tmp,2)

                 data_dipole[:,1,i]=r_tmp
                 data_dipole[:,2,i]=D_tmp
                 next!(P)
end


#=
extra_data=zeros(128,2,3*N_config)


P=Progress(3*N_config)
Threads.@threads for i in 1:N_config
                 W_path_tmp=Wilson_line_Ny()
                (r_tmp, D_tmp)=Dr_prime(W_path_tmp,2)

                 extra_data[:,1,i]=r_tmp
                 extra_data[:,2,i]=D_tmp
                 next!(P)
end
=#

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

BOOTSTRAP(data_dipole[1,2,:],N_config)

D_bs=zeros(128,3)
#D_bs_half=zeros(128,3)
Threads.@threads for i in 1:128
    D_bs[i,1]=i*2a
    BS_tmp=BOOTSTRAP(data_dipole[i,2,:],N_config)
    D_bs[i,2]=BS_tmp[1]
    D_bs[i,3]=BS_tmp[2]
end
#=
Threads.@threads for i in 1:128
    D_bs_half[i,1]=i*2a
    BS_tmp_half =BOOTSTRAP(data_dipole[i,2,:],50)
    D_bs_half[i,2]=BS_tmp_half[1]
    D_bs_half[i,3]=BS_tmp_half[2]
end

D_ave_bs=zeros()
=#

plot(D_bs[:,1],D_bs[:,2],yerr=D_bs[:,3],label="100 config")

#plot!(D_bs_half[:,1],D_bs_half[:,2],label="50 config")

plot!(D_bs[:,1],D_bs[:,2],yerr=D_bs[:,3])


#=
function D(k)
    k2 = dot(k,k)

    No=5.15627
    F=8.59946
    Q2=5.0788031044
    p=1.74633/2.0

    return No*exp(-0.25*F*k2/(k2 + Q2)*log(k2^p + 2.718281828459045))

    k = sqrt(dot(k,k))

    #return exp(-0.25*k^2)

    if k < 100.
        return Dint(k)
    end
    return 0.0
end
=#
