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
m2=(0.1)^2#1/L
#
gμ=1.45
#
const Nc=3
const Ng=Nc^2-1


# number of configurations
#N_config=10
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

     R=collect(1:N_step)*Δ_size
     pushfirst!(R,0)
     pushfirst!(D_r,1)

     return  (R,D_r)
end


function K(i,j)
    [(2/a)* sin(wave_number[i]/2), (2/a)* sin(wave_number[j]/2)]/(K2[i,j]+m2)
end

function ξ_generator()
     ξ_tmp=rand(rng,Normal(0,1),N,N,2,Ng)
end

function ξ_left(ξ_data, V_input)
     ξ_tmp=zeros(ComplexF32,N,N,2,Nc,Nc)
     Threads.@threads for i in 1:2
         for n in 1:N
              for m in 1:N
                   V_tmp=@view V_input[m,n,:,:]
                   V_dagger=adjoint(V_tmp)
                   ξ_tmp[m,n,i,:,:]=sum(ξ_data[m,n,i,a]*(V_tmp*t[a]*V_dagger) for a in 1:Ng)
              end

          end
      end
     return  ξ_tmp
end

function ξ_right(ξ_data,V_input)
    ξ_tmp=zeros(ComplexF32,N,N,2,Nc,Nc)
    Threads.@threads for i in 1:2
        for n in 1:N
             for m in 1:N
                  ξ_tmp[m,n,i,:,:].=sum(ξ_data[m,n,i,a]*t[a] for a in 1:Ng)
             end

         end
     end
    return ξ_tmp
end

function convolution(ξ_local)
    ξ_tmp_k=similar(ξ_local)
    ξ_tmp_k_prime=zeros(ComplexF32,N,N,Nc,Nc)
    ξ_tmp_x=zeros(ComplexF32,N,N,Nc,Nc)
    Threads.@threads for i in 1:2
        for j in 1:Nc
                for k in 1:Nc
                   ξ_tmp_k[:,:,i,j,k]=fft_p*ξ_local[:,:,i,j,k]
                end
        end
    end

    Threads.@threads for n in 1:N
        for m in 1:N
            k_tmp=K(m,n)
            ξ_tmp_k_prime[m,n,:,:]=sum(ξ_tmp_k[m,n,i,:,:]*k_tmp[i] for i in 1:2)
        end
    end

    Threads.@threads for j in 1:Nc
            for k in 1:Nc
               ξ_tmp_x[:,:,j,k]=ifft_p*ξ_tmp_k_prime[:,:,j,k]
            end
    end


    return  ξ_tmp_x
end

function exp_left(ξ_data, V_input, step)
     ξ_local=ξ_left(ξ_data, V_input)
     ξ_x=zeros(ComplexF32,N,N,Nc,Nc)
     e_left=similar(ξ_x)

     ξ_x=convolution(ξ_local)

     Threads.@threads for m in 1:N
         for n in 1:N
             e_left[n,m,:,:]=exp(2sqrt(step*αₛ)*ξ_x[n,m,:,:])
         end
     end

     return e_left
end


function exp_right(ξ_data, V_input, step)
    ξ_local=ξ_right(ξ_data, V_input)
    ξ_x=zeros(ComplexF32,N,N,Nc,Nc)
    e_right=similar(ξ_x)

    ξ_x=convolution(ξ_local)

    Threads.@threads for m in 1:N
        for n in 1:N
            e_right[n,m,:,:]=exp(-2sqrt(step*αₛ)*ξ_x[n,m,:,:])
        end
    end
    return e_right
end

function JIMWLK(Δ, N_of_steps, V_input)
      Y_f=N_of_steps*Δ

      V_f=zeros(ComplexF32,N,N,Nc,Nc)

      ξ_tmp=ξ_generator()

      factor_left=exp_left(ξ_tmp/a, V_input, 0.1)
      factor_right=exp_right(ξ_tmp/a, V_input, 0.1)

      for n in 1:N
          for m in 1:N
               V_tmp=@view V_input[m,n,:,:]
               left_tmp=@view factor_left[m,n,:,:]
               right_tmp=@view factor_right[m,n,:,:]
               W_tmp= left_tmp*V_tmp*right_tmp
               V_f[m,n,:,:].=W_tmp
          end
      end

      Threads.@threads for i in 1:N_of_steps-1
          ξ=ξ_generator()

          factor_left=exp_left(ξ, V_f, Δ)
          factor_right=exp_right(ξ, V_f, Δ)

          for n in 1:N
              for m in 1:N
                  V_tmp=@view V_f[m,n,:,:]
                  left_tmp=@view factor_left[m,n,:,:]
                  right_tmp=@view factor_right[m,n,:,:]
                  W_tmp= left_tmp*V_tmp*right_tmp
                  V_f[m,n,:,:].=W_tmp
              end
          end
      end

     V_f
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

function Sat_Mom(r,D)
          Qs=zeros(1)
          Rs=zeros(1)
          for i in 1:length(r)
                  δ=D[i]-exp(-0.5)
                 if δ<=0
                      Qs = sqrt(2)/r[i]
                      Rs = r[i]
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

function Gluon_field_center(V)
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



            Ax_tmp.= adjoint(V_tmp)*(V_xp-V_xm)/(2im*a)
            Ay_tmp.= adjoint(V_tmp)*(V_yp-V_ym)/(2im*a)

            for a in 1:Ng
                A[nx,ny,1,a]=2tr(Ax_tmp*t[a])
                A[nx,ny,2,a]=2tr(Ay_tmp*t[a])
            end

        end
    end
    A
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

      xG1_k=zeros(ComplexF32, N,N)
      xh1_k=zeros(ComplexF32, N,N)

      xG1_r=zeros(ComplexF32, N,N)
      xh1_r=zeros(ComplexF32, N,N)

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

      for ny in 1:N
          for nx in 1:N
               tmp_k=@view xG_k[nx,ny,:,:]
               tmp_r=@view xG_r[nx,ny,:,:]

               eigen_values_k=eigvals(tmp_k)
               eigen_values_r=eigvals(tmp_r)

               xG1_k[nx,ny]=eigen_values_k[1]+eigen_values_k[2]
               xh1_k[nx,ny]=abs(eigen_values_k[1]-eigen_values_k[2])

               xG1_r[nx,ny]=eigen_values_r+eigen_values_r[2]
               xh1_r[nx,ny]=abs(eigen_values_k[1]-eigen_values_k[2])

          end
      end



      return (xG1_k,xh1_k,xG1_k,xh1_k,)
end



#=
function xGh_vs(A)
    size=floor(Int,N/2)
    xG=zeros(Float32,size)
    xh=zeros(Float32,size)
    xG2=zeros(Float32,size)
    xh2=zeros(Float32,size)

    for r = 0:N÷2
        for j in 1:N
            for i in 1:N

                @inbounds xG[r+1]  = xG[r+1] + sum(real(A[i,j,1,b]*A[mod(i+r-1,N)+1,j,1,b]) for b in 1:Nc^2-1)/(N^2)
                @inbounds xG[r+1]  = xG[r+1] + sum(real(A[i,j,2,b]*A[mod(i+r-1,N)+1,j,2,b]) for b in 1:Nc^2-1)/(N^2)
                @inbounds xh[r+1]  = xh[r+1] + sum(real(A[i,j,1,b]*A[mod(i+r-1,N)+1,j,1,b]) for b in 1:Nc^2-1)/(N^2)
                @inbounds xh[r+1]  = xh[r+1] - sum(real(A[i,j,2,b]*A[mod(i+r-1,N)+1,j,2,b]) for b in 1:Nc^2-1)/(N^2)

                @inbounds xG2[r+1]  = xG2[r+1] + sum(real(A[i,j,1,b]*A[i,mod(j+r-1,N)+1,1,b]) for b in 1:Nc^2-1)/(N^2)
                @inbounds xG2[r+1]  = xG2[r+1] + sum(real(A[i,j,2,b]*A[i,mod(j+r-1,N)+1,2,b]) for b in 1:Nc^2-1)/(N^2)
                @inbounds xh2[r+1]  = xh2[r+1] - sum(real(A[i,j,1,b]*A[i,mod(j+r-1,N)+1,1,b]) for b in 1:Nc^2-1)/(N^2)
                @inbounds xh2[r+1]  = xh2[r+1] + sum(real(A[i,j,2,b]*A[i,mod(j+r-1,N)+1,2,b]) for b in 1:Nc^2-1)/(N^2)
            end
        end
    end

    return  (collect(1:size)*a,xG,xh,xG2,xh2)
end
=#

function xGh_hd(xg_ij,Δ)
    N_step=floor(Int,N/2)
    Δ_size=Δ*a

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

    R=collect(1:N_step)*Δ_size
    pushfirst!(R,0)
    pushfirst!(D_r,1)

    return  (collect(1:N_step)*Δ_size,xg_r,xh_r)
end

function xGh_hd_k(xg_ij_k,Δ)
    N_step=floor(Int,N/2)
    Δ_size=Δ*sqrt(K2[0,1])

    xg_k= zeros(Float32,N_step)
    xh_k= zeros(Float32,N_step)
    N_k= zeros(Float32,N_step)

    for i in 1: Int(N/2), j in 1:Int(N/2)
            xg_tmp= xg_ij[i,j,1,1]+xg_ij[i,j,2,2]
            xh_tmp= -(xg_ij[i,j,1,1]+xg_ij[i,j,2,2])+2*K(i,j)[1]^2*xg_ij[i,j,1,1]/(K2[i,j])+2*K(i,j)[2]^2*xg_ij[i,j,2,2]/(K2[i,j])+2*K(i,j)[1]*K(i,j)[2]*(xg_ij[i,j,1,2]+xg_ij[i,j,2,1])/(K2[i,j])

            k_index=K2[i,j]+1

        if  k_index< N_step
            xg_k[k_index]=xg_k[k_index]+xg_tmp
            xh_k[k_index]=xh_k[k_index]+xh_tmp
            N_k[k_index]= N_k[k_index]+1
        end
    end

    for k in 1:Int(N_step)
        xG_tmp= xg_r[k]
        xg_k[k]=xG_tmp/(N_k[k]+1e-6)

        xh_tmp= xh_r[k]
        xh_k[k]=xh_tmp/(N_k[k]+1e-6)

    end

    K_list=collect(1:N_step)*Δ_size
    pushfirst!(R,0)
    pushfirst!(xg_k,1)
    pushfirst!(xh_k,0)


    return  (collect(1:N_step)*Δ_size,xg_k,xh_k)
end

v_test=Wilson_line_Ny()
A_test=Gluon_field(v_test)

xg_test=xG_ij(A_test)
xg_k_test=


(r,dipole)=Dr_prime(v_test,2)

(r1,xG_1,xh_1,xG_y,xh_y)=xGh_vs(A_test)

(r2,xG_2,xh_2)=xGh_hd(xg_test,1)


A = [-4. -17.; ]
