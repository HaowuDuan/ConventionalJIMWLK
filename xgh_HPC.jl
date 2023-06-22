cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DelimitedFiles
using StaticArrays
using LinearAlgebra
using Random
using Distributions
using FFTW
using Statistics
using JLD2
using GellMannMatrices

I=ARGS[1]
#longitudinal layers
const Ny=50
# Transverse lattice size in one direction
const N=64*4
#lattice size
const L=32
# lattice spacing
const a=L/N
#
const gμ=1
# infra regulator m
const m2=(0.01)^2#1/L

const Nc=3

const Ng=Nc^2-1

const αₛ=1

# SU(3) color matrices
t=gellmann(Nc,skip_identity=false)/2
t[9]=t[9]*sqrt(2/3)


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

function JIMWLK(Δ,N_of_steps,V_input)
      Y_f=N_of_steps*Δ


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


V_initial=Wilson_line_Ny()

V_1=JIMWLK(0.01,10,V_initial)

V_2=JIMWLK(0.01,20,V_initial)

V_3=JIMWLK(0.01,50,V_initial)

V_4=JIMWLK(0.01,100,V_initial)

V_5=JIMWLK(0.01,200,V_initial)

V_6=JIMWLK(0.01,300,V_initial)

data_dipole=zeros(7,129)

data_dipole[1,:]=Dr_prime(V_initial,2)

data_dipole[2,:]=Dr_prime(V_1,2)

data_dipole[3,:]=Dr_prime(V_2,2)

data_dipole[4,:]=Dr_prime(V_3,2)

data_dipole[5,:]=Dr_prime(V_4,2)

data_dipole[6,:]=Dr_prime(V_5,2)

data_dipole[7,:]=Dr_prime(V_6,2)


#save("Dipole_JIMWLK_$(I)", "data", data_dipole)
