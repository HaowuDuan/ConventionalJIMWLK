cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.add("JLD2")
using DelimitedFiles
using StaticArrays
using LinearAlgebra
using LaTeXStrings
using Random
using Distributions
using FFTW
#using Plots
using Statistics
using JLD2
using FileIO
using ProgressMeter
using BenchmarkTools
#using PlotThemes
using StatsPlots
using Measures
using SpecialFunctions
using ArbNumerics
# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
include("SU3.jl")

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
m2=(0.1)^2#1/L

const Nc=3

const Ng=Nc^2-1

const αₛ=1

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


V_test=Wilson_line_Ny()


V_test2=Wilson_line_Ny()
data_i=zeros(ComplexF32,N,N,Nc,Nc,10)
dipole_i=zeros(129,2,10)
Threads.@threads for i in 1:10
    data_i[:,:,:,:,i]=Wilson_line_Ny()
end

for i in 1:10
    (dipole_i[:,1,i],dipole_i[:,2,i])=Dr_prime(data_i[:,:,:,:,i],2)
end

V_final=JIMWLK(0.01,10,V_test)
V_1=JIMWLK(0.01,20,V_test)
V_2=JIMWLK(0.01,50,V_test)
V_3=JIMWLK(0.01,100,V_test)
V_4=JIMWLK(0.01,200,V_test)


(r_i,D_i)=Dr_prime(V_test,2)
(r_i2,D_i2)=Dr_prime(V_test2,2)

(r_f,D_f)=Dr_prime(V_final,2)

(r_1,D_1)=Dr_prime(V_1,2)
(r_2,D_2)=Dr_prime(V_2,2)
(r_3,D_3)=Dr_prime(V_3,2)
(r_4,D_4)=Dr_prime(V_4,2)


plot!(dipole_i[:,1,1]*Sat_Mom(dipole_i[:,1,1],dipole_i[:,2,1])[1],dipole_i[:,2,1],label="Y=0, m=0.1")

for i in 2:10
    plot!(dipole_i[:,1,i]*Sat_Mom(dipole_i[:,1,i],dipole_i[:,2,i])[1],dipole_i[:,2,i])
end


plot!(r_f*Sat_Mom(r_f,D_f)[1],D_f,label="Y=0.01*10")

plot!(r_1*Sat_Mom(r_1,D_1)[1],D_1,label="Y=0.01*20")
plot!(r_2*Sat_Mom(r_2,D_2)[1],D_2,label="Y=0.01*50")
plot!(r_3*Sat_Mom(r_3,D_3)[1],D_3,label="Y=0.01*100")
plot!(r_4*Sat_Mom(r_4,D_4)[1],D_4,label="Y=0.01*200")


plot!(xlims=(0, 10))

plot!(ylabel = L"C(r)",
    xlabel = L"rQ_s",
    box = :on,
    foreground_color_legend = nothing,
    fontfamily = "Times New Roman",
    xtickfontsize = 8,
    ytickfontsize = 8,
    xguidefontsize = 20,
    yguidefontsize = 20,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3,yguidefontrotation=-90,left_margin=12mm,bottom_margin=5mm)


plot!(ylabel = L"C(r)",xlabel = L"rQ_s")

function Log(x)
    if x*100 >0
       return  log(x)
    else
       return 0
   end
end

plot(r_i*Sat_Mom(r_i,D_i)[1],-Log.(D_i)/Sat_Mom(r_i,D_i)[1]^2,label="Y=0.01*0, m=0.01")
plot!(r_i2*Sat_Mom(r_i2,D_i2)[1],-Log.(D_i2)/Sat_Mom(r_i2,D_i2)[1]^2,label="Y=0.01*0, m=0.12")


for i in 1:9
    plot!(dipole_i[:,1,i]*Sat_Mom(dipole_i[:,1,i],dipole_i[:,2,i])[1],-Log.(dipole_i[:,2,i])/Sat_Mom(dipole_i[:,1,i],dipole_i[:,2,i])[1]^2,label="Y=0.01*0")
end

plot!(dipole_i[:,1,10]*Sat_Mom(dipole_i[:,1,10],dipole_i[:,2,10])[1],-Log.(dipole_i[:,2,10])/Sat_Mom(dipole_i[:,1,10],dipole_i[:,2,10])[1]^2,label="Y=0.01*0")

plot(r_i*Sat_Mom(r_i,D_i)[1],-Log.(D_i)/Sat_Mom(r_i,D_i)[1]^2,label="Y=0.01*0")
plot!(r_1*Sat_Mom(r_i,D_i)[1],-Log.(D_1)/Sat_Mom(r_1,D_1)[1]^2,label="Y=0.01*20")
plot!(r_2*Sat_Mom(r_i,D_i)[1],-Log.(D_2)/Sat_Mom(r_2,D_2)[1]^2,label="Y=0.01*50")
plot!(r_3*Sat_Mom(r_i,D_i)[1],-Log.(D_3)/Sat_Mom(r_3,D_3)[1]^2,label="Y=0.01*100")
plot!(r_4*Sat_Mom(r_i,D_i)[1],-Log.(D_4)/Sat_Mom(r_4,D_4)[1]^2,label="Y=0.01*200")


plot(ylabel = L"-\ln D(r)",
    xlabel = L"rQ_s",
    box = :on,
    foreground_color_legend = nothing,
    fontfamily = "Times New Roman",
    xtickfontsize = 8,
    ytickfontsize = 8,
    xguidefontsize = 15,
    yguidefontsize = 15,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3,yguidefontrotation=-90,left_margin=12mm,bottom_margin=5mm)


plot!(xlims=(0,6))



# for initial condition
dipole_initial_average=zeros(129)
for i in 1:10
    (r_tmp,d_tmp)=Dr_prime(data_i[:,:,:,:,i],2)
    dipole_initial_average[:].= dipole_initial_average[:].+d_tmp./10
end



plot(r_i,dipole_initial_average,label="initial, average")

plot(r_i*Sat_Mom(r_i,dipole_initial_average)[1],-Log.(dipole_initial_average)/Sat_Mom(r_i,dipole_initial_average)[1]^2,label="Y=0")

# for different rapidity
Dipole_data_2=zeros(129,10)
dipole_final_average2=zeros(129)

p=Progress(10)
for i in 1:10
         data_tmp=JIMWLK(0.01,50,data_i[:,:,:,:,i])
    (r_tmp,d_tmp)=Dr_prime(data_tmp,2)
    dipole_final_average2[:].= dipole_final_average2[:].+d_tmp./10
    next!(p)
end

matrix=[1 1
        2 3]

loaded_data = load("matrix_data.jld2","matrix")

save("matrix_data.jld2", "matrix", matrix)

DF=zeros(1000,2)
DF=readdlm("dipole_full.dat",Float64)

function compelete_MV(x,m)
    (1-m*x*besselk(1,m*x))/m^2
end

function D_MV(r,m)
    return exp(-r^2*log(1/(r^2*(m^2)+1e-6)+exp(1))/4)
end

function D_GBW(r)
    return exp(-r^2/4)
end

function D_full_MV(r,m)
    return exp(-(1-m*r*besselk(1,m*r))/m^2)#exp(-compelete_MV(r,m))
end

function adj_dipole(f_dipole)
    return (Nc^2*f_dipole-1)/Ng
end


function WW_exact_asy(r,m,xG,xh)
     f_dipole=D_MV(r,0.1)
     ad=adj_dipole(f_dipole)
     Γ₁= log(1/(r^2*(m^2)+1e-6)+exp(1))-1
     Γ = r^2*log(1/(r^2*(m^2)+1e-6)+exp(1))
     xG=(1-ad)*(Γ₁-1)/Γ
     xh=(1-ad)/Γ
end

function WW_exact_full(r,m,xG,xh)
    f_dipole=D_MV(r,0.1)
    ad=adj_dipole(f_dipole)
    Γ=(1-m*r*besselk(1,m*r))/m^2
    Γ₁= besselk(0,m*r)/2
    Γ₂= -besselk(1,m*r)*m/(4r)
    xG=(1-ad)*(Γ₁-r^2*Γ₂)/Γ
    xh=(1-ad)* r^2*Γ₂/Γ
end

R=LinRange(0,20,2000)

d_MV=similar(R)
d_full_MV=similar(R)


for i in 1:2000
    d_MV[i]=D_MV(R[i],0.1)
    d_full_MV[i]=D_full_MV(R[i],0.1)
end

Sat_Mom(R,d_MV)[1]
Sat_Mom(R,d_full_MV)[1]

plot(R*Sat_Mom(R,d_MV)[1],d_MV,label="MV small m asymptotic, m=0.1")
plot!(R*Sat_Mom(R,d_full_MV)[1],d_full_MV,label="MV Full, m=0.1")
plot!(ylabel = L"D(r)",
    xlabel = L"rQ_s")
xlims!(0,10)



plot(RL,-Log.(d_MV)/Sat_Mom(RL,d_MV)[1]^2,label="MV small m asymptotic")
plot!(RL,-Log.(d_full_MV)/Sat_Mom(RL,d_full_MV)[1]^2,label="MV Full")
plot!(ylabel = L"\Gamma(r)",
    xlabel = L"rQ_s")
xlims(0,10)



# Caculate instead of dipole, xG, xh


data_test = [1  2
             3  4]
println(data_test)
cleaned_data = replace(data_test, r"[;\[\]]" => "")
