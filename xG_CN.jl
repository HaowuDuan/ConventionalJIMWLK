cd(@__DIR__)
using Pkg
Pkg.activate(".")
#Pkg.add("JLD2")
using DelimitedFiles
# using StaticArray
using LinearAlgebra
using LsqFit
using LaTeXStrings
using Random
using Cubature
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
using QuadGK
#using DifferentialEquation
using ForwardDiff
using Interpolations
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
const m2=(0.1)^2#1/L

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
    fft_p*rand(rng,Normal(0,gμ*1.5/(sqrt(Ny)*a)),N,N)# draw the color charge density from N(0,1)                                # for each layer, each point, and each color
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

function Log(x)
    if x*100 >0
       return  log(x)
    else
       return 0
   end
end

function compelete_MV(x,m)
    (1-m*x*besselk(1,m*x))/m^2
end

function D_MV(r,m,Q)
    return exp(-Q^2*r^2*log(1/(r^2*(m^2)+1e-6)+exp(1))/4)
end

function D_GBW(r)
    return exp(-r^2/4)
end

function D_full_MV(r,m,Q)
    return exp(-Q^2*(1-m*r*besselk(1,m*r))/m^2)#exp(-compelete_MV(r,m))
end

function adj_dipole(f_dipole)
    return exp(log(f_dipole)*2Nc^2/Ng)
end

function WW_exact_asy(r,m)
     f_dipole=D_MV(r,0.2,0.4)
     ad= adj_dipole(f_dipole)
     Γ₁= log(1/(r^2*(m^2)+1e-6)+exp(1))-1
     Γ = r^2*log(1/(r^2*(m^2)+1e-6)+exp(1))
     xG=(1-ad)*(Γ₁-1)/Γ
     xh=(1-ad)/Γ
     return (xG,xh)
end

function WW_exact_full(r,m)s
    f_dipole=D_full_MV(r,0.2,0.4)
    ad=adj_dipole(f_dipole)
    Γ=(1-m*r*besselk(1,m*r))/m^2
    Γ₁= besselk(0,m*r)/2
    Γ₂= -besselk(1,m*r)*m/(4r)
    xG=(1-ad)*(Γ₁+r^2*Γ₂)/Γ
    xh=-(1-ad)* r^2*Γ₂/Γ
    return (xG,xh)
end



function color_neutral_gamma(r,Q)
         quadgk(x ->2*(r^2)*x^3*(1-besselj(0,x))/(x^2+Q^2*r^2)/(x^2)^2 ,0,313.374,rtol=1e-9)
end

function dipole_color_neutral(r,Q,Qs)
         return exp(-Qs^2*color_neutral_gamma(r,Q)[1])
end

function Gamma_driv(r,Q)
    Gamma_cn_dat=similar(R)
    Gamma1_cn_dat=similar(R)
    Gamma2_cn_dat=similar(R)

    for i in 1:length(R)
        Gamma_cn_dat[i]=color_neutral_gamma(R[i],Q)[1]
    end

    Gamma_cn_r2= interpolate(R2, Gamma_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma1_cn_dat[i]=ForwardDiff.derivative(Gamma_cn_r2, R2[i])
    end

    Gamma1_cn_r2= interpolate(R2, Gamma1_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma2_cn_dat[i]=ForwardDiff.derivative(Gamma1_cn_r2, R2[i])
    end

    Gamma2_cn_r2= interpolate(R2, Gamma2_cn_dat, SteffenMonotonicInterpolation())

    return (Gamma1_cn_r2(r^2), Gamma2_cn_r2(r^2))
end

function WW_cn(r,Q,Qs)
    f_dipole=dipole_color_neutral(r,Q,Qs)
    ad=adj_dipole(f_dipole)
    Γ=color_neutral_gamma(r,Q)[1]

    Γ_dri=Gamma_driv(r,Q)

    Γ₁= Γ_dri[1]
    Γ₂= Γ_dri[2]

    xG= (1-ad)*(Γ₁+r^2*Γ₂)/Γ
    xh=-(1-ad)* r^2*Γ₂/Γ
    return (xG,xh,f_dipole)
end

function R_max(R,data)
    r_max=zeros(1)
    for i in 1:length(R)
           δ=data[i]-data[i+1]
           if δ>=0
            #    Qs = sqrt(2)/r[i]
                r_max = R[i]
               break
           end
    end
    return r_max
end

# vladi data for xg
xg_text=readdlm("Archive/S_0.0.dat")
v_xg=zeros(11,255,5)
# vladi data for dipole
v_di=zeros(11,255,2)
for i in 1:10
    y=i-1
    v_xg[i,:,:]=readdlm("Archive/xG_0.$y.dat")
    v_di[i,:,:]=readdlm("Archive/S_0.$y.dat")
end 

plot(v_di[5, :, 1,], v_di[5, :, 2] .^(9/4),label="fundamental")
plot!(v_di[5, :, 1,], v_xg[5, :, 3] ./ v_di[5, :, 2] .^ (9 / 4), label="adjoint/ fundamental, y=5")

v_xg[11, :, :] = readdlm("Archive/xG_1.0.dat")
v_di[11, :, :] = readdlm("Archive/S_1.0.dat")

Q=zeros(11)
Qs=zeros(11)
# xg is the -vg[:,:,2] adjoint dipole is vg[:,:,3]
for i in 1:11
            
    Qs[i] = Sat_Mom(v_di[i, :, 1], v_di[i, :, 2])[1]

end 

# we first compute observable which is rxG/(1-D) times ln(d)
obs=zeros(11,255)
obsa=similar(obs)
obs_im=similar(obs)
err=zeros(11,255)
errv=zeros(11,255)
for i in 1:11

    obs[i,:] .= -v_xg[1, :, 1] .* v_xg[i, :, 2] .* Log.(v_xg[i, :, 3]) ./ (v_xg[i, :, 3] .- 1)
 #   obsa[i, 5:100] .= -v_xg[1, 5:100, 1] .* v_xg[i, 5:100, 2] .* log.(v_xg[i, 5:100, 3]) ./ (v_xg[i, 5:100, 3] .- 1)
    obs_im[i, :] .= -v_xg[1, :, 1] .* v_xg[i, :, 2] .* log.(v_di[i, :, 2] .^ (9 / 4)) ./ (v_di[i, :, 2] .^ (9 / 4) .- 1)
   # err[i, 5:100] .= 
    errv[i, :] .= abs.(log.(abs.(v_xg[i, :, 3])) .* abs.(v_xg[i, :, 5]) ./ v_xg[i, :, 1]) 
end



plot(v_xg[5, 5:100, 1],- v_xg[5, 5:100, 2], yerr=v_xg[5, 5:100, 5])

Y=LinRange(0,1.0,11)

#scatter(Y,Q[1:9] ./Qs[1:9],label=L"Q/Q_s", ylabel=L"Q/Q_s", xlabel="Y")

# now we need to fit the data
#=
function modelxg(r, param, γ )
    m = abs(param[1])
    Q = abs(param[2])



    function Γ_int(p, r)
        return  p*(1.0 - besselj0(p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 

    function d2Γ_int(p, r)
        return p^3*(besselj(0,p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 
    
    Γ = zeros(length(r))
    d2Γ = zeros(length(r))

    Threads.@threads for i = 1:length(r) 
        x = r[i]
        Γ[i] =  quadgk(p -> Γuint(p,x), 0.0,  10000.0/x, rtol=1e-4)[1]
        d2Γ[i] = quadgk(p -> d2Γuint(p,x), 0.0, 10000.0/x, rtol=1e-4)[1]
    end 

    return  2.0*r.^2 . d2Γ ./ Γ param[3]
end

test_data=zeros(100,2)


for i in 1:100
   test_data[i,1]=i/10
   test_data[i,2]=(i/10)^(1/3)
end


function t_model(x,p)
        return x .^p
end 

test_fit = curve_fit((x, p) -> t_model(x, p), test_data[:, 1], test_data[:, 2], [0.2], lower=[0.1], upper=[0.5]; autodiff=:finiteforward, maxIter=200)

scatter(test_data[:,1],test_data[:,2],label="data")

plot!(test_data[:, 1], t_model.(test_data[:, 1], test_fit.param), label="param=$(test_fit.param)")
=#

function M_xg(r,pa)
      A=pa[1]
      m=abs(pa[2])
      Q=abs(pa[3])
      γ=pa[4]

    function Γ_int(p, r)
        return  p*(1.0 - besselj0(p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 

    function d2Γ_int(p, r)
        return p^3*(besselj(0,p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 
    
    Γ = zeros(length(r))
    d2Γ = zeros(length(r))

    Threads.@threads for i in 1:length(r)
       x=r[i]
       #Γ[i]=quadgk(p -> Γ_int(p,x), 0.0,  10000.0/x, rtol=1e-4)[1] 
       d2Γ[i]=quadgk(p -> d2Γ_int(p,x), 0.0,  10000.0/x, rtol=1e-4)[1]   
    end
    return A*r .* d2Γ #./ Γ 
end 

Para=zeros(4,11,2)
Para_err=zeros(4,11)
Pr=Progress(11)
p0=[10, 0.002, 0.5, 0.7]
p1=[3.0, 0.01, 0.5, 0.7]
for i in 1:11
    fit_tmp = curve_fit((r, p) -> M_xg(r, p), v_xg[1, 5:100, 1], obs[i, 5:100], 1.0 ./ (err[i, 5:100] .* 1.0), p0, lower=[0.01, 0.0001, 0.001, 0.6], upper=[100.0, 0.2, 5.0, 1.1])#; autodiff=:finiteforward, maxIter=200)
    # curve_fit((r, p) -> M_xg(r, p), v_xg[1, 5:100, 1], obs[i, 5:100], 1.0 ./ (err[i, 5:100] .* 1.5), p1, lower=[3.0, 0.0, 0.1, 0.5], upper=[6.0, 0.2, 2.0, 1.5]  ; autodiff=:finiteforward, maxIter=200)
    Para[:,i, 1] .=fit_tmp.param 
    p0 .=fit_tmp.param
    next!(Pr)
end 

for i in 1:11
    fit_tmp =  curve_fit((r, p) -> M_xg(r, p), v_xg[1, 5:100, 1], obs[i, 5:100], 1.0 ./ (err[i, 5:100] .* 1.0), p1, lower=[3.0, 0.0, 0.1, 0.5], upper=[6.0, 0.2, 2.0, 1.5] )# ; autodiff=:finiteforward, maxIter=200)
    Para[:, i, 2] .= fit_tmp.param
    p1 .= fit_tmp.param
    next!(Pr)
end
print(Para[3,:])

scatter(Y, Para[3,:,1]./Qs[:], xlabel="Y", ylabel=L"Q/Q_s", label=L"V- initial condition", ylims=(0,1))
scatter!(Y, Para[3, :, 2] ./ Qs[:], xlabel="Y", ylabel=L"Q/Q_s", label=L"H- initial condition", ylims=(0, 1))

scatter(Y, Para[4, :, 1] , xlabel="Y", ylabel=L"\gamma", label=L"V- initial condition")
scatter!(Y, Para[4, :, 2] , xlabel="Y", ylabel=L"\gamma", label=L"H- initial condition")


scatter(Y, Para[2, :, 1], xlabel="Y", ylabel=L"m", label=L"V- initial condition")
scatter!(Y, Para[2, :, 2], xlabel="Y", ylabel=L"m", label=L"H- initial condition")

scatter(Y, Para[1, :, 1], xlabel="Y", ylabel=L"A", label=L"V- initial condition")
scatter!(Y, Para[1, :, 2], xlabel="Y", ylabel=L"A", label=L"H- initial condition")



scatter(Y, Para[4, :] , label=L"\gamma", xlabel="Y")

scatter(Y, Para[1, :], label="A", xlabel="Y")


fit_tmp=curve_fit((r, p) -> M_xg(r, p), v_xg[1,5:100,1], obs[5,5:100], [3, 0.01, 0.5, 0.7],lower=[3.0, 0.0 ,0.1, 0.5],upper=[6.0,0.2,2.0,1.5]; autodiff=:finiteforward, maxIter=200)
test_fit=fit_tmp.param

scatter(v_xg[1, 5:100, 1], obs[5, 5:100], label="Y=0.4, data")
R=v_xg[1, 5:100, 1]
test_data = similar(v_xg[1, 5:100, 1])
test_data1 = similar(v_xg[1, 5:100, 1])
test_data .= M_xg(R, Para[:, 5])



begin
    y = 6
    i= y+1
    scatter(v_xg[1, 5:5:100, 1], obs[i, 5:5:100] ,label="Y=0.$(y), data")
    #scatter!(v_xg[1, 5:5:60, 1], obsa[i, 5:5:60], label="Y=0.$(y), aj data")
    test_data .= M_xg(R, Para[:, i, 1])
    test_data1 .= M_xg(R, Para[:, i, 2])
    plot!(R, test_data, label="Y=0.$(y), V")
    plot!(R, test_data1, label="Y=0.$(y), H")
end


Pr=Progress(11)
for i in 1:11
    fit_tmp = curve_fit((r, p) -> M_xg(r, p), v_xg[1, 5:100, 1], obs[i, 5:100], 1.0 ./ (errv[i, 5:100] .* 1.0), p0, lower=[0.01, 0.0001, 0.001, 0.6], upper=[100.0, 0.2, 5.0, 1.1])#; autodiff=:finiteforward, maxIter=200)
    # curve_fit((r, p) -> M_xg(r, p), v_xg[1, 5:100, 1], obs[i, 5:100], 1.0 ./ (err[i, 5:100] .* 1.5), p1, lower=[3.0, 0.0, 0.1, 0.5], upper=[6.0, 0.2, 2.0, 1.5]  ; autodiff=:finiteforward, maxIter=200)
    Para_err[:, i] .= fit_tmp.param
    p0 .= fit_tmp.param
    next!(Pr)
end

print(Para_err[3,:])



scatter(Y, Para[3,:,1]./Qs[:], xlabel="Y", ylabel=L"Q/Q_s", label=L"H- weight ", ylims=(0,1))
scatter!(Y, Para_err[3, :] ./ Qs[:], xlabel="Y", ylabel=L"Q/Q_s", label=L"V- weight", ylims=(0, 1))


begin
    y = 7
    i = y + 1
    scatter!(v_xg[1, 5:5:100, 1], obs[i, 5:5:100], label="Y=0.$(y), data")
    #scatter!(v_xg[1, 5:5:60, 1], obsa[i, 5:5:60], label="Y=0.$(y), aj data")
    test_data .= M_xg(R, Para_err[:, i])
    test_data1 .= M_xg(R, Para[:, i, 1])
    plot!(R, test_data, label="Y=0.$(y), V", linestyle=:dash, linecolor=:black)
    plot!(R, test_data1, label="Y=0.$(y), H")
end

Para1_y = zeros(4,11)
Para2_y =zeros(4,11)
p1 = [3.514, 0.0, 0.47, 1.13]
Pr=Progress(11)
p0 = [40, 0.14, 0.17, 0.998]
for i in 10:11
    fit_tmp =@time curve_fit(M_xg, v_xg[1, 5:100, 1], obs[i, 5:100], 1.0 ./ (errv[i, 5:100] .* 1.0), p0,lower=[0.01, 0.0001, 0.001, 0.6], upper=[100.0, 0.2, 5.0, 1.1]) #lower=[3.0, 0.0, 0.1, 0.5], upper=[6.0, 0.2, 2.0, 1.5])# ; autodiff=:finiteforward, maxIter=200)
    Para2_y[:, i] .= fit_tmp.param
    p0 .= fit_tmp.param
    println(i)
    # next!(Pr)
end

scatter(Y,Para2_y[3,:] ./(Qs[:]),ylims=(0,1))
scatter(Y, Para2_y[2,:])
scatter(Y, Para2_y[4, :])


# compare data
begin
    y = 8
    i = y + 1
    plot(v_xg[1, :, 1], obs[i, :], label="Y=0.$(y), D(r)")

    plot!(v_xg[1, :, 1], obs_im[i, :], label="Y=0.$(y), d(r)")
    #scatter!(v_xg[1, 5:5:60, 1], obsa[i, 5:5:60], label="Y=0.$(y), aj data")
   # test_data .= M_xg(R, Para[:, i, 1])
    #test_data1 .= M_xg(R, Para[:, i, 2])
   # plot!(R, test_data, label="Y=0.$(y), V")
   # plot!(R, test_data1, label="Y=0.$(y), H")
end

# fit using data from fundamental dipole instead 

Para3_y = zeros(4, 11)
Para4_y = zeros(4, 11)
Pr = Progress(9)
p0 = [40, 0.14, 0.17, 0.998]

for i in 1:9
    fit_tmp = @time curve_fit(M_xg, v_xg[1, 5:100, 1], obs_im[i, 5:100], 1.0 ./ (errv[i, 5:100] .* 1.0), p0, lower=[0.01, 0.0001, 0.001, 0.6], upper=[100.0, 0.2, 5.0, 1.1] ; autodiff=:finiteforward, maxIter=200)
    Para4_y[:, i] .= fit_tmp.param
    p0 .= fit_tmp.param
    println(i)
    next!(Pr)
end

scatter(Y[1:9], Para2_y[3, 1:9] ./ (Qs[1:9]), ylims=(0, 1),label="data from adjoint dipole")
scatter!(Y[1:9], Para3_y[3, 1:9] ./ (Qs[1:9]), ylims=(0, 1),label="data from fundamental dipole")
scatter!(Y[1:9], Para4_y[3, 1:9] ./ (Qs[1:9]), label="200 iteration")
scatter!(xlabel="Y", ylabel=L"Q/Q_s")
plot!(yguidefontrotation=-90, leftmargin=15mm)

savefig("fit/Q.pdf")

scatter(Y[1:9], Para2_y[4, 1:9], label="data from adjoint dipole")
scatter!(Y[1:9], Para3_y[4, 1:9], label="data from fundamental dipole")
scatter!(Y[1:9], Para4_y[4, 1:9], label="200")
scatter!(xlabel="Y", ylabel=L"\gamma")
plot!(yguidefontrotation=-90, leftmargin=15mm)

savefig("fit/gamma.pdf")

scatter(Y[1:9], Para2_y[2, 1:9], label="data from adjoint dipole")
scatter!(Y[1:9], Para3_y[2, 1:9], label="data from fundamental dipole")
scatter!(xlabel="Y", ylabel="m")
plot!(yguidefontrotation=-90, leftmargin=15mm)

savefig("fit/m.pdf")

scatter(Y[1:9], Para2_y[1, 1:9] ./(Qs[1:9] .^2), label="data from adjoint dipole")
scatter!(Y[1:9], Para3_y[1, 1:9] ./(Qs[1:9] .^2), label="data from fundamental dipole")
scatter!(xlabel="Y", ylabel=L"\frac{rxG(r)}{1-D(r)}\frac{-\ln D(r)}{Q_s^2}")
plot!(yguidefontrotation=-90, leftmargin=20mm)
savefig("fit/A.pdf")




#  Compare with xh and Gamma data


function M_xh(r, pa)
    A = pa[1]
    m = abs(pa[2])
    Q = abs(pa[3])
    γ = pa[4]

    function Γ_int(p, r)
        return p * (1.0 - besselj0(p * r)) / (p^2 + m^2)^2 * p^2 * (Q^2 / p^2)^γ / (1.0 + (Q^2 / p^2)^γ)
    end

    function d2Γ_int(p, r)
        return p^3 * (besselj(0, p * r)) / (p^2 + m^2)^2 * p^2 * (Q^2 / p^2)^γ / (1.0 + (Q^2 / p^2)^γ)
    end

    Γ = zeros(length(r))
    d2Γ = zeros(length(r))

    Threads.@threads for i in 1:length(r)
        x = r[i]
        #Γ[i]=quadgk(p -> Γ_int(p,x), 0.0,  10000.0/x, rtol=1e-4)[1] 
        d2Γ[i] = quadgk(p -> d2Γ_int(p, x), 0.0, 10000.0 / x, rtol=1e-4)[1]
    end
    return A * r .* d2Γ #./ Γ 
end