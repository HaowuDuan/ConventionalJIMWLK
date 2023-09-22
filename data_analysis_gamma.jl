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
using Interpolations
using ForwardDiff
# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.

# load the data from HPC

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

data_JIMWLK=zeros(200,129,7)

for i in 1:200
    data_tmp = readdlm("HPC_data/correct_format_$(i).dat")
    data_JIMWLK[i,:,:]=reshape(data_tmp,(129,7))
end 

R = collect(1:N÷2) * 2a
pushfirst!(R, 0)
plot(R, data_JIMWLK[1,:,7])
plot!(R, data_JIMWLK[1,:,6])
#

dipole=zeros(7,129,3)

# for extracting the error bar
function BOOTSTRAP(data, N_of_config)
    bootstrap = zeros(N_of_config)
    for i in 1:N_of_config
        BS_sample = similar(bootstrap)
        for j in 1:N_of_config
            index = rand(1:N_of_config)
            BS_sample[j] = data[index]
        end
        bootstrap[i] = mean(BS_sample)
    end

    bootstrap_MEAN = mean(bootstrap)
    bootstrap_STD = std(bootstrap)
    (bootstrap_MEAN, bootstrap_STD)
end

for y in 1:7
   for r in 1:129
       # assign the value for coordinate
       dipole[y,r,1]=R[r]
       #Bootstap results
        BS_tmp = BOOTSTRAP(data_JIMWLK[:, r, y], 200)
        
       # assign the value for dipole
        dipole[y, r, 2] = BS_tmp[1]
       # assign the value for error bar
        dipole[y, r, 3] = BS_tmp[2]

   end 
end

plot(dipole[1, :, 1] * Sat_Mom(dipole[1, :, 1], dipole[1, :, 2])[1], dipole[1, :, 2],label="y=0")
plot!(dipole[2, :, 1] * Sat_Mom(dipole[2, :, 1], dipole[2, :, 2])[1], dipole[2, :, 2], label="y=0.5")
plot!(dipole[3, :, 1] * Sat_Mom(dipole[3, :, 1], dipole[3, :, 2])[1], dipole[3, :, 2], label="y=1.0")
plot!(dipole[4, :, 1] * Sat_Mom(dipole[4, :, 1], dipole[4, :, 2])[1], dipole[4, :, 2], label="y=1.5")
plot!(dipole[5, :, 1] * Sat_Mom(dipole[5, :, 1], dipole[5, :, 2])[1], dipole[5, :, 2], label="y=2.0")
plot!(dipole[6, :, 1] * Sat_Mom(dipole[6, :, 1], dipole[6, :, 2])[1], dipole[6, :, 2], label="y=2.5")
plot!(dipole[7, :, 1] * Sat_Mom(dipole[7, :, 1], dipole[7, :, 2])[1], dipole[7, :, 2], label="y=3.0")


plot!(ylabel=L"D(r)",
    xlabel=L"rQ_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=15,
    yguidefontsize=15,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=12mm, bottom_margin=5mm)

xlims!(0,20)

savefig("JIMWLK_dipole.pdf")


Qs=zeros(7)

for i in 1:7
    Qs[i] = Sat_Mom(dipole[i, :, 1], dipole[i, :, 2])[1]
end 



plot(dipole[1, :, 1] * Qs[1], -Log.(dipole[1, :, 2])/Qs[1]^2, label="y=0")
plot!(dipole[2, :, 1] * Qs[1], -Log.(dipole[2, :, 2])/Qs[2]^2, label="y=0.5")
plot!(dipole[3, :, 1] * Qs[1], -Log.(dipole[3, :, 2])/Qs[3]^2, label="y=1.0")
plot!(dipole[4, :, 1] * Qs[1], -Log.(dipole[4, :, 2]) / Qs[4]^2, label="y=1.5")
plot!(dipole[5, :, 1] * Qs[1], -Log.(dipole[5, :, 2]) / Qs[5]^2, label="y=2.0")
plot!(dipole[6, :, 1] * Qs[1], -Log.(dipole[6, :, 2]) / Qs[6]^2, label="y=2.5")
plot!(dipole[7, :, 1] * Qs[1], -Log.(dipole[7, :, 2]) / Qs[7]^2, label="y=3.0")

plot!(ylabel=L"-\frac{\ln D(r)}{Q_s^2}",
    xlabel=L"rQ^{initial}_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=15,
    yguidefontsize=15,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=18mm, bottom_margin=5mm)


#3000 configs
dipole_3000 = zeros(7, 129, 3)
data3000_JIMWLK = zeros(3000, 129, 7)



for i in 1:3000
    data_tmp = readdlm("corrected/correct_format_$(i).dat")
    data3000_JIMWLK[i, :, :] = reshape(data_tmp, (129, 7))
end


for y in 1:7
    for r in 1:129
        # assign the value for coordinate
        dipole_3000[y, r, 1] = R[r]
        #Bootstap results
        BS_tmp = mean(data3000_JIMWLK[:, r, y])#BOOTSTRAP(data3000_JIMWLK[:, r, y], 1000)

        # assign the value for dipole
        dipole_3000[y, r, 2] = BS_tmp[1]
        # assign the value for error bar
        # dipole_3000[y, r, 3] = BS_tmp[2]

    end
end


Qs1 = zeros(7)

for i in 1:7
    Qs1[i] = Sat_Mom(dipole_3000[i, :, 1], dipole_3000[i, :, 2])[1]
end
print(Qs1)

plot(dipole_3000[1, :, 1] * Qs1[1], -Log.(dipole_3000[1, :, 2]) / Qs1[1]^2, label="y=0")

plot(dipole_3000[2, :, 1] * Qs1[1], -Log.(dipole_3000[2, :, 2]) / Qs1[2]^2, label="y=0.5")
plot(dipole_3000[3, :, 1] * Qs1[1], -Log.(dipole_3000[3, :, 2]) / Qs1[3]^2, label="y=1.0")
plot(dipole_3000[4, :, 1] * Qs1[1], -Log.(dipole_3000[4, :, 2]) / Qs1[4]^2, label="y=1.5")
plot(dipole_3000[5, :, 1] * Qs1[1], -Log.(dipole_3000[5, :, 2]) / Qs1[5]^2, label="y=2.0")
plot(dipole_3000[6, :, 1] * Qs1[1], -Log.(dipole_3000[6, :, 2]) / Qs1[6]^2, label="y=2.5")
plot(dipole_3000[7, :, 1] * Qs1[1], -Log.(dipole_3000[7, :, 2]) / Qs1[7]^2, label="y=3.0")

plot!(ylabel=L"-\frac{\ln D(r)}{Q_s^2}",
    xlabel=L"rQ^{initial}_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=15,
    yguidefontsize=15,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=18mm, bottom_margin=5mm)


plot(dipole_3000[1, :, 1] * Qs1[1], dipole_3000[1, :, 2], label="y=0")
plot!(dipole_3000[2, :, 1] * Qs1[2], dipole_3000[2, :, 2], label="y=0.5")
plot!(dipole_3000[3, :, 1] * Qs1[3], dipole_3000[3, :, 2], label="y=1.0")
plot!(dipole_3000[4, :, 1] * Qs1[4], dipole_3000[4, :, 2], label="y=1.5")
plot!(dipole_3000[5, :, 1] * Qs1[5], dipole_3000[5, :, 2], label="y=2.0")
plot!(dipole_3000[6, :, 1] * Qs1[6], dipole_3000[6, :, 2], label="y=2.5")
plot!(dipole_3000[7, :, 1] * Qs1[7], dipole_3000[7, :, 2], label="y=3.0")

plot!(ylabel=L"D(r)",
    xlabel=L"rQ_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=15,
    yguidefontsize=15,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=18mm, bottom_margin=5mm)

xlims!(0,10)

Gamma_data = similar(dipole_3000[3, :, :])

writedlm("dipole.dat", dipole_3000[5,:,:])

for i in 1:129
    Gamma_data[i, 1]=dipole_3000[3,i,1]
    Gamma_data[i, 2] = -Log.(dipole_3000[3, i, 2]) / Qs1[3]^2
end 

writedlm("Gamma_data_y1.dat", Gamma_data)


#6000 configs
dipole_6000 = zeros(7, 129, 3)

data6000_JIMWLK = zeros(6000, 129, 7)

for i in 1:6000
    data_tmp = readdlm("HPC_6000/correct_format_$(i).dat")
    data6000_JIMWLK[i, :, :] = reshape(data_tmp, (129, 7))
end


for y in 1:7
    for r in 1:129
        # assign the value for coordinate
        dipole_6000[y, r, 1] = R[r]
        #Bootstap results
        BS_tmp = BOOTSTRAP(data6000_JIMWLK[:, r, y], 6000)

        # assign the value for dipole
        dipole_6000[y, r, 2] = BS_tmp[1]
        # assign the value for error bar
         dipole_6000[y, r, 3] = BS_tmp[2]

    end
end




Qs1 = zeros(7)

Threads.@threads for i in 1:7
    Qs1[i] = Sat_Mom(dipole_6000[i, :, 1], dipole_6000[i, :, 2])[1]
end
print(Qs1)

plot(dipole_6000[1, :, 1], dipole_6000[1, :, 2])
plot!(dipole_6000[2, :, 1], dipole_6000[2, :, 2])

plot(dipole_6000[1, :, 1], -Log.(dipole_6000[1, :, 2]) / Qs1[1]^2, label=L"y=0, Q_s=0.94")
plot!(dipole_6000[2, :, 1], -Log.(dipole_6000[2, :, 2]) / Qs2[2]^2, label=L"y=0.2, Q_s=0.94")
plot!(dipole_6000[3, :, 1], -Log.(dipole_6000[3, :, 2]) / Qs2[3]^2, label=L"y=0.4,Q_s=1.13 ")
plot!(dipole_6000[4, :, 1], -Log.(dipole_6000[4, :, 2]) / Qs2[4]^2, label=L"y=0.6, Q_s=1.41")
plot!(dipole_6000[5, :, 1], -Log.(dipole_6000[5, :, 2]) / Qs2[5]^2, label=L"y=0.8, Q_s=1.89")
#plot(dipole_6000[6, :, 1] * Qs2[1], -Log.(dipole_6000[6, :, 2]) / Qs2[6]^2, label="y=2.5")
#plot(dipole_6000[7, :, 1] * Qs2[1], -Log.(dipole_6000[7, :, 2]) / Qs2[7]^2, label="y=3.0")

plot!(ylabel=L"-\frac{\ln d(r)}{Q_s^2}",
    xlabel=L"rQ^{initial}_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=15,
    yguidefontsize=15,
    thickness_scaling=1,
    legendfontsize=10,
    lengend=:topleft,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=18mm, bottom_margin=5mm)


savefig("Gamma_JIMWLK.pdf")   
xlims!(0,12)

R2 = R .^ 2


#=
function Gamma_driv(Gamma_dat, R)
    Gamma_r2 = interpolate(R2, Gamma_dat, SteffenMonotonicInterpolation())
    
    r=LinRange(0,14,2000)
    r2=r.^2

    Gamma1_dat = similar(r)
    Gamma2_dat = similar(r)


    for i in 1:length(r)
        Gamma1_dat[i] = ForwardDiff.derivative(Gamma_r2, r2[i])
    end

    Gamma1_r2 = interpolate(r2, Gamma1_dat, SteffenMonotonicInterpolation())

    for i in 1:length(r)
        Gamma2_dat[i] = ForwardDiff.derivative(Gamma1_r2, r2[i])
    end

    Gamma2_r2 = interpolate(r2, Gamma2_dat, SteffenMonotonicInterpolation())
   
    Gamma1_dat = Gamma1_r2.(R2[1:57])
    Gamma2_dat = Gamma2_r2.(R2[1:57])

    return [Gamma1_dat,Gamma2_dat]
end
=#
















































