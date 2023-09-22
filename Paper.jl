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



# This file is just to make plots and test the fit. Not including any simulation or JIMWLK code

function Sat_Mom(r, D)
    Qs = zeros(1)
    Rs = zeros(1)
    for i in 1:length(r)
        δ = D[i] - exp(-0.5)
        if δ <= 0
            Qs = sqrt(2) / r[i]
            Rs = r[i]
            break
        end
    end
    return (Qs, Rs)
end

function Log(x)
    if x*100 >0
       return  log(x)
    else
       return 0
   end
end
#= We don't need MV for our fit
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


function WW_exact_full(r,m)
    f_dipole=D_full_MV(r,0.2,0.4)
    ad=adj_dipole(f_dipole)
    Γ=(1-m*r*besselk(1,m*r))/m^2
    Γ₁= besselk(0,m*r)/2
    Γ₂= -besselk(1,m*r)*m/(4r)
    xG=(1-ad)*(Γ₁+r^2*Γ₂)/Γ
    xh=-(1-ad)* r^2*Γ₂/Γ
    return (xG,xh)
end
=#

function adj_dipole(f_dipole)
    return exp(log(f_dipole)*2Nc^2/Ng)
end

# define our model 

function color_neutral_gamma(r, Q)
    quadgk(x -> 2 * (r^2) * x^3 * (1 - besselj(0, x)) / (x^2 + Q^2 * r^2) / (x^2)^2, 0, 313.374, rtol=1e-9)
end

function dipole_color_neutral(r, Q, Qs)
    return exp(-Qs^2 * color_neutral_gamma(r, Q)[1])
end

R=LinRange(0,16,257)


function Gamma_driv(R, Gamma_cn_dat)
    
    Gamma1_cn_dat = similar(Gamma_cn_dat)
    Gamma2_cn_dat = similar(Gamma_cn_dat)


    Gamma_cn_r2 = interpolate(R .^2, Gamma_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma1_cn_dat[i] = ForwardDiff.derivative(Gamma_cn_r2, R2[i])
    end

    Gamma1_cn_r2 = interpolate(R .^2, Gamma1_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma2_cn_dat[i] = ForwardDiff.derivative(Gamma1_cn_r2, R2[i])
    end

    Gamma2_cn_r2 = interpolate(R .^2, Gamma2_cn_dat, SteffenMonotonicInterpolation())

    return (Gamma1_cn_dat, Gamma2_cn_dat)
end

function WW_cn(r, Q, Qs)
    f_dipole = dipole_color_neutral(r, Q, Qs)
    ad = adj_dipole(f_dipole)
    Γ = color_neutral_gamma(r, Q)[1]

    Γ_dri = Gamma_driv(r, Q)

    Γ₁ = Γ_dri[1]
    Γ₂ = Γ_dri[2]

    xG = (1 - ad) * (Γ₁ + r^2 * Γ₂) / Γ
    xh = -(1 - ad) * r^2 * Γ₂ / Γ
    return (xG, xh, f_dipole)
end