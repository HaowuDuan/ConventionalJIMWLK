using Cubature
using SpecialFunctions
using Plots
using LaTeXStrings
using QuadGK
using LsqFit
using DelimitedFiles
using Statistics


cd(@__DIR__) 

data = readdlm("test.dat")
dipoleData = readdlm("dipole.dat")
Gamma_data_y1 = readdlm("Gamma_data_y1.dat")
Gamma_data = readdlm("Gamma_data.dat")



xdata = data[:, 1][(data[:, 1].<4)  ]
# get rid of negative y values
ydata = data[:, 2][(data[:, 1].<4)] .- data[1, 2]


function Γ_3(r, param)
    A = param[4]
    m = param[1]
    Q = param[2]
    γ = param[3]

    Γ, err = quadgk(p->A*(p*2)*((1 .- besselj0.(p.*r))/((p.^2 + m^2)^2))*((p.^2)/(Q^2*π))*((Q^2/p.^2)^γ)/(1+(Q^2/p.^2)^γ), 0, Inf)
    return Γ
end
function D(r, param) ### this function is not correct yet 
    return exp.(-2*π*param[2]^2*Γ_3(r,param))
end

function create_weighting_vector(xdata,ydata,param)
    #print(xdata)
    y = Γ_3(xdata,param)
    #print(y)
    error = abs.(ydata-y)
    maxErr = maximum(error)
    #print(maxErr)
    scale = 1 / (var(error))
    #scale = 1 / (var(error))^(2)
    #print(scale)
    return error.*scale
end



#=
p0 = [.2, 0.5, 0.7,1.0]

fit = curve_fit(Γ_3, xdata, ydata, p0)
print(fit.param)
m, Q, γ, A = fit.param

xfit_short = range(0,xdata[length(xdata)], length=length(xdata))
yfit_short = Γ_3(xfit_short, fit.param)

plot(xdata,ydata,label="Raw Data")
plot!(xfit_short,yfit_short,label="Fitted Curve: m=$(m), Q = $(Q), γ= $(γ)")
xlabel!(L"r")
ylabel!(L"Fitted $Γ_2(r)$ and Raw Data")
=#
##### 
#=
xdata_Gamma = Gamma_data[:, 1][(Gamma_data[:, 1].<6)  ]
# get rid of negative y values
ydata_Gamma = Gamma_data[:, 2][(Gamma_data[:, 1].<6)]


p0 = [.2, 0.55, 1.0,1.0]

fit = curve_fit(Γ_3, xdata_Gamma, ydata_Gamma, p0)
print(fit.param)
m, Q, γ, A = fit.param

xfit_short = range(0,xdata_Gamma[length(xdata_Gamma)], length=length(xdata_Gamma))
yfit_short = Γ_3(xfit_short, fit.param)

plot(xdata_Gamma,ydata_Gamma,label="Raw Data")
plot!(xfit_short,yfit_short,label="Fitted Curve: m=$(m), Q = $(Q), γ= $(γ)")
xlabel!(L"r")
ylabel!(L"Fitted $Γ_2(r)$ and Raw Data")
title!("Gamma Data")
=#


##### Fitting function that uses weighed least squares based off error in first iteration
#=
xdata_Gamma_y1 = Gamma_data_y1[:, 1][(Gamma_data_y1[:, 1].<5)  ]
# get rid of negative y values
ydata_Gamma_y1 = Gamma_data_y1[:, 2][(Gamma_data_y1[:, 1].<5)]


p0 = [.1, 0.4, 0.7,1]

fit = curve_fit(Γ_3, xdata_Gamma_y1, ydata_Gamma_y1, p0)
print("\n Unweighted fit: ", fit.param)
m, Q, γ, A = fit.param

xfit_short = range(0,xdata_Gamma_y1[length(xdata_Gamma_y1)], length=length(xdata_Gamma_y1))
yfit_short = Γ_3(xfit_short, fit.param)

#now weighted least squares
wt_Gamma_y1 = create_weighting_vector(xdata_Gamma_y1,ydata_Gamma_y1,fit.param)
fit = curve_fit(Γ_3, xdata_Gamma_y1, ydata_Gamma_y1, wt_Gamma_y1, p0)
print("\n weighted fit: ", fit.param)
m_w, Q_w, γ_w, A_w = fit.param
yfit_weight = Γ_3(xfit_short, fit.param)

print("\n Average Error Unweighted: ", mean(abs.(ydata_Gamma_y1.-yfit_short)))
print("\n Average Error Weighted: ", mean(abs.(ydata_Gamma_y1.-yfit_weight)))

plot(xdata_Gamma_y1,ydata_Gamma_y1,label="Raw Data")
plot!(xfit_short,yfit_short,label="Fitted Curve: m=$(m), Q = $(Q), γ= $(γ)")
plot!(xfit_short,yfit_weight,label="Weight Fitted Curve: m=$(m_w), Q = $(Q_w), γ= $(γ_w)")
xlabel!(L"r")
ylabel!(L"Fitted $Γ_2(r)$ and Raw Data")
title!("Gamma Data y1")
#savefig("graph1.png")
=#


##### dipole fit is not working correctly
#=
xdata_dipole = dipoleData[:, 1][(dipoleData[:, 1].<7)  ]
# get rid of negative y values
ydata_dipole = dipoleData[:, 2][(dipoleData[:, 1].<7)]


p0 = [.2, 0.5, 0.7,1.0]

fit = curve_fit(D, xdata_dipole, ydata_dipole, p0)
print(fit.param)
m, Q, γ, A = fit.param

xfit_short = range(0,xdata_dipole[length(xdata_dipole)], length=length(xdata_dipole))
yfit_short = Γ_3(xfit_short, fit.param)

plot(xdata_dipole,ydata_dipole,label="Raw Data")
plot!(xfit_short,yfit_short,label="Fitted Curve: m=$(m), Q = $(Q), γ= $(γ)")
xlabel!(L"r")
ylabel!(L"Fitted $D_2(r)$ and Raw Data")
title!("Dipole Data")
=#

#y = create_weighting_vector(xdata_Gamma_y1,ydata_Gamma_y1,[1,1,1,1])

