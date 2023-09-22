using Cubature
using SpecialFunctions
using Plots
using LaTeXStrings
using QuadGK
using LsqFit
using DelimitedFiles
using Dierckx 
using Measures
using ForwardDiff
using Interpolations


cd(@__DIR__) 

function scatter_style(xl,yl)
    scatter!(
            ylabel=yl, xlabel=xl,
            grid = :off,
            box = :on,
            foreground_color_legend = nothing,
            fontfamily = "serif-roman",
            font="CMU Serif",
            xtickfontsize = 12,
            ytickfontsize = 12,
            xguidefontsize = 12,
            yguidefontsize = 12,
            thickness_scaling=1,
            legendfontsize=10,
            yguidefontrotation=0,
            #legend_font_pointsize=14,
            #legendtitlefontsize=14,
            markersize=1,
            legend=:topright,
            margin=5mm
        )
end

function modelxg(r, param)
    m = abs(param[1])
    Q = abs(param[2])
    γ = abs(param[4])

   # println(param)

    function Γuint(p, r)
        return  p*(1.0 - besselj0(p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 

    function d2Γuint(p, r)
        return p^3*(besselj(0,p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 
    
    Γ = zeros(length(r))
    d2Γ = zeros(length(r))

    Threads.@threads for i = 1:length(r) 
        x = r[i]
        #Γ[i] =  quadgk(p -> Γuint(p,x), 0.0,  10000.0/x, rtol=1e-4)[1]
        d2Γ[i] = quadgk(p -> d2Γuint(p,x), 0.0, 10000.0/x, rtol=1e-4)[1]
    end 

    return  r .* d2Γ * abs(param[3])  #  ./ Γ *param[3]
end



function plotxg(Y, p0)
    
    data = readdlm("xG_$(Y).dat")

    cut = 6.0
    cutmin = 0.15
    # truncation 
    y = data[:,1].*data[:,2].*log.(abs.(data[:,3]))./(1.0.-data[:,3])

    xdata = data[:, 1][(data[:, 1].<cut) .&& (data[:, 1].>cutmin) ]
    ydata = y[(data[:, 1].<cut) .&& (data[:, 1].>cutmin)] 
    
    tmp  = log.(abs.(data[:,3])).*abs.(data[:,5]) ./ data[:,1]
    weights = abs.(tmp[(data[:, 1].<cut) .&& (data[:, 1].>cutmin)])
    
    plot!(xdata,ydata,ribbon=weights, label="Data "*L"Y="*"$Y",xlim=(0.2,6),ylim=(0,3.5)) 
   
    fit = curve_fit( modelxg, xdata, ydata, 1.0 ./ weights, p0, lower=[0.0001, 0.001, 0.01, 0.6],upper=[0.2, 5.0, 100.0, 1.1 ])

   # println(fit.param)
    #println(weights)
    #update initial guess for the next Y
    p0.=fit.param

    xa=collect(0.01:0.05:6)
    ya = modelxg(xa,p0)
   
    plot!(xa,ya, label="", color=:black)

    out = [findmax(ydata)[1] , xdata[findmax(ydata)[2]], p0[1], p0[2], p0[3],p0[4]]

    return out 
end 

Yar = reverse([0.0, 0.1, 0.2, 0.3, 0.4 , 0.5, 0.6 , 0.7 , 0.8] )


maxi = similar(Yar)
max = similar(Yar)
mar = similar(Yar)
Qar = similar(Yar)
Aar = similar(Yar)
γar = similar(Yar)
γ = similar(Yar)

p0=[0.002,0.5,10.0,0.7]




plot()
for (i, x) in enumerate(Yar)
    println(x)

    (max[i],maxi[i],mar[i],Qar[i],Aar[i] ,γar[i]) =@time plotxg(x,p0)

end 

scatter_style(L"r", L"\frac{r xG(r)}{1-D(r)} \ln D(r) ")

savefig("newcombo.pdf")

scatter(Yar,Qar, ylim=(0,1))

scatter_style(L"Y", L"Q/Q_s")

savefig("cnscale.pdf")

print(mar[9])
print(Qar[9])
print(Aar[9])
print(γar[9])

print(mar[1])
print(Qar[1])
print(Aar[1])
print(γar[1])

scatter(Yar,γar)