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



function modelgamma(r, param)
    γ = abs(param[1])
    Q = param[2]

    function Γuint(p, r)
        return  p*(1.0 - besselj0(p*r))/(p^2+0.2^2)^2*(1.0/p^2)^(γ-1)/(1.0+(Q^2/p^2)^γ) 
    end

    Γ = zeros(length(r))
 
    Threads.@threads for i = 1:length(r) 
        x = r[i]
        Γ[i] =  quadgk(p -> Γuint(p,x), 0.0, Inf, rtol=1e-4)[1]
    end 
    Γ * param[3]
end


function gamma(Y)


    print(Y)
    
    data = readdlm("S_$(Y).dat")


    

    #data = readdlm("xh_$(Y).dat")

    #x = log.(data[:,1])[1:2]
    #y = log.(abs.(log.(abs.(data[:,2]))))[1:2]

    x = data[:,1][1:6]
    y = -log.(abs.(data[:,2]))[1:6]

    #scatter(log.(data[:,1]),log.(abs.(log.(abs.(data[:,2])))))
    scatter(x,y)

    function linearmod(l,p)
        return 2.0*p[1]*l .+ p[2]
    end

    fit = curve_fit(modelgamma , x, y, [0.7, 0.3, 4.0])

    println(fit.param)

    plot!(x,modelgamma(x,fit.param))
    abs(fit.param[1])
end




function model(r, param, γ )
    m = param[1]
    Q = param[2]



    function Γuint(p, r)
        return  p*(1.0 - besselj0(p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 

    function d2Γuint(p, r)
        return p^3*(besselj(2,p*r))/(p^2 + m^2)^2*p^2*(Q^2/p^2)^γ/(1.0+(Q^2/p^2)^γ)
    end 
    
    Γ = zeros(length(r))
    d2Γ = zeros(length(r))

    Threads.@threads for i = 1:length(r) 
        x = r[i]
        Γ[i] =  quadgk(p -> Γuint(p,x), 0.0, Inf, rtol=1e-4)[1]
        d2Γ[i] = quadgk(p -> d2Γuint(p,x), 0.0, 10000.0/x, rtol=1e-4)[1]
    end 

    return  2.0*r.^2 .* d2Γ ./ Γ *param[3]
end

function fitplot(Y, p0)

    
    print(Y)
    
    data = readdlm("xh_$(Y).dat")


    cut = 3

    # truncation 
    xdata = data[:, 1][(data[:, 1].<cut) .&& (data[:, 1].>0.8) ]
    ydata = data[:,4][(data[:, 1].<cut) .&& (data[:, 1].>0.8)] 
    weights = data[:,5][(data[:, 1].<cut) .&& (data[:, 1].>0.8)] 

    # preview 
    plot(xdata, ydata)





    #p0 = [.1,  6.6, 0.7, 2.0]

    #p0 = [.8,  14.0, .7, 2.0]
    
    γ = gamma(Y)
    scatter(xdata,ydata,yerr=weights, label="Data"*L"Y=$(Y)")
  
    xfit_short = range(0,xdata[length(xdata)], length=length(xdata))
    yfit_short = model(xfit_short, p0,  γ)

    plot!(xfit_short,yfit_short, color=:black, lw=2, label="Init")



    fit = curve_fit( (x,p) -> model(x,p, γ), xdata, ydata, 1.0 ./ weights, p0, maxIter=200,lower=[-0.7, 0.001, 0.8],upper=[0.7, 20.0, 1.3 ]; autodiff=:finiteforward)

    print(fit.param)

    m, Q, A = fit.param
    m1, Q1, A1 = map(x->round(x,digits=4),fit.param)
    γ1 = round(γ,digits=4)

  
    yfit_short = model(xfit_short, fit.param,  γ)
    

    plot!(xfit_short,yfit_short, color=:black, lw=2, label="Fitted Curve: m=$(m1), Q = $(Q1), γ= $(γ1), A= $(A1)")

    #plot!(xfit_short,yfit_short, color=:black, lw=2)

    scatter_style(L"r", L"xh/(1-D)")

    savefig("figure_$(Y).pdf")
    p0 .= fit.param
    fit.param
end 


par=[0.5, 15.0, 1.0 ]

Yar = reverse([0.0, 0.1, 0.2, 0.3, 0.4 , 0.5, 0.6 , 0.7 , 0.8, 0.9, 1.0] )
output=map((x) -> fitplot(x,par) , Yar)


fitplot(0.5, par)
plot!()



outputgamma=map((x) -> gamma(x) , Yar)


scatter(Yar, outputgamma)

scatter_style(L"Y",L"\gamma")

savefig("gama.pdf")

scatter(Yar[4:11],map(x->x[2],output)[4:11],label="")


scatter_style(L"Y",L"Q")

savefig("Q.pdf")




function Qsfunc(Y)


    print(Y)
    
    data = readdlm("S_$(Y).dat")

    Qsfunc = Spline1D(data[:,1], data[:,2] .- exp(-0.5),  k=3, bc="nearest")

    roots(Qsfunc)

    Qs=sqrt(2.0)/roots(Qsfunc)[1]
    return Qs
end 

Qsfunc(0.2)

outputQs=map((x) -> Qsfunc(x) , Yar)


scatter!(Yar,8*Yar.-2.5)


scatter(Yar,log.(outputQs))
scatter!(Yar,Yar.-0.65)


scatter(Yar,(map(x->x[2],output)))

scatter(Yar,outputQs,label="")

scatter_style(L"Y",L"Q_s")

savefig("Qs.pdf")

scatter((map(x->x[2],output))[4:11], outputQs[4:11]  ,label="")

scatter_style(L"Q",L"Q_s")

savefig("QsvsQ.pdf")



scatter(Yar,(map(x->x[3],output)))
