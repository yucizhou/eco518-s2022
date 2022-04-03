using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using GLM
using StatsBase
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using Random
using Base.Threads
using CovarianceMatrices
using ProgressMeter


include("kernals.jl")
include("kreg.jl")

Random.seed!(1234);

# Graph settings
theme(:wong)
default(fontfamily = "Computer Modern")

# Prep data
nls = DataFrame(CSV.File("../raw/nls.csv"))

# Calibrate bandwidth
h_Y = 0.9
h_X = 0.74

# Question 3: Bootstrapping the semi parametric SE
_, Yhat = kreg(nls.luwe, nls.exper, x0grid = nls.exper, h = h_Y)
_, Xhat = kreg(nls.educ, nls.exper, x0grid = nls.exper, h = h_X)

nls.resid_Y = nls.luwe - Yhat
nls.resid_X = nls.educ - Xhat

doubleresid = lm(@formula(resid_Y ~ 0 + resid_X), nls)
@show GLM.coeftable(doubleresid)
b_doubleresid = GLM.coef(doubleresid)[1]

function bstrap(; reps = 500, h_Y = 0.9, h_X = 0.74)
    coef = Array{Float64, 1}(undef, reps)
    N = length(nls.luwe)
    draw = Array{Int, 1}(undef, N)

    p = Progress(reps)
    println("Computing bootstraps ($reps)")
    @threads for i = 1:reps
        rand!(draw, 1:N)
        bsample = DataFrame(luwe = nls.luwe[draw],
            exper = nls.exper[draw], educ = nls.educ[draw])
        _, bsample.Yhat = kreg(bsample.luwe, bsample.exper, x0grid = bsample.exper, h = h_Y)
        _, bsample.Xhat = kreg(bsample.educ, bsample.exper, x0grid = bsample.exper, h = h_X)
        bsample.resid_Y = bsample.luwe - bsample.Yhat
        bsample.resid_X = bsample.educ - bsample.Xhat

        bestim = lm(@formula(resid_Y ~ 0 + resid_X), bsample)
        coef[i] = GLM.coef(bestim)[1]
        next!(p)
    end
    se = sqrt(var(coef))
    println("Bootstrap standard error ($reps reps) is ", se)
    return se
end

@time se_resid = bstrap(reps = 100, h_Y = 0.9, h_X = 0.74)

# Compare Mincer and Robinson confidence intervals
mincer = lm(@formula(luwe ~ educ + exper + exper^2), nls)
b_mincer = GLM.coef(mincer)[2]
se_mincer = stderror(HC0(), mincer)[2]
# plot confidence intervals
plot([1, 2], [b_mincer, b_doubleresid], seriestype = :scatter,
    yerror = [se_mincer * 1.96, se_resid * 1.96], legend = false)
plot!(xticks = ([1, 2], ["Mincer (1976)", "Semiparametric"]))
plot!(xlims = [0.25, 2.75], ylims = [0, 0.2],
    ylabel = L"\beta_0")
savefig("../output/nls_bstrap_se.pdf")
