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

# Question 2: point estimates
_, Yhat = kreg(nls.luwe, nls.exper, x0grid = nls.exper, h = h_Y)
_, Xhat = kreg(nls.educ, nls.exper, x0grid = nls.exper, h = h_X)

nls.resid_Y = nls.luwe - Yhat
nls.resid_X = nls.educ - Xhat

doubleresid = lm(@formula(resid_Y ~ 0 + resid_X), nls)
@show GLM.coeftable(doubleresid)
b_doubleresid = GLM.coef(doubleresid)[1]

# Compare Mincer and Robinson confidence intervals
mincer = lm(@formula(luwe ~ educ + exper + exper^2), nls)
b_mincer = GLM.coef(mincer)[2]
@show GLM.coeftable(mincer)
se_mincer = stderror(HC0(), mincer)[2]

# plot confidence intervals
plot([1, 2], [b_mincer, b_doubleresid], seriestype = :scatter,
    yerror = [se_mincer * 1.96, 0], legend = false)
plot!(xticks = ([1, 2], ["Mincer (1976)", "Semiparametric"]))
plot!(xlims = [0.25, 2.75], ylims = [0, 0.2],
    ylabel = L"\beta_0")
savefig("../output/nls_pointestimates.pdf")
