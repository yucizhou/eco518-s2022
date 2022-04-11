using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using StatsBase
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings

include("iv.jl")

# Graph settings
theme(:wong)
default(fontfamily = "Computer Modern")

# Prep data
fish = DataFrame(CSV.File("../raw/fish.csv"))

Y = fish[!, :logq]
X = fish[!, :logp]
Z = Matrix(fish[!, [:stormy, :mixed]])

# 2SLS
β_2sls, se_2sls, Ω, _ = lineariv(Y, X, Z)
println("2sls results: $β_2sls ($se_2sls)")
# Efficient weighting matrix
W_eff = inv(Ω)
β_eff, se_eff, _, J = lineariv(Y, X, Z; W = W_eff)
println("Efficient GMM results: $β_eff ($se_eff)")
println("Overidentification J statistic: $J")

bgrid = collect(-4:0.01:1)
_, ci_index = AndersonRubin(Y, X, Z; grid = bgrid)
b = bgrid[ci_index]
y = ones(length(b))
plot(b, y, st = :scatter, xlims = (-4, 1), ylims = (0.5, 1.5),
    ytick = false, legend = false, xticks = -4:.5:1,
    xlabel = "Anderson-Rubin CI")
savefig("../output/andersonrubin.pdf")
