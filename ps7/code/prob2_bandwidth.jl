using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using GLM
using StatsBase
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using Random


include("kernals.jl")
include("kreg.jl")

Random.seed!(1234);

# Graph settings
theme(:wong)
default(fontfamily = "Computer Modern")

# Prep data
nls = DataFrame(CSV.File("../raw/nls.csv"))

# Question 2: Semi parametric method
function choose_bandwidth(hgrid, CV_grid)
    CV_grid[isnan.(CV_grid)] .= Inf
    CV_argmin = findmax(-CV_grid)
    return h_cv = hgrid[CV_argmin[2]]
end

hgrid = collect(0.3:0.02:1)
println("Choosing bandwidth for Y")
@time CV_grid = CV_kreg.(hgrid, Y = nls.luwe, X = nls.exper)
h_Y = choose_bandwidth(hgrid, CV_grid)
println("The optimal bandwidth for Y (log earnings) is $h_Y.")


println("Choosing bandwidth for X")
@time CV_grid = CV_kreg.(hgrid, Y = nls.educ, X = nls.exper)
h_X = choose_bandwidth(hgrid, CV_grid)
println("The optimal bandwidth for X (education) is $h_X.")
