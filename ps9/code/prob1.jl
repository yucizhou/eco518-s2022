using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using StatsBase
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings

include("probit.jl")

# Graph settings
theme(:wong)
default(fontfamily = "Computer Modern")

# Prep data
mroz = DataFrame(CSV.File("../raw/mroz.csv"))

Y = mroz[!, :part]
X = Matrix(mroz[!, [:kidslt6, :age, :educ, :nwifeinc]])

println("Coefficient/SE order: ")
println("Intercept, :kidslt6, :age, :educ, :nwifeinc")

β, se, V = myprobit(Y, X)

println("Marginal effect at sample mean: educ")
@show me_educ_atmean = margins(β, Y, X; which_coef = 4, at = vec(mean(X, dims = 1)),
    bstraps = 100)

println("Avg partial effect: educ")
@show me_educ_avgpartial = margins(β, Y, X; which_coef = 4, bstraps = 100)

println("Avg partial effect: kids from 0 to 1")
@show me_kidslt6 = margins(β, Y, X; which_coef = 2, at = [0.0, NaN, NaN, NaN],
    bstraps = 100, discrete = true)
