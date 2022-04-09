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
V_test = V.robust[[2, 4], [2, 4]]

N = length(Y)
wald(b2, b4) = N * [β[2] - b2; β[4] - b4]' * inv(V_test) * [β[2] - b2; β[4] - b4]
crit = cquantile(Chisq(2), 0.05)
CR(b2, b4; crit = crit) = (wald(b2, b4) > crit) ? 1 : 0

b2grid = -1.2:0.0005:-0.55
b4grid = 0.05:0.0005:0.25
plot(b2grid, b4grid, CR, st=:contour, fill = true,
    colorbar = false)
plot!(xlabel = "# Children less than 6 y.o.",
    ylabel = "Years of education")
savefig("../output/kidlt6_educ_confidenceset.pdf")
