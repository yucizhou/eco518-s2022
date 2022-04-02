using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using StatsBase
using Plots, StatsPlots
using LaTeXStrings

include("kernals.jl")
include("kdensity.jl")
include("kreg.jl")
include("locpoly.jl")
include("polyseries.jl")

# Graph settings
theme(:wong)
default(fontfamily = "Computer Modern")

# Prep data
engel = DataFrame(CSV.File("../raw/engel.csv"))
engel.loginc = log.(engel.income)
engel.share = engel.foodexp ./ engel.income


# Question 1: Kdensity of log income
# with histograms
f_loginc = kdensity(engel.loginc)
plot(engel.loginc, seriestype=:histogram, legend = false, color = :goldenrod2,
    ylabel = "Counts")
plot!(twinx(), f_loginc, 5, 9, legend = false, color = :deepskyblue3, linewidth = 3,
    ylabel = "Density")
plot!(xlabel = "Log income")
savefig("../output/kdensity_loginc_normal.pdf")


# Question 2: Kernal regression of share food expenditures on log income
# Cross-validation grid search
hgrid = collect(0.01:0.01:1)
CV_grid = CV_kreg.(hgrid, Y = engel.share, X = engel.loginc)
CV_grid[isnan.(CV_grid)] .= Inf
CV_argmin = findmax(-CV_grid)
h_cv = hgrid[CV_argmin[2]]
CV_min = -CV_argmin[1]

println("The optimal bandwidth is $h_cv.")
plot(hgrid, CV_grid, color = :deepskyblue3, linewidth = 3)
plot!([h_cv], [CV_min], seriestype = :scatter, markersize = 5, color = :goldenrod2,
    annotations = (h_cv + 0.01, CV_min + 0.0001, L"h_{CV} = 0.28"),
    annotationfontsize = 10, legend = false)
plot!(xlabel = L"h", ylabel = L"CV(h)")
savefig("../output/kreg_cv.pdf")

# # Use optimal bandwidth for kernal regression
m = kreg(engel.share, engel.loginc, h = h_cv)
plot(engel.loginc, engel.share, seriestype = :scatter, markersize = 4)
plot!(m, 5, 9, legend = false, linewidth = 3)
plot!(xlabel = "Log income", ylabel = "Share of food expenditures")
savefig("../output/kreg_scatter.pdf")


# Question 3: local linear regression
hgrid = collect(1:0.01:2)
CV_grid = CV_locpoly.(hgrid, Y = engel.share, X = engel.loginc, p = 1)
CV_grid[isnan.(CV_grid)] .= Inf
CV_argmin = findmax(-CV_grid)
h_cv = hgrid[CV_argmin[2]]
CV_min = -CV_argmin[1]
println("The optimal bandwidth for local linear regression is $h_cv.")
plot(hgrid, CV_grid, color = :deepskyblue3, linewidth = 3)
plot!([h_cv], [CV_min], seriestype = :scatter, markersize = 5, color = :goldenrod2,
    annotations = (h_cv + 0.03, CV_min + 2 * 10^-7, L"h_{CV} = 1.25"),
    annotationfontsize = 10, legend = false)
plot!(xlabel = L"h", ylabel = L"CV(h)")
savefig("../output/loclinear_cv.pdf")

x0, m = locpoly(engel.share, engel.loginc, h = h_cv)
plot(engel.loginc, engel.share, seriestype= :scatter)
plot!(x0, m, legend = false, linewidth = 3)
plot!(xlabel = "Log income", ylabel = "Share of food expenditures")
savefig("../output/loclinear_scatter.pdf")

# Question 4: polynomial series regression
pgrid = collect(1:1:10)
CV_grid = CV_polyseries.(pgrid, Y = engel.share, X = engel.loginc)
CV_grid[isnan.(CV_grid)] .= Inf
CV_argmin = findmax(-CV_grid)
p_cv = pgrid[CV_argmin[2]]
CV_min = -CV_argmin[1]
println("The optimal order for polynomial series regression is $p_cv.")
plot(pgrid, CV_grid, color = :deepskyblue3, linewidth = 3)
plot!([p_cv], [CV_min], seriestype = :scatter, markersize = 5, color = :goldenrod2,
    annotations = (p_cv + -.5, CV_min + 20 , L"p_{CV} = 2"),
    annotationfontsize = 10, legend = false)
plot!(xlabel = L"p", ylabel = L"CV(p)")
savefig("../output/polyseries_cv.pdf")

x0, m = polyseries(engel.share, engel.loginc, order = p_cv)
plot(engel.loginc, engel.share, seriestype= :scatter)
plot!(x0, m, legend = false, linewidth = 3)
plot!(xlabel = "Log income", ylabel = "Share of food expenditures")
savefig("../output/polyseries_scatter.pdf")
