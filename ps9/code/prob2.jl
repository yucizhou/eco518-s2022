using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using StatsBase
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using Distributions

include("clogit.jl")

# Graph settings
theme(:wong)
default(fontfamily = "Computer Modern")

# Prep data
heating = DataFrame(CSV.File("../raw/heating.csv"))
rename!(var -> replace(var, " " => "_"), heating)
heating.id = collect(1:size(heating, 1))

# Reshaping
recode(alt) = begin
    alt_order = ["heat_pump", "gas_central",
        "electric_central", "gas_room",
        "electric_room"]
    return findfirst(alt_order .== alt) - 1
end

ic = transform(rename!(stack(heating, Symbol.(names(heating[:, r"_ic"])),
    [:id]),
    [:value => :ic, :variable => :alt]),
    :alt => (var -> recode.(replace.(var, "_ic" => ""))) => :alt)
oc = transform(rename!(stack(heating, Symbol.(names(heating[:, r"_oc"])),
    [:id]),
    [:value => :oc, :variable => :alt]),
    :alt => (var -> recode.(replace.(var, "_oc" => ""))) => :alt)
heating_long = innerjoin(ic, oc, on = [:id, :alt] )
heating_long = innerjoin(heating_long, select(heating, :id, :choice), on = :id)
sort!(heating_long, [:id, :alt])
heating_long.chosen = Int.(heating_long.choice .== heating_long.alt)

@show heating_clogit = myclogit(@formula(chosen ~ ic + oc), heating_long;
    id = :id, alt = :alt)
@show heating_clogit_noalpha = myclogit(@formula(chosen ~ ic + oc), heating_long;
    id = :id, alt = :alt, alpha = false)

r = length(heating_clogit.params[1]) - 1
lr_stat = 2(heating_clogit.loglike - heating_clogit_noalpha.loglike)
lr_pvalue = ccdf(Chisq(r), lr_stat)
println("LR stat: ", lr_stat)
println("P_value: ", lr_pvalue)
