using CSV
using Tables
using LinearAlgebra
using DataFrames, DataFramesMeta
using StatsBase
using LaTeXStrings
using Plots, StatsPlots, Plots.PlotMeasures
using Distributions
using Suppressor
using Base.Threads

include("clogit.jl")
Random.seed!(1234);

# Graph settings
theme(:wong2)
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
heating_long.oc_increased = begin
    oc_increased = []
    oc = heating_long.oc
    for (i_index, i) = enumerate(sort(unique(heating_long.id)))
        for (j_index, j) = enumerate(sort(unique(heating_long.alt)))
            index = (i_index - 1) * length(unique(heating_long.alt)) + j_index
            if j == 1 || j == 3
                push!(oc_increased, oc[index] * 1.1)
            else
                push!(oc_increased, oc[index])
            end
        end
    end
    Real.(oc_increased)
end

# Market share
function marketshare(data::DataFrame; id = :id)
    choices = sort(unique(data.alt))
    sort!(data, [id, :alt])
    model = myclogit(@formula(chosen ~ ic + oc), data;
        id = id, alt = :alt)
    phat = clogit_predict(@formula(chosen ~ ic + oc), data;
        id = id, alt = :alt,
        α = model.params[1], β = model.params[2])

    phat_increased_cost = clogit_predict(@formula(chosen ~ ic + oc_increased), data;
        id = id, alt = :alt,
        α = model.params[1], β = model.params[2])
    change = Real[]
    for (j_index, j) in enumerate(choices)
        push!(change, mean((phat_increased_cost - phat)[j_index:length(choices):end]))
    end
    return change
end

change = marketshare(heating_long)
println("Change in market share:", change)

function bstrap_marketshare(id::Symbol; bstraps = 1000)
    ids = unique(heating_long.id)
    choices = unique(heating_long.alt)
    changes = Array{Float64, 2}(undef, 0, 0)
    p = Progress(bstraps)
    @info "Computing SE for market share change ($bstraps)"
    for rep = 1:bstraps
        bsample_ids = ids[rand(DiscreteUniform(1, length(ids)), length(ids))]
        data_bsample = DataFrame()
        new_id = []
        for (b_index, bid) in enumerate(bsample_ids)
            append!(data_bsample, subset(heating_long, :id => ByRow(x -> x .== bid)))
            append!(new_id, fill(bid, length(choices)))
        end
        data_bsample.bid = new_id
        local chg
        @suppress chg = marketshare(data_bsample, id = :bid)
        try
            changes = hcat(changes, chg)
        catch
            changes = chg
        end
        next!(p)
    end
    return std(changes, dims = 2)
end

change_se = bstrap_marketshare(:id; bstraps = 1000)
println("SE of changes: ", change_se)
change_se = vec(change_se)

# plot confidence intervals yerror = change_se * 1.96,
change_str = [string(round(chg; digits=4)) for chg in change]
change_se_str = [string(round(se; digits=4)) for se in change_se]
change_se_str = "(" .* change_se_str .* ")"
scatter(collect(1:length(change)), change,
    markersize = 5,
    yerror = change_se * 1.96, legend = false)
annotate!(collect(1:length(change)), change.-0.026,
    text.(latexstring.(change_str), :bottom, 8, :steelblue4))
annotate!(collect(1:length(change)), change.-0.039,
    text.(latexstring.(change_se_str), :bottom, 8, :steelblue4))
plot!(xticks = ([1, 2, 3, 4, 5], ["Heat pump", "Gas central",
    "Electric central", "Gas room",
    "Electric room"]))
plot!(xlims = [0.25, 5.75], ylims = [-0.2, 0.2],
    ylabel = "Change in market share")
hline!([0], line = :dash)
savefig("../output/marketshare_change_bstrap_se.pdf")
