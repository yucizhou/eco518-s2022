using StatsModels
using Optim, Distributions
using LinearAlgebra, FiniteDifferences
using Random
using ProgressMeter
using Tullio

function myclogit(formula::FormulaTerm, data::DataFrame; base = nothing,
        id::Symbol, alt::Symbol, se = true, alpha = true)
    sort!(data, [id, alt])
    @show formula
    formula = apply_schema(formula, schema(formula, data))
    chosen, X = modelcols(formula, data)
    ids = sort(unique(data.id))
    choices = sort(unique(data.alt))
    if isnothing(base)
        base = minimum(choices)
    end

    # Log likelihood of sample
    function loglike_clogit(a, b)
        loglike = 0
        α = zeros(length(choices))
        k = 1
        if alpha
            for (j_index, j) in enumerate(choices)
                if j != base
                    α[j_index] = a[k]
                    k += 1
                end
            end
        end
        β = b
        for (i_index, i) in enumerate(unique(ids))
            base_index = (i_index-1) * length(choices) + 1
            for (j_index, j) in enumerate(choices)
                index = base_index + (j_index - 1)
                loglike += chosen[index] * (α[j_index] + β' * X[index, :]) - chosen[index] * log(sum(exp(α[k] + β' * X[base_index + (k - 1), :]) for k = 1:length(choices)))
            end
        end
        return loglike
    end

    params_cnt = length(choices) + size(X, 2) - 1
    @info "Number of parameters:" params_cnt
    #Define sample scores
    sample_score!(storage, θ) = begin
        s = similar(storage)
        for (i_index, i) in enumerate(ids)
            base_index = (i_index-1) * length(choices) + 1
            s .+= myscore_clogit(θ[1:length(choices)-1], θ[length(choices):end];
                choice = chosen[base_index:base_index + length(choices) - 1],
                choices = choices,
                x = X[base_index:base_index + length(choices) - 1, :],
                base = base)
        end
        copyto!(storage, -s ./ length(ids))
    end
    mle = optimize(params ->
        -loglike_clogit(params[1:length(choices)-1], params[length(choices):end]),
        zeros(params_cnt))
    loglikelihood = -Optim.minimum(mle)
    params = Optim.minimizer(mle)
    println("Log likelihood: ", loglikelihood)
    println("Coefficients: ", params)
    a = params[1:length(choices)-1]
    α = zeros(length(choices))
    k = 1
    if alpha
        for (j_index, j) in enumerate(choices)
            if j != base
                α[j_index] = a[k]
                k += 1
            end
        end
    end
    b = β = params[length(choices):end]

    # Standard error estimation
    # sandwich formula robust to MLE misspecification
    function clogitse(a, b)
        N = length(ids)
        Ω = H = zeros(params_cnt, params_cnt)
        θ = vcat(a, b)
        for (i_index, i) in enumerate(ids)
            base_index = (i_index-1) * length(choices) + 1
            s(θ) = myscore_clogit(θ[1:length(choices)-1], θ[length(choices):end];
                choice = chosen[base_index:base_index + length(choices) - 1],
                choices = choices,
                x = X[base_index:base_index + length(choices) - 1, :],
                base = base)
            Ω += s(θ) * s(θ)'
            H += jacobian(central_fdm(7, 1), s, θ)[1]
        end
        invH = inv(H)
        return invH * Ω * invH
    end
    if se
        V = clogitse(a, b)
        se = sqrt.(abs.(diag(V)))
        return (loglike = loglikelihood, params = (α, β), se = se)
    else
        return (loglike = loglikelihood, params = (α, β))
    end
end

# score for one individual
# a includes input for base level
function myscore_clogit(a, b; choice, choices, x, base)
    p = Real[]
    α = zeros(length(choices))
    k = 1
    for (j_index, j) in enumerate(choices)
        if j != base
            α[j_index] = a[k]
            k += 1
        end
    end
    β = b
    for (j_index, j) in enumerate(choices)
        push!(p, exp(α[j_index] + x[j_index, :]' * β) / sum(exp(α[k] + x[k, :]' * β) for k = 1:length(choices)))
    end
    base_index = findfirst(choices .== base)
    score_a = choice[1:end .!= base_index] - p[1:end .!= base_index]
    score_b = Real[]

    for x_index = 1:size(x, 2)
        push!(score_b, sum((choice[k] - p[k]) * x[k, x_index] for k = 1:length(choices)))
    end
    return vcat(score_a, score_b)
end

function clogit_predict(formula::FormulaTerm, data::DataFrame; base = nothing,
        id::Symbol, alt::Symbol, α::Vector, β::Vector)
    sort!(data, [id, alt])
    @show formula
    formula = apply_schema(formula, schema(formula, data))
    chosen, X = modelcols(formula, data)
    ids = sort(unique(data.id))
    choices = sort(unique(data.alt))
    p_sample = []
    for (i_index, i) in enumerate(ids)
        p = []
        x = X[(i_index - 1) * length(choices)+1:i_index* length(choices), :]
        for (j_index, j) in enumerate(choices)
            push!(p, exp(α[j_index] + x[j_index, :]' * β) / sum(exp(α[k] + x[k, :]' * β) for k = 1:length(choices)))
        end
        p_sample = vcat(p_sample, p)
    end
    return p_sample
end
