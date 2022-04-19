using Optim, Distributions
using LinearAlgebra, FiniteDifferences
using Random
using ProgressMeter
using Suppressor

#=
Runs Probit model for a sample (Y, X)
Options: intercept - whether to include intercept
se - whether to report SE calculation
Returns coefficients
(first one is intercept, other coefs ordered as in the data)
if se = true, returns se: vectors of SE based on diff calculations
V: variance-covariance matrices of the asymptotic distribution
=#
function myprobit(Y::Vector, X::Matrix; intercept = true, se = true)
    if intercept == true
        X = hcat(ones(length(Y)), X)
    end
    k = size(X, 2)
    N = length(Y)
    mle = optimize(b -> -loglike_probit(b; Y = Y, X = X), zeros(k))
    loglikelihood = -Optim.minimum(mle)
    β = Optim.minimizer(mle)
    println("Log likelihood: ", loglikelihood)
    println("Coefficients: ", β)
    if se == true
        V = probitse(Y, X, β)
        se_hess = sqrt.(diag(V.hess) ./ N)
        se_score = sqrt.(diag(V.score) ./ N)
        se_robust = sqrt.(diag(V.robust) ./ N)
        se = (hess = se_hess, robust = se_robust, score = se_score)
        println("Standard error: ")
        @show se_hess se_score se_robust
        return β, se, V
    else
        @show β
        return β
    end
end

# Log likelihood for an entire sample (Y, X)
function loglike_probit(β; Y::Vector, X::Matrix)
    Φ(x) = cdf(Normal(), x)
    φ(x) = pdf(Normal(), x)
    loglikelihood = 0
    for i = 1:length(Y)
        loglikelihood += Y[i] * log(Φ(X[i, :]' * β)) +
            (1 - Y[i])*log(1 - Φ(X[i, :]' * β))
    end
    return loglikelihood
end

# Score for one observation
function myscore_probit(β; y, x)
    Φ(z) = cdf(Normal(), z)
    φ(z) = pdf(Normal(), z)
    score = ( (y - Φ(x' * β)) / (Φ(x' * β)*(1 - Φ(x' * β)) ) ) * φ(x' * β)*x
    return score
end

# SE based on: negative inverse of Hessian, outer product of scores,
# and sandwich formula robust to MLE misspecification
function probitse(Y::Vector, X::Matrix, β::Vector)
    Φ(x) = cdf(Normal(), x)
    φ(x) = pdf(Normal(), x)
    N = length(Y)
    H = zeros(size(X, 2), size(X, 2))
    Ω = zeros(size(X, 2), size(X, 2))
    for i = 1:N
        s(b) = myscore_probit(b; y = Y[i], x = X[i, :])
        Ω += s(β) * s(β)'
        H += jacobian(central_fdm(5, 1), s, β)[1]
    end
    H = 1/N * H
    Ω = 1/N * Ω
    V_hess = -inv(H)
    V_score = inv(Ω)
    V_robust = inv(H) * Ω * inv(H)
    return (hess = V_hess, score = V_score, robust = V_robust)
end

#= Marginal/avg marginal effetc of probit model
Input: coefs (ordered), Data (Y, X)
which_coef: which variable is of interest (same index as [coefs] vector)
at: fix covariates at which specific value
(if not specified, input [NaN] and will calculate average partial effect
of the sample)
Options:
intercept: does the coef vector contain the first entry as intercept
bstraps: for standard error calculation - how many bootstrap reps
discrete: whether the var of interest is a discrete variable

Returns marginal effect, bootstrap standard error if bstraps > 0
=#
function margins(coefs::Vector, Y::Vector, X::Matrix; which_coef::Int, at = nothing,
    intercept = true, bstraps = 0, discrete = false)
    if isnothing(at)
        at = fill(NaN, size(X, 2))
    end
    if intercept
        X = hcat(ones(length(Y)), X)
        at = vcat(1, at)
    end
    at_X = similar(X)
    marginal_effect = []
    for i = 1:size(X, 1)
        at_X[i, :] = X[i, :] .* isnan.(at) + at .* .!isnan.(at)
        push!(marginal_effect, margins_calc(at_X[i, :],
            coefs, which_coef; discrete = discrete))
    end
    if bstraps == 0
        return mean(marginal_effect)
    else
        marginal_effect_bstrap = []
        p = Progress(bstraps)
        @info "Computing SE for marginal effects ($bstraps)"
        for rep = 1:bstraps
            bsample_index = rand(DiscreteUniform(1, size(X, 1)), size(X, 1))
            local coefs_bsample
            @suppress coefs_bsample = myprobit(Y[bsample_index], X[bsample_index, :];
                se = false, intercept = false)
            at_X_bsample = at_X[bsample_index, :]
            marginal_effect_bsample = []
            for i = 1:size(X, 1)
                push!(marginal_effect_bsample, margins_calc(at_X_bsample[i, :],
                    coefs_bsample, which_coef; discrete = discrete))
            end
            push!(marginal_effect_bstrap, mean(marginal_effect_bsample))
            next!(p)
        end
        return mean(marginal_effect), std(marginal_effect_bstrap)
    end
end

#=
Calculate marginal effects for one observation
Inputs vector of X, coefs, which variable of interest
Option: discrete - whether the variable of interest is a discrete var
if so calculate marginal effect as first differences of CDF.
O/w calc as derivative of link function
Returns the marginal effect
=#
function margins_calc(x, coefs, which_coef; discrete = false)
    g = x -> pdf(Normal(), x)
    G = x -> cdf(Normal(), x)
    if discrete == false
        return g.(x' * coefs) * coefs[which_coef]
    else
        add_one = zeros(length(x))
        add_one[which_coef] += 1
        return G.((x + add_one)' * coefs) - G.(x' * coefs)
    end
end
