using Optim, Distributions
using LinearAlgebra
using ProgressMeter
using Base.Threads
using GLM

function lineariv(Y::Vector, X::Union{Vector,Matrix}, Z::Union{Vector,Matrix};
    W = nothing, intercept = true)
    N = length(Y)
    if intercept
        X = hcat(ones(length(Y)), X)
        Z = hcat(ones(length(Y)), Z)
    end
    k = size(X, 2)
    r = size(Z, 2)

    if isnothing(W)
        println("Estimating using 2SLS")
        W = inv(sum(Z[i, :] * Z[i, :]' for i = 1:N) / N)
    end
    # Moment condition: E(e_t z_t) = 0
    @info "Weight: " W
    g(b; y::Real, x::Union{Real,Vector}, z::Union{Real,Vector}) = (y - b' * x)*z
    gn(b) = sum(g(b; y = Y[i], x = X[i, :], z = Z[i, :]) for i = 1:N) / N
    obj(b) = gn(b)' *  W * gn(b)
    gmm = optimize(obj, zeros(k))
    β = Optim.minimizer(gmm)
    println("Coefficients: ", β)
    # Variance estimate: Ω = g(β)g(β)'
    Ω = sum(g(β; y = Y[i], x = X[i, :], z = Z[i, :]) * g(β; y = Y[i], x = X[i, :], z = Z[i, :])' for i = 1:N) / N
    G = sum(-Z[i, :] * X[i, :]' for i = 1:N) / N
    invGWG = inv(G' * W * G)
    V = invGWG * G' * W * Ω * W * G * invGWG
    @info "Variance estimator: " V
    se = sqrt.(diag(V)/N)
    @info "SE: " se

    # Overidentification test
    J = N * gn(β)' * inv(Ω) * gn(β)
    pval_J = 1 - cdf(Chisq(r - k), J)
    Jstat = (J = J, pval = pval_J)
    return β, se, Ω, Jstat
end

function AndersonRubin(Y::Vector, X::Vector,
        Z::Union{Vector,Matrix}; grid::Vector, intercept = true)
    N = length(Y)
    if intercept
        r = size(Z, 2) + 1
    else
        r = size(Z, 2)
    end
    ci_index = Array{Bool, 1}(undef, length(grid))
    # Progress bar
    mingrid = minimum(grid)
    maxgrid = maximum(grid)
    println("Computing AR CI in [$mingrid, $maxgrid] ")
    p = Progress(length(grid))
    df = DataFrame()
    for col = 1:size(Z, 2)
        df[:, Symbol("Z", col)] = Z[:, col]
    end
    R = ones(r)
    intercept && (R[1] = 0)
    for (i, b) in enumerate(grid)
        df[:, Symbol("ε")] = Y - b * X
        aftest =  lm(term(:ε) ~ sum(term.(Symbol.(names(df, Not(:ε))))), df)
        aftest_const = lm(@formula(ε ~ 1), df)
        results = ftest(aftest.model, aftest_const.model)
        (results.pval[2] > 0.05) && (ci_index[i] = true)
        next!(p)
    end
    return grid, ci_index
end
