using Optim, Distributions
using LinearAlgebra, FiniteDifferences

function myprobit(Y::Vector, X::Matrix; intercept = true)
    if intercept == true
        X = hcat(ones(length(Y)), X)
    end
    k = size(X, 2)
    N = length(Y)
    mle = optimize(b -> -loglike(b; Y = Y, X = X), zeros(k))
    loglikelihood = -Optim.minimum(mle)
    β = Optim.minimizer(mle)
    println("Log likelihood: ", loglikelihood)
    println("Coefficients: ", β)
    V = probitse(Y, X, β)
    se_hess = sqrt.(diag(V.hess) ./ N)
    se_score = sqrt.(diag(V.score) ./ N)
    se_robust = sqrt.(diag(V.robust) ./ N)
    se = (hess = se_hess, robust = se_robust, score = se_score)
    println("Standard error: ")
    @show se_hess se_score se_robust
    return β, se, V
end

function loglike(β; Y::Vector, X::Matrix)
    Φ(x) = cdf(Normal(), x)
    φ(x) = pdf(Normal(), x)
    loglikelihood = 0
    for i = 1:length(Y)
        loglikelihood += Y[i] * log(Φ(X[i, :]' * β)) +
            (1 - Y[i])*log(1 - Φ(X[i, :]' * β))
    end
    return loglikelihood
end

function myscore(β; y, x)
    Φ(z) = cdf(Normal(), z)
    φ(z) = pdf(Normal(), z)
    score = ( (y - Φ(x' * β)) / (Φ(x' * β)*(1 - Φ(x' * β)) ) ) * φ(x' * β)*x
    return score
end

function probitse(Y::Vector, X::Matrix, β::Vector)
    Φ(x) = cdf(Normal(), x)
    φ(x) = pdf(Normal(), x)
    N = length(Y)
    H = zeros(size(X, 2), size(X, 2))
    Ω = zeros(size(X, 2), size(X, 2))
    for i = 1:N
        s(b) = myscore(b; y = Y[i], x = X[i, :])
        Ω += s(β) * s(β)'
        H += jacobian(central_fdm(5, 1), s, β)[1]
    end
    H = 1/N * H
    Ω = 1/N * Ω
    @info H Ω
    V_hess = -inv(H)
    V_score = inv(Ω)
    V_robust = inv(H) * Ω * inv(H)
    @info "Variance-covariances: " V_hess V_score V_robust
    return (hess = V_hess, score = V_score, robust = V_robust)
end
