#=
Polynomial series regression
Inputs data Y, X, specifies order of polynomial (default linear)
Returns a discretized grid of x and m(x)
=#
function polyseries(Y::Vector{Float64}, X::Vector{Float64};
        bin = 0.01, order::Int = 1, x0grid = nothing)
    N = length(X)
    if isnothing(x0grid)
        x0grid = collect(minimum(X):bin:maximum(X))
    end
    mgrid = similar(x0grid)
    for (k, x0) in enumerate(x0grid)
        mgrid[k] = _polyseries_estim(Y, X, x0 = x0, p = order)
    end
    return x0grid, mgrid
end

#=
Inpust data Y, X, point of interest x0
Specifies order of polynomial p
Returns a number that corresponds to the poly series regression at point x0
i.e. m(x0)
=#
function _polyseries_estim(Y::Vector{Float64}, X::Vector{Float64};
        x0::Real, p::Int = 1)
    N = length(X)
    w = _w_polyseries(X, x0, p = p)
    m = sum(w[i] * Y[i] for i = 1:N)
    return m
end

#=
Inputs X, specifies bandwidth h and kernal
Returns the weight when viewing locpoly regression as a linear smoother
w_i(x) = z(x)' { Σ_{j = 1 to N} [ K((X_j - X) / h) Z_j Z_j'] }^{-1} K((X_i - x) / h) Z_i
m(x) = Σ_{i = 1 to N} w_i(x) Y_i

z(x) = [1, x, ..., x^p]
Z_i = [1, X_i, ..., X_i^p]
=#
function _w_polyseries(X::Vector{Float64}, x0::Real; p::Int = 1)
    N = length(X)
    w = Array{Float64, 1}(undef, N)
    Z = Array{Vector, 1}(undef, N)
    for j = 1:N
        Z[j] = X[j] .^ collect(0:p)
    end
    # inv_ZZj = { Σ_{j = 1 to N} [ Z_j Z_j'] }^{-1}
    inv_ZZj = (sum(Z[j] * Z[j]' for j = 1:N))^(-1)
    z = x0 .^ collect(0:p)
    for i = 1:N
        w[i] = z' * inv_ZZj * Z[i]
    end
    return w
end

#=
Cross validation
Inputs given bandwidth h, returns CV(h) where
CV(h) = 1/N Σ_{i = 1 to N} [ (Y_i - m(X_i)) / (1 - w_i(X_i)) ]^2
=#
function CV_polyseries(p::Int;
        Y::Vector{Float64}, X::Vector{Float64})
    N = length(X)
    m = similar(X)
    w = Array{Float64, 1}(undef, N)
    for i = 1:N
        w[i] = _w_polyseries(X, X[i], p = p)[i]
        m[i] = _polyseries_estim(Y, X, x0 = X[i], p = p)
    end
    cv = 1/N * sum( ( (Y[i] - m[i]) / (1 - w[i]) )^2 for i = 1:N)
    return cv
end
