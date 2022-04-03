#=
Local polynomial regression
Inputs data Y, X, specifies bandwidth h, kernal, order of polynomial (default linear)
Returns a discretized grid of x and m(x)
=#
function locpoly(Y::Vector{Float64}, X::Vector{Float64};
        kernal = "normal", h::Real, x0grid = nothing,
        bin = 0.01, order::Int = 1)
    N = length(X)
    if isnothing(x0grid)
        x0grid = collect(minimum(X):bin:maximum(X))
    end
    mgrid = similar(x0grid)
    for (k, x0) in enumerate(x0grid)
        mgrid[k] = _locpoly_estim(Y, X, x0 = x0, kernal = kernal, h = h, p = order)
    end
    return x0grid, mgrid
end

#=
Inpust data Y, X, point of interest x0
Specifies bandwidth h, kernal, order of polynomial p
Returns a number that corresponds to the loc poly regression at point x0
i.e. m(x0)
=#
function _locpoly_estim(Y::Vector{Float64}, X::Vector{Float64};
        x0::Real,
        kernal = "normal", h::Real, p = 1)
    N = length(X)
    w = _w_locpoly(X, x0, kernal = kernal, h = h, p = p)
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
function _w_locpoly(X::Vector{Float64}, x0::Real;
        kernal = "normal", h::Real, p::Int = 1)
    N = length(X)
    w = Array{Float64, 1}(undef, N)
    Z = Array{Vector, 1}(undef, N)
    for j = 1:N
        Z[j] = X[j] .^ collect(0:p)
    end
    # inv_KZZj = { Σ_{j = 1 to N} [ K((X_j - X) / h) Z_j Z_j'] }^{-1}
    inv_KZZj = (sum( _K((X[j] - x0) / h) * Z[j] * Z[j]' for j = 1:N))^(-1)
    z = x0 .^ collect(0:p)
    for i = 1:N
        w[i] = z' * inv_KZZj * _K((X[i] - x0) / h) * Z[i]
    end
    return w
end

#=
Cross validation
Inputs given bandwidth h, returns CV(h) where
CV(h) = 1/N Σ_{i = 1 to N} [ (Y_i - m(X_i)) / (1 - w_i(X_i)) ]^2
=#
function CV_locpoly(h;
        Y::Vector{Float64}, X::Vector{Float64}, kernal = "normal",
        p::Int = 1)
    N = length(X)
    m = similar(X)
    w = Array{Float64, 1}(undef, N)
    for i = 1:N
        w[i] = _w_locpoly(X, X[i], kernal = kernal, h = h, p = p)[i]
        m[i] = _locpoly_estim(Y, X, x0 = X[i], kernal = kernal, h = h, p = p)
    end
    cv = 1/N * sum( ( (Y[i] - m[i]) / (1 - w[i]) )^2 for i = 1:N)
    return cv
end
