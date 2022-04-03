#=
Kernal regression
Inputs data Y, X, specifies bandwidth h, kernal
Returns a discretized grid of x and m(x)

=#
using Base.Threads

function kreg(Y::Vector, X::Vector; x0grid = nothing,
        kernal = "normal", h::Real)
    N = length(X)
    Y = convert(Vector{Float64}, Y)
    X = convert(Vector{Float64}, X)
    if isnothing(x0grid)
        x0grid = collect(minimum(X):bin:maximum(X))
    end
    mgrid = similar(x0grid)
    m = _kreg_estim(Y, X, kernal = kernal, h = h)
    mgrid = m.(x0grid)
    return x0grid, mgrid
end

#=
Inpust data Y, X, point of interest x0
Specifies bandwidth h and kernal
Returns a number that corresponds to the loc poly regression at point x0
i.e. m(x0) = Σ_{i = 1 to N} w_i(x0) Y_i
=#
function _kreg_estim(Y::Vector, X::Vector;
        kernal = "normal", h::Real)
    N = length(X)
    Y = convert(Vector{Float64}, Y)
    X = convert(Vector{Float64}, X)
    w = _w_kreg(X; kernal = kernal, h = h)
    m(x0) = sum(w[i](x0) * Y[i] for i = 1:N)
    return m
end

#=
Calculating weights
Inputs data X, specifies bandwidth h, kernal, point of interest x0
Returns an array of functions w_i(x0) where
w_i(x0) = K((X_i - x0)/h) Σ_{i = 1 to N}(K((X_j - x0)/h))
=#
function _w_kreg(X::Vector{Float64};
        kernal = "normal", h::Real)
    N = length(X)
    w = Array{Function, 1}(undef, N)
    Threads.@threads for i = 1:N
        w[i] = x0 -> _K( (X[i] - x0) / h; type=kernal) / sum( _K((X[j] - x0) / h; type=kernal) for j = 1:N )
    end
    return w
end

#=
Cross validation
Inputs given bandwidth h, returns CV(h) where
CV(h) = 1/N Σ_{i = 1 to N} [ (Y_i - m(X_i)) / (1 - w_i(X_i)) ]^2
=#
function CV_kreg(h;
        Y::Vector, X::Vector, kernal = "normal")
    N = length(X)
    Y = convert(Vector{Float64}, Y)
    X = convert(Vector{Float64}, X)

    m = _kreg_estim(Y, X; kernal = kernal, h = h).(X)
    wfun = _w_kreg(X, kernal = kernal, h = h)
    w = Array{Float64, 1}(undef, N)
    @threads for i = 1:N
        w[i] = wfun[i](X[i])
    end
    cv = 1/N * sum( ((Y[i] - m[i]) / (1 - w[i]) )^2 for i = 1:N)
    return cv
end
