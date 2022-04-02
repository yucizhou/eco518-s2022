#=
Kernal regression
Inputs data Y, X, specifies bandwidth h, kernal
Returns a function m(x) = Σ_{i = 1 to N} w_i(x0) Y_i
=#
function kreg(Y::Vector{Float64}, X::Vector{Float64};
        kernal = "normal", h::Real)
    N = length(X)
    w = _w_kreg(X; kernal = kernal, h = h)
    w = Array{Function, 1}(undef, N)
    for i = 1:N
        w[i] = x0 -> _K( (X[i] - x0) / h; type=kernal) / sum( _K((X[j] - x0) / h; type=kernal) for j = 1:N )
    end
    m(x0) = sum(w[i](x0) * Y[i] for i = 1:N)
    return m
end

#=
Calculating weights
Inputs data X, specifies bandwidth h, kernal
Returns an array of functions w_i(x0) where
w_i(x0) = K((X_i - x0)/h) Σ_{i = 1 to N}(K((X_j - x0)/h))
=#
function _w_kreg(X::Vector{Float64};
        kernal = "normal", h::Real)
    N = length(X)
    w = Array{Function, 1}(undef, N)
    for i = 1:N
        w[i] = x0 -> _K( (X[i] - x0) / h; type=kernal) / sum( _K((X[j] - x0) / h; type=kernal) for j = 1:N )
    end
    return w
end

#=
Cross validation
Inputs given bandwidth h, returns CV(h) where
CV(h) = 1/N Σ_{i = 1 to N} [ (Y_i - m(X_i)) / (1 - w_i(X_i)) ]^2
=#
function CV_kreg(h,;
        Y::Vector{Float64}, X::Vector{Float64}, kernal = "normal")
    N = length(X)
    m = kreg(Y, X, kernal = kernal, h = h).(X)
    wfun = _w_kreg(X; kernal = kernal, h = h)
    w = similar(X)
    for i = 1:N
        w[i] = wfun[i](X[i])
    end
    cv = 1/N * sum( ((Y[i] - m[i]) / (1 - w[i]) )^2 for i = 1:N)
    return cv
end
