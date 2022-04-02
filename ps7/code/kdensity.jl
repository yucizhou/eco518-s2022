#= Kernal density estimation
Inputs data X, specifies kernal type and bandwidth h
Returns an estimated density function f
=#
function kdensity(X::Vector{Float64}; kernal = "normal", h::Real = NaN)
    N = length(X)
    if isnan(h)
        # use normal reference rule
        σ = sqrt(var(X))
        h = 1.059 * σ / N^(1/5)
    end
    f(x0) = 1/(N * h) * sum( _K((X[i] - x0) / h; type=kernal) for i = 1:N )
    return f
end
