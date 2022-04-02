using CSV, Tables, LinearAlgebra

function filter(;sigma2, theta, P0, s0, y)
    Q = (1 - 2 * theta + theta^2) * sigma2
    R = theta * sigma2
    H = 1
    T = length(y)
    println("T = $T" )
    println("Q = $Q")
    println("R = $R")

    P = similar(y, T+1, T+1)
    s = similar(y, T+1)
    s[1] = s0
    P[1, 1] = P0

    for t = 2:T
        P[t, t-1] = P[t-1, t-1] + Q
        s[t] = s[t-1] + (P[t, t-1] / (P[t, t-1] + R)) * (y[t] - s[t - 1])
        P[t, t] = P[t, t-1] - P[t, t-1]^2 / (P[t, t-1] + R)
        if t % 10 == 0
            println("t = $t")
            println("P = ", P[t, t])
            println("s = ", s[t])
        end
    end
    return s, P
end

function smoother(;f_P, f_s, y)
    T = length(y)
    s = similar(f_s)
    P = similar(f_s)
    s[T] = f_s[T]
    P[T] = f_P[T, T]
    J = similar(P, T)
    for t = T-1:-1:1
        J[t] = f_P[t, t] / f_P[t+1, t]
        s[t] = f_s[t] + J[t] * (s[t+1] - f_s[t+1])
        P[t] = f_P[t, t] + J[t]^2 * (P[t+1] - f_P[t+1, t])
        if t % 10 == 0
            println("t = $t")
            println("P = ", P[t])
            println("s = ", s[t])
        end
    end
    return s, P
end

Params = CSV.File("../output/params.csv")
Pi = CSV.File("../output/jcxfe.csv").pi
sigma2 = first(Params.sigma2)
theta = first(Params.theta)
f_s, f_P = filter(sigma2 = sigma2,
    theta = theta,
    P0 = 10000.0,
    s0 = 0.0,
    y = Pi)
CSV.write("../output/filter_s.csv",  Tables.table(f_s), header=["filtered_s"])

s_s, s_P = smoother(
    f_P = f_P,
    f_s = f_s,
    y = Pi)
println("$s_s")
CSV.write("../output/smoother_s.csv",  Tables.table(s_s), header=["smoothed_s"])
