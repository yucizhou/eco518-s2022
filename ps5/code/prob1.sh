stata-se -b params.do
julia kalman.jl > kalman.log
stata-se -b plot.do
