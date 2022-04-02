clear all
cap mkdir ../output/

program main
    load_data
    params y
    local theta = r(theta)
    local sigma2 = r(sigma2)
    save_estimates, sigma2(`sigma2') theta(`theta') saveas(params)
    forecast y, theta(`theta') gen(f_y) until(2022q4)
    gen f_pi = .
    replace f_pi = pi in 2
    replace f_pi = L.pi + f_y if _n > 2
    replace f_pi = L.f_pi + f_y if mi(f_pi)
    keep t y f_y pi f_pi
    save ../output/forecast, replace
end

program load_data
    import delimited ../raw/JCXFE.csv, varnames(1) clear
    gen t = qofd(date(date, "MDY"))
    format t %tqccYY!Qq
    tsset t
    rename jcxfe p
    gen pi = 400 * log(p / L.p)
    gen y = pi - L.pi
    export delimited using ../output/jcxfe.csv, replace
end

program params, rclass
    syntax anything(name=y)
    quietly {
        sum `y'
        scalar g0 = r(Var)
        corr `y' L.`y', cov
        scalar g1 = r(cov_12)
    }
    di "Autocovariances"
    di "gamma(0): " g0
    di "gamma(1): " g1

    /*
    gamma(0) = (1 + \theta^2) * sigma^2
    gamma(1) = -theta * sigma^2
    Use these to obtain MA(1) params
    */
    scalar n = -g0 / g1
    scalar theta = (- sqrt(n^2 - 4) + n) / 2
    scalar sigma2 = -g1 / theta
    di _newline "MA(1) estimates"
    di "Theta: " theta
    di "sigma_e^2: " sigma2

    foreach s in g0 g1 theta sigma2 {
        return scalar `s' = `s'
    }
end

program forecast
    syntax anything(name=y), ///
        theta(real) ///
        gen(str) until(str)
    local f_y `gen'
    tempvar thetap
    gen `thetap' = .

    gen `f_y' = .
    local T = _N
    tsappend, last(`until') tsfmt(tq)
    forval t = 3/`T' {
        if mod(`t', 100) == 0 di "t = `t'"
        qui {
            mkmat y if _n <= `t', matrix(y) nomissing
            replace `thetap' = theta^(rowsof(y) + 1 - _n)
            mkmat `thetap', matrix(thetap)
            mkmat y if _n <= `t', matrix(y) nomissing
            matrix yhat = (thetap[1..`=rowsof(y)', 1]' * - y)
            scalar yhat = yhat[1, 1]
            replace `f_y' = yhat in `=`t'+1'
            if `t' == 10 {
                noisily {
                    mat list y
                    mat test = thetap[1..`=rowsof(y)', 1]
                    mat list test
                    di "Theta^t:"
                    di theta^`=rowsof(y)'
                }
            }
        }
    }
    replace `f_y' = - `theta' * L.`f_y' if mi(`f_y')
    label var `f_y' "Forecast value of `y'"
end

program save_estimates
    syntax, sigma2(real) theta(real) saveas(str)
    preserve
    clear
    set obs 1
    gen sigma2 = `sigma2'
    gen theta = `theta'
    export delimited using ../output/`saveas'.csv, replace
end

* Execute
main
