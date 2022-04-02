clear all
set scheme Modern

program main
    load_kalman
    merge 1:1 t using ../output/forecast, assert(1 2 3) keep(2 3) ///
        nogen
    forecast_actual f_pi, lcolor(maroon) name("MA(1) forecast") ///
        saveas(ma_pi)
    forecast_actual filtered_s, lcolor(ebblue) name("Kalman filter") ///
        saveas(kalman_pi)
end

program forecast_actual
    syntax anything(name=f_pi), lcolor(str) name(str) saveas(str)
    sort t
    format t %tqccYY!Qq
    twoway (line `f_pi' pi t, ///
            lcolor(`lcolor' gs12) lpattern(solid dash)), ///
        legend(order(1 "`name'" 2 "Actual inflation")) ///
        ytitle("Inflation") xtitle(Quarters)
    graph export ../output/`saveas'.pdf, replace
end

program load_kalman
    tempfile filter smoother
    foreach f in smoother filter  {
        import delimited using ../output/`f'_s.csv, clear
        save ``f'', replace
    }
    merge 1:1 _n using `smoother', nogen
    gen t = tq(1959q1) + _n
end

* Execute
main
