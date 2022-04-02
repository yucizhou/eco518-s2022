clear all
set scheme Modern

program main        
	prep_data
	ar_coefs x, ar(4) estim_save(gdpc96)
	matrix phis = r(table)
	local s2 = r(s2)
	gen_forecast, last(2022q4) estim(gdpc96)
	sdensity, phis(phis) ar(4) s2(`s2') saveas(gdpc96_sdensity)
	esttab gdpc96 using ../output/gdpc96.tex, label noisily wide varlabels(L.x "$ x_{t-1}$" L2.x "$ x_{t-2}$" L3.x "$ x_{t-3}$" L4.x "$ x_{t-4}$" _cons "Constant") mtitles("$ x_t$") booktabs  substitute(_ _) nofix replace
end

program prep_data
	import delimited "../raw/Real_GDP.csv", clear
	gen date_mdy = date(date, "MDY")
	gen quarter = qofd(date_mdy)
	format quarter %tqccYY!Qq
	tsset quarter
	gen x = 400 * log(gdp/L.gdp)
end

program ar_coefs, rclass
	syntax anything(name=x), ar(int) [estim_save(str)]
	forvalues i = 1/`ar' {
		local lags `lags' L`i'.`x'
		local phis `phis' phi`i'
	}
	reg `x' `lags'
	if !mi("`estim_save'") estimates store `estim_save'
	tempname table
	matrix `table' = r(table)["b", .]
	matrix colnames `table' = `phis' cons
	return matrix table = `table'
	return scalar s2 = e(rmse)^2
	
end

program gen_forecast
	syntax, last(str) estim(str) 
	tsappend, last(`last') tsfmt(tq)
	forecast create `estim'
	forecast estimates `estim'
	qui {
		tsset 
		local timevar = `r(timevar)'
	}
	
	forecast solve
	twoway (line x quarter if !mi(x)) ///
		(line f_x quarter if quarter >= `=tq(2021q4)'), ///
		legend(off) xtitle(Quarter) ytitle(Real GDP Growth (Percentage Points, Annualized)) ///
		xline(`=tq(2021q4)')
	graph export ../output/`estim'_forecast_`last'.pdf, replace
end

program sdensity
	syntax, phis(str) ar(int) s2(real) saveas(str)
	local phisquared = 0
	forvalues i = 1/`ar' {
		local phi`i' = phis["b", "phi`i'"]
		local phisquared = `phisquared' + (`phi`i'')^2
		local lambdas `lambdas' - `phi`i'' * cos(`i' * x)
	}
	forvalues i = 1/`ar' {
		di "i = `i'"
		forvalues j = `=`i'+1'/`ar' {
			di "j = `j'"
			local lambdas `lambdas' + `=`phi`i''* `phi`j'' ' * cos(`=`j'-`i'' * x)
		}
	}
	
	di `" y = `s2' * (1 + `phisquared'  + 2 * (`lambdas'))^(-1)  / 2 /c(pi) , range(0 `=c(pi)')"'
	twoway (function y = `s2' * (1 + `phisquared'  + 2 * (`lambdas'))^(-1)  / 2 / c(pi) , range(0 `=c(pi)') ), ///
		xtitle("{&lambda}") ytitle("{it:f} ({&lambda})") ///
		xlabel(0 `=c(pi)/4' "{&pi}/4" `=c(pi)/2' "{&pi}/2" `=3*c(pi)/4' "3{&pi}/4" `=c(pi)' "{&pi}")
	graph export ../output/`saveas'.pdf, replace
end


* Execute
main
