clear all

program main
	prep_data
	parametric_lrv, lags(12)
	pop_mean_ci, lrv(`=r(w2)')
	newey_west
	pop_mean_ci, lrv(`=r(w2)')
	prew_ar1
	pop_mean_ci, lrv(`=r(w2)')
	pergram_lwr, first(10)
	pop_mean_ci, lrv(`=r(w2)')
	pergram_lwr, first(3)
	pop_mean_ci_t, lrv(`=r(w2)') n(3)
end

program prep_data
	import delimited ../raw/IRateSpread.csv, clear
	gen date_num = date(date, "MDY")
	format date_num %tdCCYY-NN-DD
	gen t = mofd(date_num)
	format t %tmCCYY-NN
	tsset t
	rename t10y3m y
	sum y
	local mean = r(mean)
	gen y_dm = y - `mean'
end

program pop_mean_ci, rclass
	syntax, lrv(real)
	qui sum y, meanonly
	local mean = r(mean)
    local T = _N
	scalar cil = `mean' - sqrt(`lrv'/`T') * invnormal(0.975)
	scalar ciu = `mean' + sqrt(`lrv'/`T') * invnormal(0.975)
	di "Population mean CI (normal):"
	di "[" cil ", " ciu "]"
	return scalar cil = cil
	return scalar ciu = ciu
end

program pop_mean_ci_t, rclass
	syntax, lrv(real) n(int)
	qui sum y, meanonly
	local mean = r(mean)
    local T = _N
	scalar cil = `mean' - sqrt(`lrv'/`T') * invt(`=2*`n'', 0.975)
	scalar ciu = `mean' + sqrt(`lrv'/`T') * invt(`=2*`n'', 0.975)
	di "Population mean CI (t-dist):"
	di "[" cil ", " ciu "]"
	return scalar cil = cil
	return scalar ciu = ciu
end


program parametric_lrv, rclass
	syntax, lags(int)
	reg y L(1/`lags').y
	scalar sigma2 = e(rmse)^2
	mat b = r(table)["b", 1..`lags']
	matrix phis = b * J(`lags', 1, 1)
	scalar denom = (1 - phis[1, 1])^2
	scalar w2 = sigma2 / denom
	di "Long run variance (parametric) is"
	di w2
	return scalar w2 = w2
end

program newey_west
	nw_wt y
	scalar w2 = r(w2)
	di "Long run variance (NW) is"
	di w2
end

program prew_ar1, rclass
	reg y L.y
	scalar rho = _b[L.y]
	predict u, resid
	nw_wt u
	di r(w2)
	scalar w2 = r(w2) / (1 - rho)^2
	di "Long run variance (pre-whitening) is"
	di w2
	return scalar w2 = w2
end

program nw_wt, rclass
	syntax varname
	local u `varlist'
	local T = _N
	scalar bT = 0.75 * `T'^(1/3)
	scalar w2 = 0
	forvalues k = 0/`=`T'-1' {
		cap corr `u' L`k'.`u', cov
		scalar gamma_`k' = r(cov_12)
		if gamma_`k' == . scalar gamma_`k' = 0
		if `k' == 0 scalar w2 = w2 + gamma_`k' * max((1 - `k'/bT), 0)
		else scalar w2 = w2 + 2 * gamma_`k' * max((1 - `k'/bT), 0)
	}
	return scalar w2 = w2
end

program pergram_lwr, rclass
	syntax, first(int)
	local T = _N
	tempvar t
	gen `t' = _n
	foreach s in cos sin {
		matrix Z_`s' = J(1, `T', .)
	}
	scalar w2 = 0
	forval l = 1/`first' {
		gen Z_cos = cos(2 * c(pi) * `l' * (`t' - 1) / `T') * y
		gen Z_sin = sin(2 * c(pi) * `l' * (`t' - 1) / `T') * y
		foreach s in cos sin {
			qui sum Z_`s'
			local sum = r(sum)
			matrix Z_`s'[1, `l'] = sqrt(2/`T') * `sum'
			drop Z_`s'
		}
		scalar w2 = w2 + (Z_cos[1, `l']^2 + Z_sin[1, `l']^2) / 2
	}
	di "Long run variance (first `first' coord of periodogram) is"
	scalar w2 = w2 / `first'
	di w2
	return scalar w2 = w2
end


* Execute
main
