clear all
set scheme Modern

program main
    prep_data
	svar_run
	export_irf, impulse(I M) response(M) ///
		saveas(incomeresponse)
	reducedform_test
	lr_test
end

program prep_data
    import delimited ../raw/IncomeMoneyGrowth.csv, varnames(1) clear
    rename v1 date_str
    gen date = date(date_str, "MDY")
    gen t = mofd(date)
    format t %tmCCYY/NN
    format date %tdCCYY/NN/DD
	tsset t
	rename (incomegrowth m3growth) (I M)
end

program svar_run
	matrix rest = (., .) \ (0, .)
	svar I M, lags(1/12) acns(rest)
	matrix A = e(A)
end

program export_irf
	syntax, impulse(str) response(passthru) saveas(str)
	cap erase test.irf
	irf create test, set(test) step(48)
	
	foreach i of local impulse {
		irf graph irf, impulse(`i') `response' ///
		title("") note("") subtitle("") caption("") xtitle(Time) ///
		ustep(48) legend(off) individual xlabel(0(4)48) ///
		ytitle("Income growth response")
		graph export ../output/`saveas'`i'.pdf, replace
	}
end

program reducedform_test 
	var I M, lags(1/12) 
	di "Reduced form test on quarterly restrictions"
	
	test ([I]L1.M * `=A[1,1]' + [M]L1.M * `=A[1,2]' = 0) ///
	([I]L2.M * `=A[1,1]'  + [M]L2.I * `=A[1,2]' = 0)
end

program lr_test
	qui var I M, lags(1/12)
	mat B = inv(A)
	forval l = 1/12 {
		if `l' == 1 {
			foreach V in I M {
				local `V'_I ``V'_I' [`V']L`l'.I
				local `V'_M ``V'_M' [`V']L`l'.M
			}
		}
		else {
			foreach V in I M {
				local `V'_I ``V'_I' + [`V']L`l'.I
				local `V'_M ``V'_M' + [`V']L`l'.M
			}
		}
	} 
	test ((1 -`M_M') * `=B[1, 1]' + (1 -`I_M') * `=B[2, 1]' = 0) ///
		((1 -`M_M') * `=B[1, 2]' + (1 -`I_M') * `=B[2, 2]' = 0) 
end

* Execute
main
