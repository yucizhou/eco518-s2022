clear all

program main
    import delimited ../raw/mroz.csv, clear
    probit part kidslt6 age educ nwifeinc, robust
    margins, dydx(educ) atmeans
    margins, dydx(educ)
end

* Execute
main
