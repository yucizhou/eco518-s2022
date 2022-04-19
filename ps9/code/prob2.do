clear all

program main
    import delimited ../raw/heating.csv, clear
    gen id = _n
    reshape long @ic @oc, i(id) j(alt) string
    recode_alt
    cmset id alt
    cmclogit chosen ic oc, robust
end

program recode_alt
    replace alt = strtrim(alt)
    gen alt_cd = .
    replace alt_cd = 0 if alt == "heatpump",
    replace alt_cd = 1 if alt == "gascentral",
    replace alt_cd = 2 if alt ==  "electriccentral"
    replace alt_cd = 3 if alt == "gasroom"
    replace alt_cd = 4 if alt == "electricroom"
    gen chosen = alt_cd == choice
end

* Execute
main
