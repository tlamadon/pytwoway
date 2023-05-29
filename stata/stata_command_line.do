* For running pytwoway in stata
* Requires libjson: install using
* ssc install libjson, replace
* Requires insheetjson: install using
* ssc install insheetjson, replace

capture program drop leedtwoway
program define leedtwoway, rclass
    syntax namelist, config(string) [env(string) os(string) ho he]
    * namelist: fe and/or cre
    * config: filepath for config file
    * env (optional): conda environment name
    * os (optional, required if using env): operating system ("mac", "windows", or "linux")
    * ho (optional): if True, store homoskedastic bias correction results
    * he (optional): if True, store heteroskedastic bias correction results
    * Save data
    qui save "leedtwoway_temp_data.dta", replace
    qui local command = "!"

    if "`env'" != "" {
        * Add conda path
        if "`os'" == "mac" {
            qui local command = "`command'source /opt/anaconda3/etc/profile.d/conda.sh"
        }
        else if "`os'" == "windows" {
            qui local command = "`command'source /opt/anaconda3/etc/profile.d/conda.sh"
        }
        else if "`os'" == "linux" {
            qui local command = "`command'source /opt/anaconda3/etc/profile.d/conda.sh"
        }
        else {
            di "input `os' for operating system is invalid"
        }
        * Activate environment
        qui local command = "`command'; conda activate `env'"
    }

    * Run estimator
    foreach name in `namelist' {
        if "`name'" == "fe" {
            di "fe"
            qui local command_fe = "`command'; pytw --my-config `config' --stata --fe"
            `command_fe'
            qui capture drop var_y_fe
            qui capture drop var_psi_fe
            qui capture drop cov_psi_alpha_fe
            qui capture drop var_eps_fe
            qui gen str240 var_y_fe = ""
            qui gen str240 var_psi_fe = ""
            qui gen str240 cov_psi_alpha_fe = ""
            qui gen str240 var_eps_fe = ""
            * insheetjson using "res_fe.json", showresponse
            insheetjson var_y_fe var_psi_fe cov_psi_alpha_fe var_eps_fe using "res_fe.json", col("var(y)" "var(psi)_fe" "cov(psi,_alpha)_fe" "var(eps)_fe")
            foreach var in var_y_fe var_psi_fe cov_psi_alpha_fe var_eps_fe {
                qui destring `var', replace
                qui scalar `var' = `var'[0]
                * The following line is required, otherwise scalar dropped
                qui return scalar `var' = `var'
                drop `var'
            }
            if "`ho'" == "ho" {
                qui capture drop var_psi_ho
                qui capture drop cov_psi_alpha_ho
                qui capture drop var_eps_ho
                qui gen str240 var_psi_ho = ""
                qui gen str240 cov_psi_alpha_ho = ""
                qui gen str240 var_eps_ho = ""
                insheetjson var_psi_ho cov_psi_alpha_ho var_eps_ho using "res_fe.json", col("var(psi)_ho" "cov(psi,_alpha)_ho" "var(eps)_ho")
                foreach var in var_psi_ho cov_psi_alpha_ho var_eps_ho {
                    qui destring `var', replace
                    qui scalar `var' = `var'[0]
                    * The following line is required, otherwise scalar dropped
                    qui return scalar `var' = `var'
                    drop `var'
                }
            }
            if "`he'" == "he" {
                qui capture drop var_psi_he
                qui capture drop cov_psi_alpha_he
                qui capture drop var_eps_he
                qui gen str240 var_psi_he = ""
                qui gen str240 cov_psi_alpha_he = ""
                qui gen str240 var_eps_he = ""
                insheetjson var_psi_he cov_psi_alpha_he var_eps_he using "res_fe.json", col("var(psi)_he" "cov(psi,_alpha)_he" "var(eps)_he")
                foreach var in var_psi_he cov_psi_alpha_he var_eps_he {
                    qui destring `var', replace
                    qui scalar `var' = `var'[0]
                    * The following line is required, otherwise scalar dropped
                    qui return scalar `var' = `var'
                    drop `var'
                }
            }
        }
        else if "`name'" == "cre" {
            di "cre"
            qui local command_cre = "`command'; pytw --my-config `config' --stata --cre"
            `command_cre'
            qui capture drop var_y_cre
            qui capture drop var_cre
            qui capture drop cov_cre
            qui gen str240 var_y_cre = ""
            qui gen str240 var_cre = ""
            qui gen str240 cov_cre = ""
            * insheetjson using "res_cre.json", showresponse
            insheetjson var_y_cre var_cre cov_cre using "res_cre.json", col("var_y" "tot_var" "tot_cov")
            foreach var in var_y_cre var_cre cov_cre {
                qui destring `var', replace
                qui scalar `var' = `var'[0]
                * The following line is required, otherwise scalar dropped
                qui return scalar `var' = `var'
                drop `var'
            }
        }
        else {
            di "input `name' for estimator is invalid"
        }
    }

    * Remove file
    erase "leedtwoway_temp_data.dta"
end

/*
Example code:
set more off
set maxvar 100000
cd "/Users/adamalexanderoppenheimer/Desktop/pytwoway/stata"
import delimited "twoway_sample_data.csv", clear

leedtwoway fe cre, config("config.txt") env("stata-env") os("mac") ho he
*/
