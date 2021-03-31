* For running pytwoway in stata
* Requires libjson: install using
* ssc install libjson, replace
* Requires insheetjson: install using
* ssc install insheetjson, replace

capture program drop statatwoway
program define statatwoway, rclass
    syntax namelist, config(string) [env(string)]
    * namelist: fe and/or cre
    * config: filepath for config file
    * env (optional): conda environment name
    * Save data
    qui save "statatwoway.dta", replace
    qui local command = "!"

    if "`env'" != "" {
        qui local command = "`command'source ~/opt/anaconda3/etc/profile.d/conda.sh" // Add conda path
        qui local command = "`command'; conda activate `env'" // Activate environment
    }

    * Run estimator
    foreach name in `namelist' {
        if "`name'" == "fe" {
            di "fe"
            qui local command_fe = "`command'; pytw --my-config `config' --fe"
            `command_fe'
            qui gen str240 var_fe = ""
            qui gen str240 cov_fe = ""
            qui gen str240 var_ho = ""
            qui gen str240 cov_ho = ""
            /* insheetjson using "res_fe.json", showresponse */
            insheetjson var_fe cov_fe var_ho cov_ho using "res_fe.json", col("var_fe" "cov_fe" "var_ho" "cov_ho")
            foreach var in var_fe cov_fe var_ho cov_ho {
                qui destring `var', replace
                qui scalar `var' = `var'[0]
                qui return scalar `var' = `var' // This line required, otherwise scalar dropped
                drop `var'
            }
        }
        else if "`name'" == "cre" {
            di "cre"
            qui local command_cre = "`command'; pytw --my-config `config' --cre"
            `command_cre'
            qui gen str240 var_cre = ""
            qui gen str240 cov_cre = ""
            /* insheetjson using "res_cre.json", showresponse */
            insheetjson var_cre cov_cre using "res_cre.json", col("tot_var" "tot_cov")
            foreach var in var_cre cov_cre {
                qui destring `var', replace
                qui scalar `var' = `var'[0]
                qui return scalar `var' = `var' // This line required, otherwise scalar dropped
                drop `var'
            }
        }
        else {
            di "input `name' for estimator is invalid"
        }
    }

    erase "statatwoway.dta" // Remove file
end

/*
Example code:
set more off
set maxvar 100000
cd "/Users/adamalexanderoppenheimer/Desktop/pytwoway/stata"
import delimited "twoway_sample_data.csv", clear

statatwoway fe cre, config("config.txt") env("stata-env")
*/
