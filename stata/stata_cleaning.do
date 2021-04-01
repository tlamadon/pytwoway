* statatwoway
* Requires ereplace: install using
* ssc install ereplace, replace
* Requires nwcommands: install using
* net install nwcommands-ado, from(http://www.nwcommands.org) replace

capture program drop contiguous_ids
program define contiguous_ids
    syntax varlist
    * Make column of ids contiguous.
    * varlist: variables to make contiguous
    foreach id_var in `varlist' {
        sort `id_var' // Sort to ensure order maintained
        levelsof `id_var', local(unique_ids) // Get all unique values
        gen id_contig = .
        local i = 0
        foreach id in `unique_ids' {
            replace id_contig = `i' if `id_var' == `id'
            local i = `i' + 1
        }
        drop `id_var'
        rename id_contig `id_var'
    }
end

capture program drop conset
program define conset
    * Update data to include only the largest connected set of movers
    * Source: https://www.statalist.org/forums/forum/general-stata-discussion/general/1311808-finding-the-largest-connected-set
    preserve
        bysort wid: egen fid_max = max(fid) // Link workers to a central node
        nwset fid fid_max, edgelist undirected // Generate network
        nwgen largest_cc = lgc(network) // Compute largest connected set
        keep if largest_cc == 1
        keep _nodelab
        rename _nodelab fid
        destring fid, replace // _nodelab defaults to string
        tempfile largest_cc
        save `largest_cc'
    restore
    merge m:1 fid using `largest_cc'
    keep if _merge == 3 // Keep if in largest connected set
    drop _merge
end

capture program drop approx_cdfs
program define approx_cdfs
    * Generate cdfs of compensation for firms.
    * `1': number of percentiles (cdf resolution)
    local cdf_resolution = `1'
    local pctiles = ""
    capture drop quantile_* // Drop quantiles if they already exist
    forvalue i=1/`cdf_resolution' {
        local pctiles = "`pctiles' `=100/`cdf_resolution' * `i' - 0.1'"
    }
    di "percentiles: `pctiles'"
    _pctile comp, percentiles(`pctiles')
    sort fid
    forvalue i=1/`cdf_resolution' {
        gen lessthan = comp <= r(r`i')
        by fid: ereplace lessthan = total(lessthan)
        by fid: gen quantile_`i' = lessthan / _N
        drop lessthan
    }
end

capture program drop cluster_firms
program define cluster_firms
    * Cluster
    * Source: https://wlm.userweb.mwn.de/Stata/wstatclu.htm
    * `1': number of clusters
    * `2': number of percentiles (cdf resolution)
    approx_cdfs `2' // Generate cdfs of compensation for firms
    local n_clusters = `1'
    capture drop j // Drop j if it already exists
    preserve
        collapse (first) quantile_*, by(fid) // Keep 1 observation per firm
        cluster ward quantile_*
        cluster gen j_init = gr(`n_clusters')
        cluster k quantile_*, k(`n_clusters') name(j) start(group(j_init))
        keep fid j
        replace j = j - 1 // Starts at 1, we start at 0
        tempfile firm_clusters
        save `firm_clusters'
    restore
    drop quantile_* // Don't need quantiles anymore
    merge m:1 fid using `firm_clusters'
    keep if _merge == 3
    drop _merge
end

capture program drop clean_data
program define clean_data
    * Clean data
    * `1': number of clusters
    * `2': number of percentiles (cdf resolution)
    * Drop missing data
    drop if missing(wid, fid, year, comp)

    * Drop worker-year duplicates, keeping only the highest paying job
    gsort wid year -comp
    duplicates drop wid year, force

    /* * Collapse by employment spells
    sort wid year
    gen new_spell = (wid != wid[_n - 1]) | (fid != fid[_n - 1])
    gen spell_id = sum(new_spell) - 1
    collapse (first) wid fid year_start=year (last) year_end=year (mean) comp, by(spell_id)
    drop spell_id */

    * Compute largest connected set
    conset

    * Cluster data
    cluster_firms `1' `2'

    * Make wids and fids contiguous
    qui contiguous_ids wid fid

    * Order columns
    order wid fid year_start year_end comp j
end

/* 
Example code:
set more off
set maxvar 10000
cd "/Users/adamalexanderoppenheimer/Desktop/pytwoway/stata"
import delimited "twoway_sample_data.csv", clear

zigzag_mata, tolerance(1e-15) max_iters(1000)
zigzag
clean_data 6 10
*/

capture program drop zigzag_mata
program define zigzag_mata
    syntax [, tolerance(real 1e-7) max_iters(integer 1000)]
    * max_iters (optional): maximum number of iterations
    * tolerance (optional): stop when estimators get this close
    * Run zigzag AKM estimator in Mata
    capture drop alpha_hat psi_hat // Drop if exist
    mata: zigzag(`tolerance', `max_iters')
end

mata
mata clear
void zigzag(real scalar tolerance, real scalar max_iters) {
    Y = st_data(., "wid fid comp")
    unique_wids = uniqrows(Y[, 1]) // Automatically sorts
    unique_fids = uniqrows(Y[, 2]) // Automatically sorts

    // For new, first column gives sum of (comp - psi/alpha), second column gives number of observations (then can compute average); for old, first column gives average(comp - psi/alpha)
    alpha_old = J(rows(unique_wids), 1, 0)
    alpha_new = J(rows(unique_wids), 2, 0)
    psi_old = J(rows(unique_fids), 1, 0)
    psi_new = J(rows(unique_fids), 2, 0)

    // Set number of observations per worker/firm (only has to be done once)
    for(i=1; i<=rows(Y); i++) {
        alpha_new[Y[i, 1] + 1, 2] = alpha_new[Y[i, 1] + 1, 2] + 1 // Our wids start at 0, so need to add 1
        psi_new[Y[i, 2] + 1, 2] = psi_new[Y[i, 2] + 1, 2] + 1 // Our fids start at 0, so need to add 1
    }

    alpha_chg = tolerance + 1
    psi_chg = tolerance + 1
    l = 1
    while ((l <= max_iters) & ((alpha_chg > tolerance) | (psi_chg > tolerance))) {
        if (mod(l, 10) == 0) {
            printf("loop %g\n", l)
        }
        // Compute alpha = avg(comp - psi) by worker
        alpha_new[, 1] = J(rows(alpha_new), 1, 0) // Reset column
        for(i=1; i<=rows(Y); i++) {
            alpha_new[Y[i, 1] + 1, 1] = alpha_new[Y[i, 1] + 1, 1] + (Y[i, 3] - psi_new[Y[i, 2] + 1, 1]) // Our wids start at 0, so need to add 1
        }
        alpha_new[, 1] = alpha_new[, 1] :/ alpha_new[, 2] // Compute average

        // Compute psi = avg(comp - alpha) by firm
        psi_new[, 1] = J(rows(psi_new), 1, 0) // Reset column
        for(i=1; i<=rows(Y); i++) {
            psi_new[Y[i, 2] + 1, 1] = psi_new[Y[i, 2] + 1, 1] + (Y[i, 3] - alpha_new[Y[i, 1] + 1, 1]) // Our fids start at 0, so need to add 1
        }
        psi_new[, 1] = psi_new[, 1] :/ psi_new[, 2] // Compute average
        psi_new[1, 1] = 0 // Normalize one firm to 0

        // Compute max change in parameters
        alpha_chg = max(abs(alpha_new[, 1] - alpha_old))
        psi_chg = max(abs(psi_new[, 1] - psi_old))

        // Update values from previous iteration
        alpha_old = alpha_new[, 1]
        psi_old = psi_new[, 1]
        l++
    }
    // Fill in values for alpha_hat and psi_hat in the data
    alpha_psi_hat = J(rows(Y), 2, 0)
    for(i=1; i<=rows(Y); i++) {
        alpha_psi_hat[i, 1] = alpha_new[Y[i, 1] + 1, 1]
        alpha_psi_hat[i, 2] = psi_new[Y[i, 2] + 1, 1]
    }
    // Save into Stata
    idx = st_addvar("float", ("alpha_hat", "psi_hat"))
    st_store(., idx, alpha_psi_hat)
}
end

void zigzag_slow(real scalar tolerance, real scalar max_iters) {
    Y = st_data(., "wid fid comp")
    wid_indices = panelsetup(Y, 1) // Indices for each wid (already sorted by wid)
    fid_indices = panelsetup(sort(Y, 2), 2) // Indices for each fid (sort required for by to work)
    // J(r, c, v) creates matrix with rows, column, fill value
    alpha_psi_old = J(rows(Y), 5, 0)
    alpha_psi_new = J(rows(Y), 5, 0)
    alpha_psi_new[, (1::3)] = Y

    alpha_chg = tolerance + 1
    psi_chg = tolerance + 1
    i = 1
    while ((i <= max_iters) & ((alpha_chg > tolerance) | (psi_chg > tolerance))) {
        if (mod(i, 50) == 0) {
            printf("loop %g\n", i)
        }
        // Compute alpha = avg(comp - psi) by worker
        alpha_psi_new = sort(alpha_psi_new, 4) // Sort by wid
        for (i=1; i<=rows(wid_indices); i++) {
            // Source: https://www.stata.com/manuals/m-5panelsetup.pdf
            // Source: https://www.stata.com/statalist/archive/2009-02/msg01127.html
            groupbywid = panelsubmatrix(alpha_psi_new, i, wid_indices)
            alpha_psi_new[wid_indices[i, 1]::wid_indices[i, 2], 4] = J(rows(groupbywid), 1, mean(groupbywid[, 3] - groupbywid[, 5]))
        }

        // Compute psi = avg(comp - alpha) by firm
        alpha_psi_new = sort(alpha_psi_new, 5) // Sort by fid
        for (i=1; i<=rows(fid_indices); i++) {
            // Source: https://www.stata.com/manuals/m-5panelsetup.pdf
            // Source: https://www.stata.com/statalist/archive/2009-02/msg01127.html
            groupbyfid = panelsubmatrix(alpha_psi_new, i, fid_indices)
            alpha_psi_new[fid_indices[i, 1]::fid_indices[i, 2], 5] = J(rows(groupbyfid), 1, mean(groupbyfid[, 3] - groupbyfid[, 4]))
        }

        // Compute max change in parameters
        alpha_chg = max(abs(alpha_psi_new[, 4] - alpha_psi_old[, 4]))
        psi_chg = max(abs(alpha_psi_new[, 5] - alpha_psi_old[, 5]))

        // Update values from previous iteration
        alpha_psi_old = alpha_psi_new

        if (mod(i, 50) == 0) {
            // Compute objective
            objective = sum((alpha_psi_new[, 3] - alpha_psi_new[, 4] - alpha_psi_new[, 5]) :^ 2) // :^ for element-by-element power
            printf("objective %f\n", objective)
        }
        i++
    }
    // Generate Stata variables
    alpha_psi_new = sort(alpha_psi_new, 4) // Ensure parameters merge properly
    idx = st_addvar("float", ("alpha_hat", "psi_hat"))
    st_store(., idx, alpha_psi_new[, (4, 5)])
}
end

capture program drop zigzag
program define zigzag
    syntax [, tolerance(real 1e-7) max_iters(integer 10000)]
    * max_iters (optional): maximum number of iterations
    * tolerance (optional): stop when estimators get this close
    * Run zigzag AKM estimator
    qui tempvar alpha_old alpha_new alpha_diff
    qui gen `alpha_old' = 0
    qui gen `alpha_new' = 0
    qui gen `alpha_diff' = 0.
    qui tempvar psi_old psi_new psi_diff
    qui gen `psi_old' = 0
    qui gen `psi_new' = 0
    qui gen `psi_diff' = 0.

    qui local alpha_chg = `tolerance' + 1 // To force an iteration
    qui local psi_chg = `tolerance' + 1 // To force an iteration

    * Begin iterations
    qui local i = 1
    while (`i' <= `max_iters') & ((`alpha_chg' > `tolerance') | (`psi_chg' > `tolerance')) {
        if mod(`i', 50) == 0 {
            di "Loop `i'"
        }
        qui replace `alpha_old' = `alpha_new'
        qui replace `psi_old' = `psi_new'
        * Compute alpha = avg(comp - psi) by worker
        qui bysort wid: ereplace `alpha_new' = mean(comp - `psi_old')
        * Compute psi = avg(comp - alpha) by firm
        qui bysort fid: ereplace `psi_new' = mean(comp - `alpha_old') if fid != 0 // Keep 1 firm as baseline

        * Compute change in parameters between loops
        qui replace `alpha_diff' = abs(`alpha_new' - `alpha_old')
        qui replace `psi_diff' = abs(`psi_new' - `psi_old')

        qui su `alpha_diff'
        qui local alpha_chg = r(max)
        qui su `psi_diff'
        qui local psi_chg = r(max)
        qui local i = `i' + 1
        /* di "max alpha difference: `alpha_chg'"
        di "max psi difference: `psi_chg'" */

        qui gen objective = (comp - `alpha_new' - `psi_new') ^ 2
        qui su objective
        qui local objective = r(mean)
        di "objective: `objective'"
        qui drop objective
    }
    gen alpha_hat = `alpha_new'
    gen psi_hat = `psi_new'
end


V=panelsetup(id, 1)
st_view(x, ., (tokens(in)))
sx=J(rows(x),cols(x),.)
for (i=1; i<=rows(V); i++) {
                         panelsubview(X, x, i, V)
      				  sx[V[i,1]::V[i,2],.]=J(rows(X),1,colsum(X))
					}

idx = st_addvar("float", (tokens(out)))
st_store(. , idx , sx)
}
