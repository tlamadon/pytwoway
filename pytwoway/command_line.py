'''
Script to run pytwoway through the command line

Usage example:
pytw --my-config config.txt --fe --cre
'''
import configargparse
import ast
from numpy.random import default_rng
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw

def clear_dict(d):
    '''
    Purpose:
        Remove dictionary entries where value is None.

    Inputs:
        d (dict): dictionary

    Returns:
        new_d (dict): cleared dictionary
    '''
    new_d = {}
    for key, val in d.items():
        if val is not None:
            new_d[key] = val
    return new_d

def str2bool(v):
    '''
    Allows bools to be input as strings.
    Source: https://stackoverflow.com/a/43357954
    Note: replace all "action='store_true'" with 'type=str2bool'
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

def main():
    p = configargparse.ArgParser()
    p.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

    # Options to select filetype
    p.add('--filetype', required=False, help="file format of data; default is 'csv'")

    # Options to run FE or CRE
    p.add('--fe', action='store_true', help='run FE estimation') # This option can be set in a config file because it starts with '--'
    p.add('--cre', action='store_true', help='run CRE estimation')

    ##### TwoWay start #####
    p.add('--data', required=False, help='path to labor data file')
    p.add('--collapse', type=str2bool, required=False, help='if True, run estimators on data collapsed at the worker-firm spell level', default=True)
    p.add('--seed', required=False, help='seed for rng')
    ##### TwoWay end #####

    ##### Stata start #####
    p.add('--stata', action='store_true', required=False, help='if True, running estimators on Stata')
    ##### Stata end #####

    ##### FE start #####
    p.add('--ho', type=str2bool, required=False, help='if True, compute the homoskedastic correction when estimating fe')
    p.add('--he', type=str2bool, required=False, help='if True, compute the heteroskedastic correction when estimating fe')
    p.add('--ndraw_trace_sigma_2', required=False, help='number of draws to use in trace approximation for sigma^2 when estimating fe')
    p.add('--ndraw_trace_ho', required=False, help='number of draws to use in trace approximation for homoskedastic correction when estimating fe')
    p.add('--ndraw_trace_he', required=False, help='number of draws to use in trace approximation for heteroskedastic correction when estimating fe')
    p.add('--ndraw_lev_he', required=False, help='number of draw to use in leverage approximation for heteroskedastic correction when estimating fe')
    p.add('--outputfile_fe', required=False, help='outputfile where fe results are saved')
    p.add('--ncore_fe', required=False, help='number of cores to use when estimating fe')
    p.add('--solver', required=False, help="solver to use when estimating fe; options are 'bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'minres', 'qmr', and 'amg'. 'minres' is recommended for small datasets and 'amg' is recommended for large datasets (100 million observations+).")
    p.add('--preconditioner', required=False, help="preconditioner to use when estimating fe; options are None, 'jacobi', 'vcycle', 'ichol', and 'ilu'. 'ichol' is recommended for small datasets and 'jacobi' is recommended if 'ichol' raises an error.")
    ##### FE end #####

    ##### Cluster start #####
    #### General start ####
    p.add('--measures', required=False, help="how to compute measures for clustering. Options are 'cdfs' for cdfs and 'moments' for moments. Can use a list for multiple measures. Details on options can be seen in bipartitepandas.measures.")
    p.add('--grouping', required=False, help="how to group firms based on measures. Options are 'kmeans' for kmeans and 'quantiles' for quantiles. Details on options can be seen in bipartitepandas.grouping.")
    p.add('--stayers_movers', required=False, help="if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers")
    p.add('--t', required=False, help='if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)')
    p.add('--weighted', type=str2bool, required=False, help='if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)')
    p.add('--dropna', type=str2bool, required=False, help="if True, drop observations where firms aren't clustered; if False, keep all observations")
    #### General end ####
    #### Measures start ####
    ### CDFs start ###
    p.add('--cdf_resolution', required=False, help='how many values to use to approximate the cdfs when clustering')
    p.add('--measure_cdfs', required=False, help='''
    how to compute the cdfs when clustering ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary
    ''')
    ### CDFs end ###
    ### Moments start ###
    p.add('--measures_moments', required=False, help='''
    how to compute the measures when clustering ('mean' to compute average income within each firm; 'var' to compute variance of income within each firm; 'max' to compute max income within each firm; 'min' to compute min income within each firm)
    ''')
    ### Moments end ###
    #### Measures end ####
    #### Grouping start ####
    ### KMeans start ###
    p.add('--n_clusters', required=False, help='the number of clusters to form as well as the number of centroids to generate for the k-means algorithm.')
    p.add('--init', required=False, help='''
    Method for initialization of the k-means algorithm:

    'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.

    'random': choose n_clusters observations (rows) at random from data for the initial centroids.

    If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

    If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.
    ''')
    p.add('--n_init', required=False, help='number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.')
    p.add('--max_iter', required=False, help='maximum number of iterations of the k-means algorithm for a single run.')
    p.add('--tol', required=False, help='relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence for the k-means algorithm.')
    p.add('--precompute_distances', required=False, help='precompute distances for the k-means algorithm (faster but takes more memory).')
    p.add('--verbose', required=False, help='verbosity mode for the k-means algorithm.')
    p.add('--random_state', required=False, help='determines random number generation for centroid initialization for the k-means algorithm. Use an int to make the randomness deterministic.')
    p.add('--copy_x', required=False, help='when pre-computing distances it is more numerically accurate to center the data first for the k-means algorithm. If copy_x is True (default), then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x is False.')
    p.add('--n_jobs', required=False, help='the number of OpenMP threads to use for the k-means computation. Parallelism is sample-wise on the main cython loop which assigns each sample to its closest center.')
    p.add('--algorithm', required=False, help='''
    k-means algorithm to use. The classical EM-style algorithm is "full". The "elkan" variation is more efficient on data with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).

    For now "auto" (kept for backward compatibiliy) chooses "elkan" but it might change in the future for a better heuristic.
    ''')
    ### KMeans end ###
    ### Quantiles start ###
    p.add('--n_quantiles', required=False, help='number of quantiles to compute for groups when clustering')
    ### Quantiles end ###
    #### Grouping end ####
    ##### Cluster end #####

    ##### CRE start #####
    p.add('--ncore_cre', required=False, help='number of cores to use when estimating cre')
    p.add('--ndraw_trace_cre', required=False, help='number of draws to use in approximation for traces when estimating cre')
    p.add('--ndp', required=False, help=' number of draw to use in approximation for leverages when estimating cre')
    p.add('--outputfile_cre', required=False, help='outputfile where cre results are saved')
    p.add('--posterior', type=str2bool, required=False, help='whether to compute the posterior variance when estimating cre')
    p.add('--wo_btw', required=False, help='sets between variation to 0, pure RE when estimating cre')
    ##### CRE end #####

    ##### Clean start #####
    p.add('--connectedness', required=False, help="for data cleaning, when computing largest connected set of firms: if 'connected', keep observations in the largest connected set of firms; if 'strongly_connected', keep observations in the largest strongly connected set of firms; if 'leave_out_observation', keep observations in the largest leave-one-observation-out connected set; if 'leave_out_spell', keep observations in the largest leave-one-spell-out connected set; if 'leave_out_match', keep observations in the largest leave-one-match-out connected set; if 'leave_out_worker', keep observations in the largest leave-one-worker-out connected set; if 'leave_out_firm', keep observations in the largest leave-one-firm-out connected set; if None, keep all observations.")
    p.add('--i_t_how', required=False, help="for data cleaning, when dropping i-t duplicates: if 'max', keep max paying job; otherwise, take `i_t_how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `i_t_how` can take any input valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in collapsed long and event study formats), then data is converted to long, cleaned, then converted back to its original format.")
    p.add('--drop_returns', required=False, help="for data cleaning, if 'returns', drop observations where workers leave a firm then return to it; if 'returners', drop workers who ever leave then return to a firm; if 'keep_first_returns', keep first spell where a worker leaves a firm then returns to it; if 'keep_last_returns', keep last spell where a worker leaves a firm then returns to it; if False, keep all observations.")
    ##### Clean end #####

    params = p.parse_args()

    ##### Stata start #####
    if params.stata:
        params.data = 'leedtwoway_temp_data.dta'
        params.filetype = 'dta'
    ##### Stata end #####

    ##### FE start #####
    fe_params = {'ho': params.ho, 'he': params.he, 'ndraw_trace_sigma_2': params.ndraw_trace_sigma_2, 'ndraw_trace_ho': params.ndraw_trace_ho, 'ndraw_trace_he': params.ndraw_trace_he, 'ndraw_lev_he': params.ndraw_lev_he, 'outputfile': params.outputfile_fe, 'ncore': params.ncore_fe, 'solver': params.solver, 'preconditioner': params.preconditioner}
    if params.stata:
        fe_params['outputfile'] = 'res_fe.json'
    fe_params = tw.fe_params(clear_dict(fe_params))
    ##### FE end #####

    ##### Cluster start #####
    #### Measures start ####
    ### CDFs start ###
    cdf_params = {'cdf_resolution': params.cdf_resolution, 'measure': params.measure_cdfs}
    cdf_params = clear_dict(cdf_params)
    ### CDFs end ###
    ### Moments start ###
    if params.measures_moments is not None:
        # Have to do ast.literal_eval twice for it to work properly
        measures_moments = ast.literal_eval(ast.literal_eval(params.measures_moments))
    else:
        measures_moments = params.measures_moments
    moments_params = {'measures_moments': measures_moments}
    moments_params = clear_dict(moments_params)
    ### Moments end ###
    #### Measures end ####
    if params.grouping is not None:
        #### Grouping start ####
        if params.grouping.lower() == 'kmeans':
            ### KMeans start ###
            KMeans_params = {'n_clusters': params.n_clusters, 'init': params.init, 'n_init': params.n_init, 'max_iter': params.max_iter, 'tol': params.tol, 'precompute_distances': params.precompute_distances, 'verbose': params.verbose, 'random_state': params.random_state, 'copy_x': params.copy_x, 'n_jobs': params.n_jobs, 'algorithm': params.algorithm}
            KMeans_params = clear_dict(KMeans_params)
            params.grouping = bpd.grouping.KMeans(**KMeans_params)
            ### KMeans end ###
        elif params.grouping.lower() == 'quantiles':
            ### Quantiles start ###
            quantiles_params = {'n_quantiles': params.n_quantiles}
            quantiles_params = clear_dict(quantiles_params)
            params.grouping = bpd.grouping.Quantiles(**quantiles_params)
            ### Quantiles end ###
        #### Grouping end ####
    #### General start ####
    if params.measures is not None:
        # Have to do ast.literal_eval twice for it to work properly
        measures_raw = ast.literal_eval(ast.literal_eval(params.measures))
        measures_raw = bpd.to_list(measures_raw)
        measures = []
        for measure in measures_raw:
            if measure.lower() == 'cdfs':
                measure_fn = bpd.measures.CDFs(**cdf_params)
            elif measure.lower() == 'moments':
                measure_fn = bpd.measures.Moments(**moments_params)
            measures.append(measure_fn)
    else:
        measures = params.measures
    cluster_params = {'measures': measures, 'grouping': params.grouping, 'stayers_movers': params.stayers_movers, 't': params.t, 'weighted': params.weighted, 'dropna': params.dropna}
    cluster_params = clear_dict(cluster_params)
    #### General end ####
    ##### Cluster end #####

    ##### CRE start #####
    cre_params = {'ncore': params.ncore_cre, 'ndraw_trace': params.ndraw_trace_cre, 'ndp': params.ndp, 'outputfile': params.outputfile_cre, 'posterior': params.posterior, 'wo_btw': params.wo_btw}
    if params.stata:
        cre_params['outputfile'] = 'res_cre.json'
    cre_params = tw.cre_params(clear_dict(cre_params))
    ##### CRE end #####

    ##### Prepare data start #####
    # Generate data
    pd_from_filetype = {
        'csv': pd.read_csv,
        'json': pd.read_json,
        'ftr': pd.read_feather,
        'feather': pd.read_feather,
        'dta': pd.read_stata,
        'stata': pd.read_stata,
        'parquet': pd.read_parquet,
        'excel': pd.read_excel,
        'xlsx': pd.read_excel,
        'sql': pd.read_sql
    }
    if params.filetype is None:
        params.filetype = 'csv'
    df = pd_from_filetype[params.filetype.lower()](params.data)

    # Clean data
    clean_params = {'connectedness': params.connectedness, 'collapse_at_connectedness_measure': True, 'i_t_how': params.i_t_how, 'drop_returns': params.drop_returns}
    clean_params = bpd.clean_params(clear_dict(clean_params))

    bdf = bpd.BipartiteDataFrame(df).clean(clean_params)
    if (params.collapse is not None) and params.collapse and (not type(bdf) == bpd.BipartiteLongCollapsed):
        bdf = bdf.collapse(is_sorted=True, copy=False)
    ##### Prepare data end #####

    # Run estimation
    rng = default_rng(params.seed)
    if params.fe:
        fe_estimator = tw.FEEstimator(bdf, fe_params)
        fe_estimator.fit(rng=rng)
    if params.cre:
        bdf = bdf.cluster(cluster_params, rng=rng)
        cre_estimator = tw.CREEstimator(bdf.to_eventstudy(is_sorted=True, copy=False).get_cs(is_sorted=True, copy=False), cre_params)
        cre_estimator.fit(rng=rng)
