'''
Script to run pytwoway through the command line

Usage example:
pytw --my-config config.txt --fe --cre
'''
import configargparse
import ast
import pandas as pd
import bipartitepandas as bpd
from pytwoway import TwoWay as tw

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
    p.add('--filetype', required=False, help='file format of data')

    # Options to run FE or CRE
    p.add('--fe', action='store_true', help='run FE estimation') # This option can be set in a config file because it starts with '--'
    p.add('--cre', action='store_true', help='run CRE estimation')

    ##### TwoWay start #####
    p.add('--data', required=False, help='path to labor data file')
    p.add('--format', required=False, help="labor data format ('long' for long; 'long_collapsed' for collapsed long; 'es' for event study; or 'es_collapsed' for collapsed event study)")
    p.add('--col_dict', required=False, help='dictionary to correct column names')
    p.add('--collapsed', type=str2bool, required=False, help='if True, run estimators on collapsed data', default=True)
    ##### TwoWay end #####

    ##### Stata start #####
    p.add('--stata', action='store_true', required=False, help='if True, running estimators on Stata')
    ##### Stata end #####

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

    ##### FE start #####
    p.add('--ncore_fe', required=False, help='number of cores to use when computing fe')
    p.add('--batch', required=False, help='batch size to send in parallel when computing fe')
    p.add('--ndraw_pii', required=False, help='number of draw to use in approximation for leverages when computing fe')
    p.add('--levfile', required=False, help='file to load precomputed leverages when computing fe')
    p.add('--ndraw_tr_fe', required=False, help='number of draws to use in approximation for traces when computing fe')
    p.add('--he', type=str2bool, required=False, help='if True, compute the heteroskedastic correction when computing fe')
    p.add('--out_fe', required=False, help='outputfile where fe results are saved')
    p.add('--statsonly', type=str2bool, required=False, help='save data statistics only when computing fe')
    p.add('--Q', required=False, help="which Q matrix to consider when computing fe. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'")
    # p.add('--con', required=False, help='computes the smallest eigen values when computing fe, this is the filepath where these results are saved')
    # p.add('--logfile', required=False, help='log output to a logfile when computing fe')
    # p.add('--check', type=str2bool, required=False, help='whether to compute the non-approximated estimates as well when computing fe')
    ##### FE end #####

    ##### CRE start #####
    p.add('--ncore_cre', required=False, help='number of cores to use when computing cre')
    p.add('--ndraw_tr_cre', required=False, help='number of draws to use in approximation for traces when computing cre')
    p.add('--ndp', required=False, help=' number of draw to use in approximation for leverages when computing cre')
    p.add('--out_cre', required=False, help='outputfile where cre results are saved')
    p.add('--posterior', type=str2bool, required=False, help='whether to compute the posterior variance when computing cre')
    p.add('--wo_btw', required=False, help='sets between variation to 0, pure RE when computing cre')
    ##### CRE end #####

    ##### Clean start #####
    p.add('--connectedness', required=False, help="for data cleaning, if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations")
    p.add('--i_t_how', required=False, help="for data cleaning, if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids")
    p.add('--copy_clean', required=False, help='for data cleaning, if False, avoid copy')
    ##### Clean end #####

    params = p.parse_args()

    ##### TwoWay start #####
    if params.col_dict is not None:
        # Have to do ast.literal_eval twice for it to work properly
        col_dict = ast.literal_eval(ast.literal_eval(params.col_dict))
    else:
        col_dict = params.col_dict

    ##### Stata start #####
    if params.stata:
        params.data = 'leedtwoway_temp_data.dta'
        params.filetype = 'dta'
    ##### Stata end #####

    # Generate TwoWay dictionary
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
    tw_params = {'data': pd_from_filetype[params.filetype.lower()](params.data), 'formatting': params.format, 'col_dict': col_dict}
    tw_params = clear_dict(tw_params)
    ##### TwoWay end #####

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
    #### Grouping start ####
    ### KMeans start ###
    KMeans_params = {'n_clusters': params.n_clusters, 'init': params.init, 'n_init': params.n_init, 'max_iter': params.max_iter, 'tol': params.tol, 'precompute_distances': params.precompute_distances, 'verbose': params.verbose, 'random_state': params.random_state, 'copy_x': params.copy_x, 'n_jobs': params.n_jobs, 'algorithm': params.algorithm}
    KMeans_params = clear_dict(KMeans_params)
    ### KMeans end ###
    ### Quantiles start ###
    quantiles_params = {'n_quantiles': params.n_quantiles}
    quantiles_params = clear_dict(quantiles_params)
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
                measure_fn = bpd.measures.cdfs(**cdf_params)
            elif measure.lower() == 'moments':
                measure_fn = bpd.measures.moments(**moments_params)
            measures.append(measure_fn)
    else:
        measures = params.measures
    cluster_params = {'measures': measures, 'grouping': params.grouping, 'stayers_movers': params.stayers_movers, 't': params.t, 'weighted': params.weighted, 'dropna': params.dropna}
    cluster_params = clear_dict(cluster_params)
    #### General end ####
    ##### Cluster end #####

    ##### FE start #####
    fe_params = {'ncore': params.ncore_fe, 'batch': params.batch, 'ndraw_pii': params.ndraw_pii, 'levfile': params.levfile, 'ndraw_tr': params.ndraw_tr_fe, 'he': params.he, 'out': params.out_fe, 'statsonly': params.statsonly, 'Q': params.Q} # 'con': params.con, 'logfile': params.logfile, 'check': params.check
    fe_params = clear_dict(fe_params)
    ##### FE end #####

    ##### CRE start #####
    cre_params = {'ncore': params.ncore_cre, 'ndraw_tr': params.ndraw_tr_cre, 'ndp': params.ndp, 'out': params.out_cre, 'posterior': params.posterior, 'wo_btw': params.wo_btw}
    cre_params = clear_dict(cre_params)
    ##### CRE end #####

    ##### Clean start #####
    clean_params = {'connectedness': params.connectedness, 'i_t_how': params.i_t_how, 'copy': params.copy_clean}
    clean_params = clear_dict(clean_params)
    ##### Clean end #####

    # Run estimation
    if params.fe or params.cre:
        tw_net = tw(**tw_params)

        if params.fe:
            tw_net.prep_data(collapsed=params.collapsed, user_clean=clean_params)
            tw_net.fit_fe(user_fe=fe_params)

        if params.cre:
            tw_net.prep_data(collapsed=params.collapsed, user_clean=clean_params, he=params.he) # Note that if params.he is None the code still works
            tw_net.cluster(**cluster_params)
            tw_net.fit_cre(user_cre=cre_params)
