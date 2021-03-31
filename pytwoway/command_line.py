'''
Script to run twfe_network through the command line

Usage example:
pytw --my-config config.txt --fe --cre
'''
import configargparse
import ast
import pandas as pd
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

def main():
    p = configargparse.ArgParser()
    p.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

    # Options to select filetype
    p.add('--filetype', required=False, help='file format of data')

    # Options to run FE or CRE
    p.add('--fe', action='store_true', help='run FE estimation') # this option can be set in a config file because it starts with '--'
    p.add('--cre', action='store_true', help='run CRE estimation')

    ##### twfe_network start #####
    p.add('--data', required=True, help='path to labor data file')
    p.add('--format', required=False, help="labor data format ('long' or 'es' for event study)")
    p.add('--col_dict', required=False, help='dictionary to correct column names')
    ##### twfe_network end #####

    ##### KMeans start #####
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
    ##### KMeans end #####

    ##### Cluster start #####
    p.add('--cdf_resolution', required=False, help='how many values to use to approximate the cdf when clustering')
    p.add('--grouping', required=False, help='''
    how to group the cdfs when clustering ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
    ''')
    p.add('--year', required=False, help='if None, uses entire dataset when clustering; if int, gives year of data to consider when clustering')
    ##### Cluster end #####

    ##### FE start #####
    p.add('--ncore_fe', required=False, help='number of cores to use when computing fe')
    p.add('--batch', required=False, help='batch size to send in parallel when computing fe')
    p.add('--ndraw_pii', required=False, help='number of draw to use in approximation for leverages when computing fe')
    p.add('--ndraw_tr_fe', required=False, help='number of draws to use in approximation for traces when computing fe')
    p.add('--check', required=False, help='whether to compute the non-approximated estimates as well when computing fe')
    p.add('--hetero', required=False, help='whether to compute the heteroskedastic estimates when computing fe')
    p.add('--out_fe', required=False, help='filepath for fe results')
    p.add('--con', required=False, help='computes the smallest eigen values when computing fe, this is the filepath where these results are saved')
    p.add('--logfile', required=False, help='log output to a logfile when computing fe')
    p.add('--levfile', required=False, help='file to load precomputed leverages when computing fe')
    p.add('--statsonly', required=False, help='save data statistics only when computing fe')
    p.add('--Q', required=False, help="which Q matrix to consider when computing fe. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'")
    ##### FE end #####

    ##### CRE start #####
    p.add('--ncore_cre', required=False, help='number of cores to use when computing cre')
    p.add('--ndraw_tr_cre', required=False, help='number of draws to use in approximation for traces when computing cre')
    p.add('--ndp', required=False, help=' number of draw to use in approximation for leverages when computing cre')
    p.add('--out_cre', required=False, help='filepath for cre results')
    p.add('--posterior', required=False, help='whether to compute the posterior variance when computing cre')
    p.add('--wo_btw', required=False, help='sets between variation to 0, pure RE when computing cre')
    ##### CRE end #####

    params = p.parse_args()

    ##### twfe_network start #####
    if params.col_dict is not None:
        # Have to do ast.literal_eval twice for it to work properly
        col_dict = ast.literal_eval(ast.literal_eval(params.col_dict))
    else:
        col_dict = params.col_dict

    # Generate twfe_params dictionary
    if params.filetype == 'csv':
        twfe_params = {'data': pd.read_csv(params.data), 'formatting': params.format, 'col_dict': col_dict}
    elif params.filetype == 'ftr':
        twfe_params = {'data': pd.read_feather(params.data), 'formatting': params.format, 'col_dict': col_dict}
    elif params.filetype == 'dta':
        twfe_params = {'data': pd.read_stata(params.data), 'formatting': params.format, 'col_dict': col_dict}
    twfe_params = clear_dict(twfe_params)
    ##### twfe_network end #####

    ##### KMeans start #####
    KMeans_params = {'n_clusters': params.n_clusters, 'init': params.init, 'n_init': params.n_init, 'max_iter': params.max_iter, 'tol': params.tol, 'precompute_distances': params.precompute_distances, 'verbose': params.verbose, 'random_state': params.random_state, 'copy_x': params.copy_x, 'n_jobs': params.n_jobs, 'algorithm': params.algorithm}
    KMeans_params = clear_dict(KMeans_params)
    ##### KMeans end #####

    ##### Cluster start #####
    cluster_params = {'cdf_resolution': params.cdf_resolution, 'grouping': params.grouping, 'year': params.year, 'user_KMeans': KMeans_params}
    cluster_params = clear_dict(cluster_params)
    ##### Cluster end #####

    ##### FE start #####
    fe_params = {'ncore': params.ncore_fe, 'batch': params.batch, 'ndraw_pii': params.ndraw_pii, 'ndraw_tr': params.ndraw_tr_fe, 'check': params.check, 'hetero': params.hetero, 'out': params.out_fe, 'con': params.con, 'logfile': params.logfile, 'levfile': params.levfile, 'statsonly': params.statsonly, 'Q': params.Q}
    fe_params = clear_dict(fe_params)
    ##### FE end #####

    ##### CRE start #####
    cre_params = {'ncore': params.ncore_cre, 'ndraw_tr': params.ndraw_tr_cre, 'ndp': params.ndp, 'out': params.out_cre, 'posterior': params.posterior, 'wo_btw': params.wo_btw}
    cre_params = clear_dict(cre_params)
    ##### CRE end #####

    # Run estimation
    if params.fe or params.cre:
        tw_net = tw(**twfe_params)

        if params.fe:
            tw_net.fit_fe(user_fe=fe_params)

        if params.cre:
            tw_net.fit_cre(user_cre=cre_params, user_cluster=cluster_params)
