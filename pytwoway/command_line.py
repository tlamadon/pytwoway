'''
Script to run twfe_network through the command line

Usage example:
python3 run_twfe.py --my-config config.txt --filetype csv --akm --cre
'''
import configargparse
import ast
import pandas as pd
from pytwoway import twfe_network
tn = twfe_network.twfe_network

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

    # Options to run AKM or CRE
    p.add('--akm', action='store_true', help='run AKM estimation') # this option can be set in a config file because it starts with '--'
    p.add('--cre', action='store_true', help='run CRE estimation')

    ##### twfe_network start #####
    p.add('--data', required=True, help='path to labor data file')
    p.add('--format', required=False, help="labor data format ('long' or 'es' for event study)")
    p.add('--col_dict', required=False, help='dictionary to correct column names')
    ##### twfe_network end #####

    ##### KMeans start #####
    p.add('--n_clusters', required=False, help='number of clusters for KMeans algorithm')
    p.add('--init', required=False)
    p.add('--n_init', required=False)
    p.add('--max_iter', required=False)
    p.add('--tol', required=False)
    p.add('--precompute_distances', required=False)
    p.add('--verbose', required=False)
    p.add('--random_state', required=False)
    p.add('--copy_x', required=False)
    p.add('--n_jobs', required=False)
    p.add('--algorithm', required=False)
    ##### KMeans end #####

    ##### Cluster start #####
    p.add('--cdf_resolution', required=False, help='length of cdf array for computing clusters')
    p.add('--grouping', required=False, help='how to compute cdfs')
    p.add('--year', required=False, help='cluster on specific year')
    ##### Cluster end #####

    ##### AKM start #####
    p.add('--ncore_akm', required=False, help='number of cores when computing akm')
    p.add('--batch', required=False)
    p.add('--ndraw_pii', required=False)
    p.add('--ndraw_tr_akm', required=False)
    p.add('--check', required=False)
    p.add('--hetero', required=False)
    p.add('--out_akm', required=False, help='filepath for akm results')
    p.add('--con', required=False)
    p.add('--logfile', required=False)
    p.add('--levfile', required=False)
    p.add('--statsonly', required=False, help='only compute basic statistics')
    p.add('--Q', required=False, help='custom Q matrix')
    ##### AKM end #####

    ##### CRE start #####
    p.add('--ncore_cre', required=False, help='number of cores when computing cre')
    p.add('--ndraw_tr_cre', required=False)
    p.add('--ndp', required=False)
    p.add('--out_cre', required=False, help='filepath for cre results')
    p.add('--posterior', required=False)
    p.add('--wobtw', required=False)
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

    ##### AKM start #####
    akm_params = {'ncore': params.ncore_akm, 'batch': params.batch, 'ndraw_pii': params.ndraw_pii, 'ndraw_tr': params.ndraw_tr_akm, 'check': params.check, 'hetero': params.hetero, 'out': params.out_akm, 'con': params.con, 'logfile': params.logfile, 'levfile': params.levfile, 'statsonly': params.statsonly, 'Q': params.Q}
    akm_params = clear_dict(akm_params)
    ##### AKM end #####

    ##### CRE start #####
    cre_params = {'ncore': params.ncore_cre, 'ndraw_tr': params.ndraw_tr_cre, 'ndp': params.ndp, 'out': params.out_cre, 'posterior': params.posterior, 'wobtw': params.wobtw}
    cre_params = clear_dict(cre_params)
    ##### CRE end #####

    # Run estimation
    if params.akm or params.cre:
        net = tn(**twfe_params)
        net.clean_data()
        net.refactor_es()

        if params.akm:
            net.run_akm_corrected(user_akm=akm_params)

        if params.cre:
            net.cluster(user_cluster=cluster_params)
            net.run_cre(user_cre=cre_params)
