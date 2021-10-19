'''
Class for Attrition plots
'''
from multiprocessing import Pool
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bipartitepandas as bpd
import pytwoway as tw
from tqdm import trange
import warnings

class TwoWayAttrition:
    '''
    Class of TwoWayAttrition, which generates attrition plots using bipartite labor data.

    Arguments:
        bdf (BipartiteBase): bipartite dataframe
    '''

    def __init__(self, bdf):
        self.bdf = bdf # type(bdf)(bdf, include_id_reference_dict=True)

        # Prevent plotting until results exist
        self.res = False

        self.default_attrition = {
            'type_and_subsets': ('increasing', np.linspace(0.1, 0.5, 5)), # How to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets)
            'threshold': 15, # Minimum number of movers required to keep a firm
            'copy': True # If False, avoid copy
        }

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _attrition_interior(self, bdf, fe_params={}, cre_params={}, cluster_params={}):
        '''
        Estimate all parameters of interest. This is the interior function to attrition_single.

        Arguments:
            bdf (BipartiteBase): bipartite dataframe
            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    he (bool, default=False): if True, compute heteroskedastic correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    feonly (bool, default=False): if True, compute only fixed effects and not variances

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

                    seed (int, default=None): NumPy RandomState seed

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    ndp (int, default=50): number of draw to use in approximation for leverages

                    out (str, default='res_cre.json'): outputfile where results are saved

                    posterior (bool, default=False): if True, compute posterior variance

                    wo_btw (bool, default=False): if True, sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    measures (function or list of functions): how to compute measures for clustering. Options can be seen in bipartitepandas.measures.

                    grouping (function): how to group firms based on measures. Options can be seen in bipartitepandas.grouping.

                    stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool, default=False): if True, drop observations where firms aren't clustered; if False, keep all observations

        Returns:
            {'fe': tw_net.fe_res, 'cre': tw_net.cre_res} (dict): FE results, CRE results
        '''
        # Use data to create TwoWay object (note that data is already clean)
        tw_net = tw.TwoWay(bdf)
        # Estimate FE model
        tw_net.fit_fe(user_fe=fe_params)
        # Cluster data
        tw_net.cluster(**cluster_params)
        # Estimate CRE model
        tw_net.fit_cre(user_cre=cre_params)

        return {'fe': tw_net.fe_res, 'cre': tw_net.cre_res}

    def _attrition_single(self, ncore=1, attrition_params={}, fe_params={}, cre_params={}, cluster_params={}, clean_params={}):
        '''
        Run attrition estimations of TwoWay to estimate parameters given fraction of movers remaining. This is the interior function to attrition.

        Arguments:
            ncore (int): how many cores to use
            attrition_params (dict): dictionary of parameters for attrition

                Dictionary parameters:

                    type_and_subsets (tuple of (str, list), default=('increasing', np.linspace(0.1, 0.5, 5))): how to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets)

                    threshold (int, default=15): minimum number of movers required to keep a firm

                    copy (bool, default=True): if False, avoid copy

            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    he (bool, default=False): if True, compute heteroskedastic correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    feonly (bool, default=False): if True, compute only fixed effects and not variances

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

                    seed (int, default=None): NumPy RandomState seed

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    ndp (int, default=50): number of draw to use in approximation for leverages

                    out (str, default='res_cre.json'): outputfile where results are saved

                    posterior (bool, default=False): if True, compute posterior variance

                    wo_btw (bool, default=False): if True, sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    measures (function or list of functions): how to compute measures for clustering. Options can be seen in bipartitepandas.measures.

                    grouping (function): how to group firms based on measures. Options can be seen in bipartitepandas.grouping.

                    stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool, default=False): if True, drop observations where firms aren't clustered; if False, keep all observations

            clean_params (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy

        Returns:
            res_all (dict of dicts of lists): in the first dictionary we choose 'connected' or 'biconnected'; in the second dictionary we choose 'fe' or 'cre'; and finally, we are given a list of results for each attrition percentage.
        '''
        # Create lists to save results
        res_connected = {'fe': [], 'cre': []} # For non-HE
        res_biconnected = {'fe': [], 'cre': []} # For HE
        res_all = [res_connected, res_biconnected]
        # FE params
        fe_params_connected = fe_params.copy()
        fe_params_connected['he'] = False
        fe_params_biconnected = fe_params.copy()
        fe_params_biconnected['he'] = True
        fe_params_all = [fe_params_connected, fe_params_biconnected]
        # Clean params
        clean_params_connected = clean_params.copy()
        clean_params_connected['connectedness'] = 'connected'
        clean_params_biconnected = clean_params.copy()
        clean_params_biconnected['connectedness'] = 'biconnected'
        clean_params_all = [clean_params_connected, clean_params_biconnected]

        for i in range(2):
            # Create iterator
            if attrition_params['type_and_subsets'][0] == 'increasing':
                attrition_iterator = self.bdf.attrition_increasing(subsets=attrition_params['type_and_subsets'][1], threshold=attrition_params['threshold'], user_clean=clean_params_all[i], copy=attrition_params['copy'])
            elif attrition_params['type_and_subsets'][0] == 'decreasing':
                attrition_iterator = self.bdf.attrition_increasing(subsets=attrition_params['type_and_subsets'][1], threshold=attrition_params['threshold'], user_clean=clean_params_all[i], copy=attrition_params['copy'])

            # Use multi-processing
            if ncore > 1:
                # Estimate
                with Pool(processes=ncore) as pool:
                    V = pool.starmap(self._attrition_interior, [(attrition_subset, fe_params_all[i], cre_params, cluster_params) for attrition_subset in attrition_iterator])
                for res in enumerate(V):
                    res_all[i]['fe'].append(res[1]['fe'])
                    res_all[i]['cre'].append(res[1]['cre'])
            else:
                for attrition_subset in attrition_iterator:
                    # Estimate
                    res = self._attrition_interior(attrition_subset, fe_params=fe_params_all[i], cre_params=cre_params, cluster_params=cluster_params)
                    res_all[i]['fe'].append(res['fe'])
                    res_all[i]['cre'].append(res['cre'])

        res_all = {'connected': res_all[0], 'biconnected': res_all[1]}

        return res_all

    def attrition(self, N=10, ncore=1, attrition_params={}, fe_params={}, cre_params={}, cluster_params={}, clean_params={}):
        '''
        Run Monte Carlo on attrition estimations of TwoWay to estimate variance of parameter estimates given fraction of movers remaining.

        Arguments:
            N (int): number of simulations
            ncore (int): how many cores to use
            attrition_params (dict): dictionary of parameters for attrition

                Dictionary parameters:

                    type_and_subsets (tuple of (str, list), default=('increasing', np.linspace(0.1, 0.5, 5))): how to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets)

                    threshold (int, default=15): minimum number of movers required to keep a firm

                    copy (bool, default=True): if False, avoid copy

            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    he (bool, default=False): if True, compute heteroskedastic correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    feonly (bool, default=False): if True, compute only fixed effects and not variances

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

                    seed (int, default=None): NumPy RandomState seed

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    ndp (int, default=50): number of draw to use in approximation for leverages

                    out (str, default='res_cre.json'): outputfile where results are saved

                    posterior (bool, default=False): if True, compute posterior variance

                    wo_btw (bool, default=False): if True, sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    measures (function or list of functions): how to compute measures for clustering. Options can be seen in bipartitepandas.measures.

                    grouping (function): how to group firms based on measures. Options can be seen in bipartitepandas.grouping.

                    stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool, default=False): if True, drop observations where firms aren't clustered; if False, keep all observations

            clean_params (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy

        Returns:
            res_all (dict of dicts of lists of lists): in the first dictionary we choose 'connected' or 'biconnected'; in the second dictionary we choose 'fe' or 'cre'; then, we are given a list of results for each Monte Carlo simulation; and finally, for a particular Monte Carlo simulation, we are given a list of results for each attrition percentage.
        '''
        # Create lists to save results
        res_connected = {'fe': [], 'cre': []} # For non-HE
        res_biconnected = {'fe': [], 'cre': []} # For HE

        attrition_params_full = bpd.util.update_dict(self.default_attrition, attrition_params)

        # Use multi-processing
        if False: # ncore > 1:
            # Estimate
            with Pool(processes=ncore) as pool:
                V = pool.starmap(self._attrition_single, [(ncore, attrition_params_full, fe_params, cre_params, cluster_params, clean_params) for _ in range(N)])
            for res in enumerate(V):
                res_connected['fe'].append(res[1]['connected']['fe'])
                res_connected['cre'].append(res[1]['connected']['cre'])
                res_biconnected['fe'].append(res[1]['biconnected']['fe'])
                res_biconnected['cre'].append(res[1]['biconnected']['cre'])
        else:
            for i in trange(N):
                # Estimate
                res = self._attrition_single(ncore=ncore, attrition_params=attrition_params_full, fe_params=fe_params, cre_params=cre_params, cluster_params=cluster_params, clean_params=clean_params)
                res_connected['fe'].append(res['connected']['fe'])
                res_connected['cre'].append(res['connected']['cre'])
                res_biconnected['fe'].append(res['biconnected']['fe'])
                res_biconnected['cre'].append(res['biconnected']['cre'])

        res_all = {'connected': res_connected, 'biconnected': res_biconnected}

        self.res = res_all

    def plot_attrition(self, subsets=np.linspace(0.1, 0.5, 5)):
        '''
        Plot results from Monte Carlo simulations.

        Arguments:
            subsets (list): subsets to consider for attrition
        '''
        if not self.monte_carlo_res:
            warnings.warn('Must generate attrition data before results can be plotted. This can be done by running .attrition()')

        else:
            # Get N, M
            N = len(self.res['connected']['fe']) # Number of estimations
            M = len(self.res['connected']['fe'][0]) # Number of attritions per estimation
            # Extract results
            # Connected set
            conset_var_psi_pct = np.zeros(shape=[N, M, 3])
            conset_cov_psi_alpha_pct = np.zeros(shape=[N, M, 3])
            # Biconnected set
            biconset_var_psi_pct = np.zeros(shape=[N, M, 4])
            biconset_cov_psi_alpha_pct = np.zeros(shape=[N, M, 4])
            for i in range(N):
                for j in range(M):
                    # Connected set
                    con_res_fe_dict = self.res['connected']['fe'][i][j]
                    con_res_cre_dict = self.res['connected']['cre'][i][j]
                    # Var(psi)
                    conset_var_psi_pct[i, j, 0] = float(con_res_fe_dict['var_fe']) / float(con_res_fe_dict['var_y'])
                    conset_var_psi_pct[i, j, 1] = float(con_res_fe_dict['var_ho']) / float(con_res_fe_dict['var_y'])
                    conset_var_psi_pct[i, j, 2] = float(con_res_cre_dict['tot_var']) / float(con_res_cre_dict['var_y'])
                    # Cov(psi, alpha)
                    conset_cov_psi_alpha_pct[i, j, 0] = 2 * float(con_res_fe_dict['cov_fe']) / float(con_res_fe_dict['var_y'])
                    conset_cov_psi_alpha_pct[i, j, 1] = 2 * float(con_res_fe_dict['cov_ho']) / float(con_res_fe_dict['var_y'])
                    conset_cov_psi_alpha_pct[i, j, 2] = 2 * float(con_res_cre_dict['tot_cov']) / float(con_res_cre_dict['var_y'])

                    # Biconnected set
                    bicon_res_fe_dict = self.res['biconnected']['fe'][i][j]
                    bicon_res_cre_dict = self.res['biconnected']['cre'][i][j]
                    # Var(psi)
                    biconset_var_psi_pct[i, j, 0] = float(bicon_res_fe_dict['var_fe']) / float(bicon_res_fe_dict['var_y'])
                    biconset_var_psi_pct[i, j, 1] = float(bicon_res_fe_dict['var_ho']) / float(bicon_res_fe_dict['var_y'])
                    biconset_var_psi_pct[i, j, 2] = float(bicon_res_cre_dict['tot_var']) / float(bicon_res_cre_dict['var_y'])
                    biconset_var_psi_pct[i, j, 3] = float(bicon_res_fe_dict['var_he']) / float(bicon_res_fe_dict['var_y'])

                    # Cov(psi, alpha)
                    biconset_cov_psi_alpha_pct[i, j, 0] = 2 * float(bicon_res_fe_dict['cov_fe']) / float(bicon_res_fe_dict['var_y'])
                    biconset_cov_psi_alpha_pct[i, j, 1] = 2 * float(bicon_res_fe_dict['cov_ho']) / float(bicon_res_fe_dict['var_y'])
                    biconset_cov_psi_alpha_pct[i, j, 2] = 2 * float(bicon_res_cre_dict['tot_cov']) / float(bicon_res_cre_dict['var_y'])
                    biconset_cov_psi_alpha_pct[i, j, 3] = 2 * float(bicon_res_fe_dict['cov_he']) / float(bicon_res_fe_dict['var_y'])

            # x-axis
            x_axis = (100 * subsets).astype(int)
            conset_var_psi_pct = 100 * conset_var_psi_pct
            conset_cov_psi_alpha_pct = 100 * conset_cov_psi_alpha_pct
            biconset_var_psi_pct = 100 * biconset_var_psi_pct
            biconset_cov_psi_alpha_pct = 100 * biconset_cov_psi_alpha_pct

            # Plot figures
            # Firm effects (connected set)
            plt.plot(x_axis, conset_var_psi_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, conset_var_psi_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, conset_var_psi_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.title('Firm effects (connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Firm Effects: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.show()
            # Firm effects (biconnected set)
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 3].mean(axis=0), label='HE')
            plt.title('Firm effects (biconnected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Firm Effects: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.show()

            # Sorting (connected set)
            plt.plot(x_axis, conset_cov_psi_alpha_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, conset_cov_psi_alpha_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, conset_cov_psi_alpha_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.title('Sorting (connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Sorting: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.show()
            # Sorting (biconnected set)
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 3].mean(axis=0), label='HE')
            plt.title('Sorting (biconnected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Sorting: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.show()

            # Plot boxplots
            # Firm effects (connected set)
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE']

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(conset_var_psi_pct[:, :, i], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[i])
            fig.suptitle('Firm effects (connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            fig.show()
            # Firm effects (biconnected set)
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
            order = [0, 1, 3, 2] # Because data is FE, FE-HO, CRE, FE-HE

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(biconset_var_psi_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Firm effects (biconnected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            fig.show()

            # Sorting (connected set)
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE']

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(conset_cov_psi_alpha_pct[:, :, i], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[i])
            fig.suptitle('Sorting (connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            fig.show()
            # Sorting (biconnected set)
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
            order = [0, 1, 3, 2] # Because data is FE, FE-HO, CRE, FE-HE

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(biconset_cov_psi_alpha_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Sorting (biconnected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            fig.show()
