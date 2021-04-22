'''
Class for a two-way fixed effect network
'''
import warnings
import pytwoway as tw
import bipartitepandas as bpd

class TwoWay():
    '''
    Class of TwoWay, where TwoWay gives a network of firms and workers. Inherits from bipartitepandas.
    '''

    def __init__(self, data, formatting='long', col_dict=None):
        '''
        Arguments:
            data (Pandas DataFrame): data giving firms, workers, and compensation
            formatting (str): if 'long', then data in long format; if 'long_collapsed' then in collapsed long format; if 'es', then data in event study format; if 'es_collapsed' then in collapsed event study format. If simulating data, keep default value of 'long'
            col_dict (dict): make data columns readable. Keep None if column names already correct. Options for:

                long: requires: i (worker id), j (firm id), y (compensation), t (period); optional: g (firm cluster), m (0 if stayer, 1 if mover)

                collapsed long: requires: i (worker id), j (firm id), y (compensation), t1 (first period in spell), t2 (last period in spell); optional: w (weight), g (firm cluster), m (0 if stayer, 1 if mover)

                event study: requires: i (worker id), j1 (firm 1 id), j2 (firm 2 id), y1 (compensation 1), y2 (compensation 2); optional: t1 (time of observation 1), t2 (time of observation 2), g1 (firm 1 cluster), g2 (firm 2 cluster), m (0 if stayer, 1 if mover)

                collapsed event study: requires: i (worker id), j1 (firm 1 id), j2 (firm 1 id), y1 (compensation 1), y2 (compensation 2); optional: t11 (first period in observation 1 spell), t12 (last period in observation 1 spell), t21 (first period in observation 2 spell), t22 (last period in observation 2 spell), w1 (weight 1), w2 (weight 2), g1 (firm 1 cluster), g2 (firm 2 cluster), m (0 if stayer, 1 if mover)
        '''
        # Start logger
        bpd.logger_init(self)
        # self.logger.info('initializing TwoWay object')

        type_dict = { # Determine type based on formatting
            'long': bpd.BipartiteLong,
            'long_collapsed': bpd.BipartiteLongCollapsed,
            'es': bpd.BipartiteEventStudy,
            'es_collapsed': bpd.BipartiteEventStudyCollapsed
        }

        self.data = type_dict[formatting](data, col_dict=col_dict)

        self.formatting = formatting
        self.clean = False # Whether data is clean

        # self.logger.info('TwoWay object initialized')

    def __prep_data(self, collapsed=True, user_clean={}):
        '''
        Prepare bipartite network for running estimators.

        Arguments:
            collapsed (bool): if True, run estimators on collapsed data
            user_clean (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

        Returns:
            frame (BipartitePandas): prepared data
        '''
        if not self.clean:
            self.data = self.data.clean_data(user_clean=user_clean)
            self.clean = True

        frame = self.data.copy()

        if not collapsed:
            if self.formatting == 'long':
                frame = frame.get_es()
            elif self.formatting != 'es':
                warnings.warn('Data already collapsed, running estimator on collapsed data')
                collapsed = True
        if collapsed:
            if self.formatting == 'es':
                frame = frame.get_long()
            if self.formatting in ['es', 'long']:
                frame = frame.get_collapsed_long()
            if self.formatting in ['es', 'long', 'long_collapsed']:
                frame = frame.get_es()

        return frame

    def fit_fe(self, user_fe={}, collapsed=True, user_clean={}):
        '''
        Fit the bias-corrected FE estimator. Saves two dictionary attributes: self.fe_res (complete results) and self.fe_summary (summary results).

        Arguments:
            user_fe (dict): dictionary of parameters for bias-corrected FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    h2 (bool, default=False): if True, compute h2 correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

            collapsed (bool): if True, run estimators on collapsed data
            user_clean (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
        '''
        # Prepare data
        frame = self.__prep_data(collapsed=collapsed, user_clean=user_clean)

        # Run estimator
        fe_solver = tw.FEEstimator(frame.get_cs(), user_fe)
        fe_solver.fit_1()
        fe_solver.construct_Q() # Comment out this line and manually create Q if you want a custom Q matrix
        fe_solver.fit_2()

        self.fe_res = fe_solver.res
        self.fe_summary = fe_solver.summary

    def fit_cre(self, user_cre={}, user_cluster={}, collapsed=True, user_clean={}):
        '''
        Fit the CRE estimator. Saves two dictionary attributes: self.cre_res (complete results) and self.cre_summary (summary results).

        Arguments:
            user_cre (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    ndp (int, default=50): number of draw to use in approximation for leverages

                    out (str, default='res_cre.json'): outputfile where results are saved

                    posterior (bool, default=False): if True, compute posterior variance

                    wo_btw (bool, default=False): if True, sets between variation to 0, pure RE

            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int, default=10): how many values to use to approximate the cdfs

                    grouping (str, default='quantile_all'): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool, default=False): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

            collapsed (bool): if True, run estimators on collapsed data
            user_clean (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
        '''
        # Prepare data
        frame = self.__prep_data(collapsed=collapsed, user_clean=user_clean)
        frame = frame.cluster(user_cluster=user_cluster)

        # Run estimator
        cre_solver = tw.CREEstimator(frame.get_cs(), user_cre)
        cre_solver.fit()

        self.cre_res = cre_solver.res
        self.cre_summary = cre_solver.summary

    def summary_fe(self):
        '''
        Return summary results for FE estimator.

        Returns:
            self.fe_summary (dict): dictionary of FE summary results
        '''
        return self.fe_summary

    def summary_cre(self):
        '''
        Return summary results for CRE estimator.

        Returns:
            self.cre_summary (dict): dictionary of CRE summary results
        '''
        return self.cre_summary
