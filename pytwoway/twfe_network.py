'''
Class for a two-way fixed effect network
'''
import logging
from pathlib import Path
import pytwoway as tw

class TwoWay:
    '''
    Class of TwoWay, where TwoWay gives a network of firms and workers.
    '''

    def __init__(self, data, formatting='long', col_dict=None):
        '''
        Arguments:
            data (Pandas DataFrame): data giving firms, workers, and compensation
            formatting (str): if 'long', then data in long format; if 'es', then data in event study format. If simulating data, keep default value of 'long'
            col_dict (dict): make data columns readable (requires:
                if long: wid (worker id), comp (compensation), fid (firm id), year;
                if event study: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover);
                    optionally include: year_end_1 (last year of observation 1), year_end_2 (last year of observation 2), w1 (weight 1), w2 (weight 2)).
                Keep None if column names already correct
        '''
        # Begin logging
        self.logger = logging.getLogger('twoway')
        self.logger.setLevel(logging.DEBUG)
        # Create logs folder
        Path('twoway_logs').mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('twoway_logs/twoway_spam.log')
        fh.setLevel(logging.DEBUG)
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info('initializing TwoWay object')

        # Define some attributes
        self.b_net = tw.BipartiteData(data, formatting, col_dict)

        # Define default parameter dictionaries
        self.default_fe = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'hetero': False, 'out': 'res_fe.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False, 'Q': 'cov(alpha, psi)'} # Do not define 'data' because will be updated later

        self.default_cre = {'ncore': 1, 'ndraw_tr': 5, 'ndp': 50, 'out': 'res_cre.json', 'posterior': False, 'wo_btw': False} # Do not define 'data' because will be updated later

        self.logger.info('TwoWay object initialized')

    def __prep_fe(self):
        '''
        Prepare bipartite network for running fit_fe.
        '''
        self.b_net.clean_data()
        self.b_net.long_to_es()

    def __prep_cre(self, user_cluster={}):
        '''
        Prepare bipartite network for running fit_cre.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified
        '''
        self.b_net.clean_data()
        self.b_net.long_to_es()
        self.b_net.cluster(user_cluster=user_cluster)

    def fit_fe(self, user_fe={}):
        '''
        Fit the bias-corrected FE estimator.

        Arguments:
            user_fe (dict): dictionary of parameters for bias-corrected FE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    batch (int): batch size to send in parallel

                    ndraw_pii (int): number of draw to use in approximation for leverages

                    ndraw_tr (int): number of draws to use in approximation for traces

                    check (bool): whether to compute the non-approximated estimates as well

                    hetero (bool): whether to compute the heteroskedastic estimates

                    out (str): outputfile

                    con (str): computes the smallest eigen values, this is the filepath where these results are saved

                    logfile (str): log output to a logfile

                    levfile (str): file to load precomputed leverages

                    statsonly (bool): save data statistics only

                    Q (str): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

        Returns:
            fe_res (dict): dictionary of results
        '''
        self.__prep_fe()
        fe_params = tw.update_dict(self.default_fe, user_fe)

        fe_params['data'] = self.b_net.es_to_cs() # Make sure to use up-to-date bipartite network

        fe_solver = tw.FESolver(fe_params)
        fe_solver.fit_1()
        fe_solver.construct_Q() # Comment out this line and manually create Q if you want a custom Q matrix
        fe_solver.fit_2()

        fe_res = fe_solver.res

        return fe_res

    def fit_cre(self, user_cre={}, user_cluster={}):
        '''
        Fit the CRE estimator.

        Arguments:
            user_cre (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    ndraw_tr (int): number of draws to use in approximation for traces

                    ndp (int): number of draw to use in approximation for leverages

                    out (str): outputfile

                    posterior (bool): compute posterior variance

                    wo_btw (bool): sets between variation to 0, pure RE

            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified

        Returns:
            cre_res (dict): dictionary of results
        '''
        self.__prep_cre(user_cluster=user_cluster)
        cre_params = tw.update_dict(self.default_cre, user_cre)

        cre_params['data'] = self.b_net.es_to_cs() # Make sure to use up-to-date data

        cre_solver = tw.CRESolver(cre_params)
        cre_solver.fit()

        cre_res = cre_solver.res

        return cre_res
