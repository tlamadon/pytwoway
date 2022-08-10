'''
Borovickova and Shimer replication code.
'''
import warnings
import numpy as np
from pytwoway.util import weighted_var

def _compute_mean_sq(col_groupby, col_grouped, weights=None):
    '''
    Compute lambda_i_sq and mu_j_sq.

    Arguments:
        col_groupby (NumPy Array): data to group by
        col_grouped (NumPy Array): data to group
        weights (NumPy Array or None): weight column

    Returns:
        (NumPy Array): computed lambda_i_sq or mu_j_sq
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        if weights is None:
            # Split data
            agg_array = np.split(col_grouped, np.unique(col_groupby, return_index=True)[1])[1:] # aggregate(col_groupby, col_grouped, 'array', fill_value=[])
        else:
            # Split data
            groups = np.unique(col_groupby, return_index=True)[1]
            agg_array = np.split(col_grouped, groups)[1:] # aggregate(col_groupby, col_grouped, 'array', fill_value=[])
            weights = np.split(weights, groups)[1:] # aggregate(col_groupby, col_grouped, 'array', fill_value=[])

    res = np.zeros(len(agg_array))
    for i, agg_subarray in enumerate(agg_array):
        # Compute Cartesian product (source: https://stackoverflow.com/a/56261134/17333120)
        cart_prod = agg_subarray[None, :] * agg_subarray[:, None]
        # Faster sum source: https://stackoverflow.com/a/54629889/17333120
        cart_prod = (np.sum(cart_prod) - np.trace(cart_prod)) / 2

        if weights is None:
            # Taking mean over ((N * (N - 1)) / 2) parameters
            cart_prod_weights = (len(agg_subarray) * (len(agg_subarray) - 1)) / 2
        else:
            cart_prod_weights = weights[i][None, :] * weights[i][:, None]
            # Taking mean over ((N * (N - 1)) / 2) parameters
            cart_prod_weights = (np.sum(cart_prod_weights) - np.trace(cart_prod_weights)) / 2

        res[i] = cart_prod / cart_prod_weights

    return res

class BSEstimator():
    '''
    Class for estimating the non-parametric sorting model from Borovickova and Shimer. Results are stored in class attribute .res.
    '''

    def __init__(self):
        self.res = None

    def fit(self, adata, alternative_estimator=False, weighted=True):
        '''
        Estimate the non-parametric sorting model from Borovickova and Shimer.

        Arguments:
            adata (BipartiteDataFrame): long or collapsed long format labor data
            alternative_estimator (bool): if True, estimate using alternative estimator
            weighted (bool): if True, run estimator with weights
        '''
        if not adata.no_returns:
            # Borovickova-Shimer requires no returns
            raise ValueError("Cannot run the Borovickova-Shimer estimator if there are returns in the data. When cleaning your data, please set the parameter 'drop_returns' to drop returns.")

        if not adata._col_included('w'):
            # Skip weighting if no weight column included
            weighted = False

        if not alternative_estimator:
            #### Standard estimator ####
            ### Worker estimates ###
            if weighted:
                # If weighted, convert y to weighted y and compute sqrt(weights)
                adata.loc[:, 'unweighted_y'] = adata.loc[:, 'y']
                adata.loc[:, 'y'] = adata.loc[:, 'w'].to_numpy() * adata.loc[:, 'unweighted_y'].to_numpy()
                adata.loc[:, 'sqrt_w'] = np.sqrt(adata.loc[:, 'w'].to_numpy())
                # var(y)
                var_y = weighted_var(adata.loc[:, 'y'].to_numpy(), adata.loc[:, 'w'].to_numpy())
            else:
                # var(y)
                var_y = weighted_var(adata.loc[:, 'y'].to_numpy())

            ## Worker mean ##
            groupby_i = adata.groupby('i', sort=False)
            if weighted:
                weights_sum_i = groupby_i['w'].sum().to_numpy()
                lambda_i = groupby_i['y'].sum().to_numpy() / weights_sum_i
            else:
                weights_sum_i = groupby_i['y'].size().to_numpy()
                lambda_i = groupby_i['y'].mean().to_numpy()

            ## Worker mean squared ##
            if weighted:
                # If weighted, convert y to weighted y
                weights = adata.loc[:, 'sqrt_w'].to_numpy()
                weighted_y = weights * adata.loc[:, 'unweighted_y'].to_numpy()
            else:
                weights = None
                weighted_y = adata.loc[:, 'y'].to_numpy()

            lambda_i_sq = _compute_mean_sq(adata.loc[:, 'i'].to_numpy(), weighted_y, weights=weights)

            ## y_bar ##
            y_bar = np.sum(weights_sum_i * lambda_i) / np.sum(weights_sum_i)

            ## sigma_lambda_sq ##
            sigma_lambda_sq = np.sum(weights_sum_i * lambda_i_sq) / np.sum(weights_sum_i) - (y_bar ** 2)

            ## Covariance (worker component) ##
            y_i = groupby_i['y'].transform('sum').to_numpy()
            if weighted:
                weights_sum_i = groupby_i['w'].transform('sum').to_numpy()
                c_i = (y_i - adata.loc[:, 'y'].to_numpy()) / (weights_sum_i - adata.loc[:, 'w'].to_numpy())
            else:
                weights_sum_i = groupby_i['y'].transform('size').to_numpy()
                c_i = (y_i - adata.loc[:, 'y'].to_numpy()) / (weights_sum_i - 1)
            adata.loc[:, 'c_i'] = c_i

            ### Firm estimates ###
            adata.sort_values('j', axis=0, inplace=True)

            ## Firm mean ##
            groupby_j = adata.groupby('j', sort=False)
            if weighted:
                weights_sum_j = groupby_j['w'].sum().to_numpy()
                mu_j = groupby_j['y'].sum().to_numpy() / weights_sum_j
            else:
                weights_sum_j = groupby_j['y'].size().to_numpy()
                mu_j = groupby_j['y'].mean()

            ## Firm mean squared ##
            if weighted:
                # If weighted, convert y to weighted y
                weights = adata.loc[:, 'sqrt_w'].to_numpy()
                weighted_y = weights * adata.loc[:, 'unweighted_y'].to_numpy()
            else:
                weights = None
                weighted_y = adata.loc[:, 'y'].to_numpy()

            mu_j_sq = _compute_mean_sq(adata.loc[:, 'j'].to_numpy(), weighted_y, weights=weights)

            ## sigma_mu_sq ##
            sigma_mu_sq = np.sum(weights_sum_j * mu_j_sq) / np.sum(weights_sum_j) - (y_bar ** 2)

            ## Covariance (firm component) ##
            y_j = groupby_j['y'].transform('sum').to_numpy()
            if weighted:
                weights_sum_j = groupby_j['w'].transform('sum').to_numpy()
                c_j = (y_j - adata.loc[:, 'y'].to_numpy()) / (weights_sum_j - adata.loc[:, 'w'].to_numpy())
            else:
                weights_sum_j = groupby_j['y'].transform('size').to_numpy()
                c_j = (y_j - adata.loc[:, 'y'].to_numpy()) / (weights_sum_j - 1)

            ### Covariance ###
            c_i_j = adata.loc[:, 'c_i'].to_numpy() * c_j
            if weighted:
                cov = np.sum(adata.loc[:, 'w'].to_numpy() * c_i_j) / np.sum(adata.loc[:, 'w'].to_numpy()) - (y_bar ** 2)
            else:
                cov = np.mean(c_i_j) - (y_bar ** 2)

            # Store results
            self.res = {
                'var(y)': var_y,
                'mean(y)': y_bar,
                'var(lambda)': sigma_lambda_sq,
                'var(mu)': sigma_mu_sq,
                'cov(lambda, mu)': cov,
                'corr(lambda, mu)': cov / np.sqrt(sigma_lambda_sq * sigma_mu_sq)
            }

        else:
            #### Alternative estimator ####
            ### Worker estimates ###
            if weighted:
                # If weighted, convert y to weighted y and compute sqrt(weights)
                adata.loc[:, 'unweighted_y'] = adata.loc[:, 'y']
                adata.loc[:, 'y'] = adata.loc[:, 'w'].to_numpy() * adata.loc[:, 'unweighted_y'].to_numpy()
                # var(y)
                var_y = weighted_var(adata.loc[:, 'y'].to_numpy(), adata.loc[:, 'w'].to_numpy())
            else:
                # var(y)
                var_y = weighted_var(adata.loc[:, 'y'].to_numpy())

            ## Worker mean ##
            groupby_i = adata.groupby('i', sort=False)
            weighted_lambda_i = groupby_i['y'].mean().to_numpy()
            if weighted:
                weights_i = groupby_i['w'].mean().to_numpy()
                weights_sum_i = groupby_i['w'].sum().to_numpy()
            else:
                weights_i = 1
                weights_sum_i = groupby_i['y'].size().to_numpy()

            ## Worker mean squared ##
            weighted_lambda_i_sq = _compute_mean_sq(adata.loc[:, 'i'].to_numpy(), adata.loc[:, 'y'].to_numpy(), weights=None)
            if weighted:
                # FIXME I am not sure why this needs to be multiplied by 2
                weights_sq_i = 2 * _compute_mean_sq(adata.loc[:, 'i'].to_numpy(), adata.loc[:, 'w'].to_numpy(), weights=None)
            else:
                # FIXME I am not sure why this needs to be multiplied by 2
                weights_sq_i = 2 * (1 / 2)

            ## sigma_lambda_sq ##
            y_bar_i = np.sum(weights_sum_i * weighted_lambda_i) / np.sum(weights_sum_i * weights_i)
            sigma_lambda_sq = np.sum(weights_sum_i * weighted_lambda_i_sq) / np.sum(weights_sum_i * weights_sq_i) - (y_bar_i ** 2)

            ## Covariance (worker component) ##
            size_i = groupby_i['y'].transform('size').to_numpy()
            weighted_c_i = (groupby_i['y'].transform('sum').to_numpy() - adata.loc[:, 'y'].to_numpy()) / (size_i - 1)
            if weighted:
                weights_c_i = (groupby_i['w'].transform('sum').to_numpy() - adata.loc[:, 'w'].to_numpy()) / (size_i - 1)
            else:
                weights_c_i = 1
            adata.loc[:, 'weighted_c_i'] = weighted_c_i
            adata.loc[:, 'weights_c_i'] = weights_c_i

            ### Firm estimates ###
            adata.sort_values('j', axis=0, inplace=True)

            ## Firm mean ##
            groupby_j = adata.groupby('j', sort=False)
            weighted_mu_j = groupby_j['y'].mean().to_numpy()
            if weighted:
                weights_j = groupby_j['w'].mean().to_numpy()
                weights_sum_j = groupby_j['w'].sum().to_numpy()
            else:
                weights_j = 1
                weights_sum_j = groupby_j['y'].size().to_numpy()

            ## Firm mean squared ##
            weighted_mu_j_sq = _compute_mean_sq(adata.loc[:, 'j'].to_numpy(), adata.loc[:, 'y'].to_numpy(), weights=None)
            if weighted:
                # FIXME I am not sure why this needs to be multiplied by 2
                weights_sq_j = 2 * _compute_mean_sq(adata.loc[:, 'j'].to_numpy(), adata.loc[:, 'w'].to_numpy(), weights=None)
            else:
                # FIXME I am not sure why this needs to be multiplied by 2
                weights_sq_j = 2 * (1 / 2)

            ## sigma_mu_sq ##
            y_bar_j = np.sum(weights_sum_j * weighted_mu_j) / np.sum(weights_sum_j * weights_j)
            sigma_mu_sq = np.sum(weights_sum_j * weighted_mu_j_sq) / np.sum(weights_sum_j * weights_sq_j) - (y_bar_j ** 2)

            ## Covariance (firm component) ##
            size_j = groupby_j['y'].transform('size').to_numpy()
            weighted_c_j = (groupby_j['y'].transform('sum').to_numpy() - adata.loc[:, 'y'].to_numpy()) / (size_j - 1)
            if weighted:
                weights_c_j = (groupby_j['w'].transform('sum').to_numpy() - adata.loc[:, 'w'].to_numpy()) / (size_j - 1)
            else:
                weights_c_j = 1

            ### Covariance ###
            weighted_c_i_j = adata.loc[:, 'weighted_c_i'].to_numpy() * weighted_c_j
            weights_c_i_j = adata.loc[:, 'weights_c_i'].to_numpy() * weights_c_j
            if weighted:
                cov = np.sum(adata.loc[:, 'w'].to_numpy() * weighted_c_i_j) / np.sum(adata.loc[:, 'w'].to_numpy() * weights_c_i_j) - (y_bar_i * y_bar_j)
            else:
                cov = np.mean(weighted_c_i_j) - (y_bar_i * y_bar_j)

            # Store results
            
            self.res = {
                'var(y)': var_y,
                # 'y_bar_i': y_bar_i,
                # 'y_bar_j': y_bar_j,
                'mean(y)': (y_bar_i + y_bar_j) / 2,
                'var(lambda)': sigma_lambda_sq,
                'var(mu)': sigma_mu_sq,
                'cov(lambda, mu)': cov,
                'corr(lambda, mu)': cov / np.sqrt(sigma_lambda_sq * sigma_mu_sq)
            }

        ### Restore values ###
        if weighted:
            # Restore y
            adata.loc[:, 'y'] = adata.loc[:, 'unweighted_y']

            # Drop columns
            for col in ['unweighted_y', 'sqrt_w']:
                if col in adata.columns:
                    adata.drop(col, axis=1, inplace=True)

        # Drop columns
        for col in ['c_i', 'weighted_c_i', 'weights_c_i']:
            if col in adata.columns:
                adata.drop(col, axis=1, inplace=True)

        # Sort
        adata.sort_rows(copy=False)
