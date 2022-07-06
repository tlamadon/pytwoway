'''
Borovickova and Shimer replication code.
'''
import warnings
import numpy as np

def _compute_mean_sq(col_groupby, col_grouped, weights=None):
    '''
    Compute lambda_sq_i and mu_sq_j.

    Arguments:
        col_groupby (NumPy Array): data to group by
        col_grouped (NumPy Array): data to group
        weights (NumPy Array or None): weight column

    Returns:
        (NumPy Array): computed lambda_sq_i or mu_sq_j
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
        cart_prod = np.sum(np.triu(cart_prod, 1))

        if weights is None:
            cart_prod_weights = len(agg_subarray) * (len(agg_subarray) - 1)
        else:
            cart_prod_weights = weights[i][None, :] * weights[i][:, None]
            # Multiply by 2 because estimator divides by (N * (N - 1)) for ((N * (N - 1)) / 2) parameters, so it is equivalent to taking the mean divided by 2
            cart_prod_weights = 2 * np.sum(np.triu(cart_prod_weights, 1))

        res[i] = cart_prod / cart_prod_weights

    return res

class BorovickovaShimerEstimator():
    '''
    Class for estimating the non-parametric sorting model from Borovickova and Shimer.
    '''

    def __init__(self):
        self.res = None

    def fit(self, adata, weighted=True):
        '''
        Estimate the non-parametric sorting model from Borovickova and Shimer.

        Arguments:
            adata (BipartiteDataFrame): long or collapsed long format labor data
            weighted (bool): if True, run estimator with weights. These come from data column 'w'.
        '''
        if not adata._col_included('w'):
            # Skip weighting if no weight column included
            weighted = False

        ### Worker estimates ###
        if weighted:
            # If weighted, convert y to weighted y and compute sqrt(weights)
            adata.loc[:, 'unweighted_y'] = adata.loc[:, 'y']
            adata.loc[:, 'y'] = adata.loc[:, 'w'].to_numpy() * adata.loc[:, 'unweighted_y'].to_numpy()
            adata.loc[:, 'sqrt_w'] = np.sqrt(adata.loc[:, 'w'].to_numpy())

        ## Worker mean ##
        groupby_i = adata.groupby('i', sort=False)
        if weighted:
            weights_i = groupby_i['w'].sum().to_numpy()
            lambda_i = groupby_i['y'].sum().to_numpy() / weights_i
        else:
            lambda_i = groupby_i['y'].mean().to_numpy()

        ## Worker mean squared ##
        if weighted:
            # If weighted, convert y to weighted y
            weights = adata.loc[:, 'sqrt_w'].to_numpy()
            weighted_y = weights * adata.loc[:, 'unweighted_y'].to_numpy()
        else:
            weights = None
            weighted_y = adata.loc[:, 'y'].to_numpy()

        ## w_bar ##
        if weighted:
            w_bar = np.sum(weights_i * lambda_i) / np.sum(weights_i)
        else:
            w_bar = np.mean(lambda_i)

        ## sigma_sq_lambda ##
        lambda_sq_i = _compute_mean_sq(adata.loc[:, 'i'].to_numpy(), weighted_y, weights=weights)
        if weighted:
            sigma_sq_lambda = np.sum(weights_i * lambda_sq_i) / np.sum(weights_i) - (w_bar ** 2)
        else:
            sigma_sq_lambda = np.mean(lambda_sq_i) - (w_bar ** 2)

        ## Covariance (worker component) ##
        y_i = groupby_i['y'].transform('sum').to_numpy()
        if weighted:
            weights_i = groupby_i['w'].transform('sum').to_numpy()
            c_i = (y_i - adata.loc[:, 'y'].to_numpy()) / (weights_i - adata.loc[:, 'w'].to_numpy())
        else:
            weights_i = groupby_i['y'].transform('size').to_numpy()
            c_i = (y_i - adata.loc[:, 'y'].to_numpy()) / (weights_i - 1)

        ### Firm estimates ###
        adata.sort_values('j', axis=0, inplace=True)

        ## Firm mean ##
        groupby_j = adata.groupby('j', sort=False)
        if weighted:
            weights_j = groupby_j['w'].sum().to_numpy()
            mu_j = groupby_j['y'].sum().to_numpy() / weights_j
        else:
            mu_j = groupby_j['y'].mean()

        ## Firm mean squared ##
        if weighted:
            # If weighted, convert y to weighted y
            weights = adata.loc[:, 'sqrt_w'].to_numpy()
            weighted_y = weights * adata.loc[:, 'unweighted_y'].to_numpy()
        else:
            weights = None
            weighted_y = adata.loc[:, 'y'].to_numpy()

        ## sigma_sq_mu ##
        mu_sq_j = _compute_mean_sq(adata.loc[:, 'j'].to_numpy(), weighted_y, weights=weights)
        if weighted:
            sigma_sq_mu = np.sum(weights_j * mu_sq_j) / np.sum(weights_j) - (w_bar ** 2)
        else:
            sigma_sq_mu = np.mean(mu_sq_j) - (w_bar ** 2)

        ## Covariance (firm component) ##
        y_j = groupby_j['y'].transform('sum').to_numpy()
        if weighted:
            weights_j = groupby_j['w'].transform('sum').to_numpy()
            c_j = (y_j - adata.loc[:, 'y'].to_numpy()) / (weights_j - adata.loc[:, 'w'].to_numpy())
        else:
            weights_j = groupby_j['y'].transform('size').to_numpy()
            c_j = (y_j - adata.loc[:, 'y'].to_numpy()) / (weights_j - 1)

        ### Covariance ###
        c_i_j = c_i * c_j
        if weighted:
            cov = np.sum(adata.loc[:, 'w'].to_numpy() * c_i_j) / np.sum(adata.loc[:, 'w'].to_numpy()) - (w_bar ** 2)
        else:
            cov = np.mean(c_i_j) - (w_bar ** 2)

        ### Restore values ###
        if weighted:
            # Restore y
            adata.loc[:, 'y'] = adata.loc[:, 'unweighted_y']
            adata.drop(['unweighted_y', 'sqrt_w'], axis=1, inplace=True)

        # Sort
        adata.sort_rows(copy=False)
        
        # Store results
        self.res = {
            'w_bar': w_bar,
            'sigma_sq_lambda': sigma_sq_lambda,
            'sigma_sq_mu': sigma_sq_mu,
            'cov(lambda, mu)': cov,
            'corr(lambda, mu)': cov / np.sqrt(sigma_sq_lambda * sigma_sq_mu)
        }
