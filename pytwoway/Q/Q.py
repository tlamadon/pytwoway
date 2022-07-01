'''
Classes for constructing the variance-covariance matrix for AKM and its bias corrections.
'''
'''
NOTE: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).
TODO:
    -cov(psi_i, psi_j)
'''
import numpy as np
from scipy.sparse import csc_matrix, diags, hstack
from bipartitepandas.util import fast_shift

###############################
##### FE without controls #####
###############################

class VarPsi():
    '''
    Generate Q to estimate var(psi).
    '''

    def __init__(self):
        pass

    def _get_Q(self, adata, nf, nw, J, W, Dp):
        '''
        Construct Q matrix to use when estimating var(psi).

        Arguments:
            adata (BipartiteDataFrame): data
            nf (int): number of firm types
            nw (int): number of worker types
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (half of Q matrix, psi/alpha, weights, degrees of freedom) --> (J, 'psi', Dp, nf - 1)
        '''
        return (J, 'psi', Dp, nf - 1)

    def _Q_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Q matrix by vectors related to psi and alpha to use when estimating var(psi).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ psi
        '''
        return Q_matrix @ psi

class VarAlpha():
    '''
    Generate Q to estimate var(alpha).
    '''

    def __init__(self):
        pass

    def _get_Q(self, adata, nf, nw, J, W, Dp):
        '''
        Construct Q matrix to use when estimating var(alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            nf (int): number of firm types
            nw (int): number of worker types
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (half of Q matrix, psi/alpha, weights, degrees of freedom) --> (W, 'alpha', Dp, nw)
        '''
        return (W, 'alpha', Dp, nw)

    def _Q_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Q matrix by vectors related to psi and alpha to use when estimating var(alpha).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ alpha
        '''
        return Q_matrix @ alpha

# NOTE: var(gamma) is effected by normalization, i.e. it isn't identified
# class VarGamma():
#     '''
#     Generate Q to estimate var(gamma), where gamma = [psi.T alpha.T].T.
#     '''

#     def __init__(self):
#         pass

#     def _get_Q(self, adata, nf, nw, J, W, Dp):
#         '''
#         Construct Q matrix to use when estimating var(gamma).

#         Arguments:
#             adata (BipartiteDataFrame): data
#             nf (int): number of firm types
#             nw (int): number of worker types
#             J (CSC Sparse Matrix): firm ids matrix representation
#             W (CSC Sparse Matrix): worker ids matrix representation
#             Dp (NumPy Array): weights

#         Returns:
#             (tuple): (half of Q matrix, psi/alpha, weights, degrees of freedom) --> (hstack((J, W)), 'gamma', Dp, nw)
#         '''
#         return (hstack((J, W)), 'gamma', Dp, nf + nw)

#     def _Q_mult(self, Q_matrix, psi, alpha):
#         '''
#         Multiply Q matrix by vectors related to psi and alpha to use when estimating var(gamma).

#         Arguments:
#             Q_matrix (NumPy Array): Q matrix
#             psi (NumPy Array): vector related to psi
#             alpha (NumPy Array): vector related to alpha

#         Returns:
#             (NumPy Array): Q_matrix @ np.concatenate((psi, alpha))
#         '''
#         return Q_matrix @ np.concatenate((psi, alpha))

class CovPsiAlpha():
    '''
    Generate Q to estimate cov(psi, alpha).
    '''

    def __init__(self):
        pass

    def _get_Ql(self, adata, nf, nw, J, W, Dp):
        '''
        Construct Ql matrix (Q-left) to use when estimating cov(psi, alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            nf (int): number of firm types
            nw (int): number of worker types
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (left term of Q matrix, psi/alpha, weights, degrees of freedom) --> (J, 'psi', Dp, (nf - 1) + nw)
        '''
        return (J, 'psi', Dp, (nf - 1) + nw)

    def _get_Qr(self, adata, nf, nw, J, W, Dp):
        '''
        Construct Qr matrix (Q-right) to use when estimating cov(psi, alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            nf (int): number of firm types
            nw (int): number of worker types
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (right term of Q matrix, psi/alpha, weights) --> (W, 'alpha', Dp)
        '''
        return (W, 'alpha', Dp)

    def _Ql_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Ql matrix by vectors related to psi and alpha to use when estimating cov(psi, alpha).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ psi
        '''
        return Q_matrix @ psi

    def _Qr_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Qr matrix by vectors related to psi and alpha to use when estimating cov(psi, alpha).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ alpha
        '''
        return Q_matrix @ alpha

class CovPsiPrevPsiNext():
    '''
    Generate Q to estimate cov(psi_t, psi_{t+1}).
    '''

    def __init__(self):
        pass

    def _get_Ql(self, adata, nf, nw, J, W, Dp):
        '''
        Construct Ql matrix (Q-left) to use when estimating cov(psi_t, psi_{t+1}).

        Arguments:
            adata (BipartiteDataFrame): data
            nf (int): number of firm types
            nw (int): number of worker types
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (left term of Q matrix, psi/alpha, weights, degrees of freedom) --> (Ql, 'psi', Dp, nf - 1)
        '''
        # Get i for this and next period
        i_col = adata.loc[:, 'i'].to_numpy()
        i_next = fast_shift(i_col, -1, fill_value=-2)
        # Drop the last observation for each worker
        keep_rows = (i_col == i_next)
        n_rows = np.sum(keep_rows)
        # Construct Ql
        Ql = csc_matrix((np.ones(n_rows), (np.arange(n_rows, dtype=int), adata.loc[keep_rows, 'j'].to_numpy())), shape=(n_rows, nf))

        # Normalize one firm to 0
        Ql = Ql[:, range(nf - 1)]

        # Update weights
        if not isinstance(Dp, (float, int)):
            Dp = Dp[keep_rows]

        return (Ql, 'psi', Dp, nf - 1)

    def _get_Qr(self, adata, nf, nw, J, W, Dp):
        '''
        Construct Qr matrix (Q-right) to use when estimating cov(psi_t, psi_{t+1}).

        Arguments:
            adata (BipartiteDataFrame): data
            nf (int): number of firm types
            nw (int): number of worker types
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (right term of Q matrix, psi/alpha, weights) --> (Qr, 'psi', Dp)
        '''
        # Get i for this and last period
        i_col = adata.loc[:, 'i'].to_numpy()
        i_prev = fast_shift(i_col, 1, fill_value=-2)
        # Drop the first observation for each worker
        keep_rows = (i_col == i_prev)
        n_rows = np.sum(keep_rows)
        # Construct Qr
        Qr = csc_matrix((np.ones(n_rows), (np.arange(n_rows, dtype=int), adata.loc[keep_rows, 'j'].to_numpy())), shape=(n_rows, nf))

        # Normalize one firm to 0
        Qr = Qr[:, range(nf - 1)]

        # Update weights
        if not isinstance(Dp, (float, int)):
            Dp = Dp[keep_rows]

        return (Qr, 'psi', Dp)

    def _Ql_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Q matrix by vectors related to psi and alpha to use when estimating cov(psi_t, psi_{t+1}).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ psi
        '''
        return Q_matrix @ psi

    def _Qr_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Q matrix by vectors related to psi and alpha to use when estimating cov(psi_t, psi_{t+1}).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ psi
        '''
        return Q_matrix @ psi

############################
##### FE with controls #####
############################

class VarCovariate():
    '''
    Generate Q to estimate var(covariate).

    Arguments:
        cov_name (str): covariate name
    '''

    def __init__(self, cov_name):
        self.cov_name = cov_name

    def name(self):
        '''
        Return name of variance to be estimated.

        Returns:
            (str): var_{self.cov_name}
        '''
        return f'var_{self.cov_name}'

    def _get_Q(self, adata, A, Dp, cov_indices):
        '''
        Construct Q matrix to use when estimating var(covariate).

        Arguments:
            adata (BipartiteDataFrame): data
            A (CSC Sparse Matrix): matrix of covariates
            Dp (NumPy Array): weights
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (tuple): (half of Q matrix, cov_name, weights, degrees of freedom) --> (A[covariate], covariate name, Dp, n_covariate - 1)
        '''
        idx_start, idx_end = cov_indices[self.cov_name]
        return (A[:, idx_start: idx_end], self.cov_name, Dp, (idx_start - idx_end) - 1)

    def _Q_mult(self, Q_matrix, gamma_hat_dict):
        '''
        Multiply Q matrix by component of gamma vector related to covariate to use when estimating var(covariate).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            gamma_hat_dict (dict of NumPy Arrays): dictionary linking each covariate to its component of gamma_hat

        Returns:
            (NumPy Array): Q_matrix @ gamma_hat_dict[covariate]
        '''
        gamma_hat = gamma_hat_dict[self.cov_name]
        if isinstance(gamma_hat, (float, int)):
            # Continuous
            gamma_hat = np.array([gamma_hat])

        return Q_matrix @ gamma_hat

class CovCovariate():
    '''
    Generate Q to estimate cov(covariate 1, covariate 2).

    Arguments:
        cov_name_1 (str): first covariate name
        cov_name_2 (str): second covariate name
    '''

    def __init__(self, cov_name_1, cov_name_2):
        if cov_name_1 == cov_name_2:
            raise ValueError('cov_name_1 and cov_name_2 must be different.')
        self.cov_name_1 = cov_name_1
        self.cov_name_2 = cov_name_2

    def name(self):
        '''
        Return name of covariance to be estimated.

        Returns:
            (str): cov_{self.cov_name_1}_{self.cov_name_2}
        '''
        return f'cov_{self.cov_name_1}_{self.cov_name_2}'

    def _get_Ql(self, adata, A, Dp, cov_indices):
        '''
        Construct Ql matrix (Q-left) to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            adata (BipartiteDataFrame): data
            A (CSC Sparse Matrix): matrix of covariates
            Dp (NumPy Array): weights
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (tuple): (left term of Q matrix, cov_name_1, weights, degrees of freedom) --> (A[covariate 1], covariate 1 name, Dp, n_control_variable_1 - 1)
        '''
        idx_start, idx_end = cov_indices[self.cov_name_1]
        return (A[:, idx_start: idx_end], self.cov_name_1, Dp, (idx_start - idx_end) - 1)

    def _get_Qr(self, adata, A, Dp, cov_indices):
        '''
        Construct Qr matrix (Q-right) to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            adata (BipartiteDataFrame): data
            A (CSC Sparse Matrix): matrix of covariates
            Dp (NumPy Array): weights
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (tuple): (right term of Q matrix, cov_name_2, weights, degrees of freedom) --> (A[covariate 2], covariate 2 name, Dp, n_control_variable_2 - 1)
        '''
        idx_start, idx_end = cov_indices[self.cov_name_2]
        return (A[:, idx_start: idx_end], self.cov_name_2, Dp, (idx_start - idx_end) - 1)

    def _Ql_mult(self, Q_matrix, gamma_hat_dict):
        '''
        Multiply Ql matrix by component of gamma vector related to covariate 1 to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            gamma_hat_dict (dict of NumPy Arrays): dictionary linking each covariate to its component of gamma_hat

        Returns:
            (NumPy Array): Q_matrix @ gamma_hat_dict[covariate 1]
        '''
        gamma_hat = gamma_hat_dict[self.cov_name_1]
        if isinstance(gamma_hat, (float, int)):
            # Continuous
            gamma_hat = np.array([gamma_hat])

        return Q_matrix @ gamma_hat

    def _Qr_mult(self, Q_matrix, gamma_hat_dict):
        '''
        Multiply Qr matrix by component of gamma vector related to covariate 2 to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            gamma_hat_dict (dict of NumPy Arrays): dictionary linking each covariate to its component of gamma_hat

        Returns:
            (NumPy Array): Q_matrix @ gamma_hat_dict[covariate 2]
        '''
        gamma_hat = gamma_hat_dict[self.cov_name_2]
        if isinstance(gamma_hat, (float, int)):
            # Continuous
            gamma_hat = np.array([gamma_hat])

        return Q_matrix @ gamma_hat
