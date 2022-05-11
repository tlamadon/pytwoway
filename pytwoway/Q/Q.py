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
        Dp = diags(Dp.data[0][keep_rows])

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
        Dp = diags(Dp.data[0][keep_rows])

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
