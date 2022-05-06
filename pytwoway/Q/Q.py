'''
Classes for constructing the variance-covariance matrix for AKM and its bias corrections. Note: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).

TODO: cov(psi_i, psi_j), var(gamma)
'''
import numpy as np
from scipy.sparse import csc_matrix, diags
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
        Ql = csc_matrix((np.ones(n_rows), (np.arange(n_rows).astype(int, copy=False), adata.loc[keep_rows, 'j'].to_numpy())), shape=(n_rows, nf))

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
        Qr = csc_matrix((np.ones(n_rows), (np.arange(n_rows).astype(int, copy=False), adata.loc[keep_rows, 'j'].to_numpy())), shape=(n_rows, nf))

        # Normalize one firm to 0
        Qr = Qr[:, range(nf - 1)]

        # Update weights
        Dp = diags(Dp.data[0][keep_rows])

        return (Qr, 'psi', Dp)
