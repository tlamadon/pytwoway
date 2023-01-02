'''
Classes for constructing the variance-covariance matrix for AKM and its bias corrections.
'''
'''
NOTE: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).
TODO:
    -cov(psi_i, psi_j)
'''
import numpy as np
from scipy.sparse import csc_matrix, hstack
from bipartitepandas.util import to_list, fast_shift

###############################
##### FE without controls #####
###############################

class VarPsi():
    '''
    Generate Q to estimate var(psi).

    Arguments:
        category (NumPy Array or None): categorical vector of 0s and 1s; None is equivalent to all 1s
        category_name (str or None): category 1 string representation; if None, just print 'psi'
    '''

    def __init__(self, category=None, category_name=None):
        self.category = category
        self.category_name = category_name

    def name(self):
        '''
        Return string representation of var(psi).

        Returns:
            (str): string representation of var(psi)
        '''
        str_name = 'var(psi'
        if self.category_name is not None:
            str_name += f'-{self.category_name}'
        str_name += ')'
        return str_name

    def _get_Q(self, adata, J, W, Dp):
        '''
        Construct Q matrix to use when estimating var(psi).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (half of Q matrix, weights, psi/alpha) --> (J, Dp, 'psi')
        '''
        if self.category is None:
            return (J, Dp, 'psi')
        return (J[self.category, :], Dp[self.category], 'psi')

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

    Arguments:
        category (NumPy Array or None): categorical vector of 0s and 1s; None is equivalent to all 1s
        category_name (str or None): category 1 string representation; if None, just print 'alpha'
    '''

    def __init__(self, category=None, category_name=None):
        self.category = category
        self.category_name = category_name

    def name(self):
        '''
        Return string representation of var(alpha).

        Returns:
            (str): string representation of var(alpha)
        '''
        str_name = 'var(alpha'
        if self.category_name is not None:
            str_name += f'-{self.category_name}'
        str_name += ')'
        return str_name

    def _get_Q(self, adata, J, W, Dp):
        '''
        Construct Q matrix to use when estimating var(alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (half of Q matrix, weights, psi/alpha) --> (W, Dp, 'alpha')
        '''
        if self.category is None:
            return (W, Dp, 'alpha')
        return (W[self.category, :], Dp[self.category], 'alpha')

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

class VarPsiPlusAlpha():
    '''
    Generate Q to estimate var(psi + alpha).

    Arguments:
        category (NumPy Array or None): categorical vector of 0s and 1s; None is equivalent to all 1s
        category_name (str or None): category 1 string representation; if None, just print 'psi + alpha'
    '''

    def __init__(self, category=None, category_name=None):
        self.category = category
        self.category_name = category_name

    def name(self):
        '''
        Return string representation of var(psi + alpha).

        Returns:
            (str): string representation of var(psi + alpha)
        '''
        str_name = 'var(psi + alpha'
        if self.category_name is not None:
            str_name += f'-{self.category_name}'
        str_name += ')'
        return str_name

    def _get_Q(self, adata, J, W, Dp):
        '''
        Construct Q matrix to use when estimating var(psi + alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (half of Q matrix, weights, psi/alpha) --> (hstack([J, W]), Dp, 'gamma')
        '''
        if self.category is None:
            return (hstack([J, W]), Dp, 'gamma')
        return (hstack([J, W])[self.category, :], Dp[self.category], 'gamma')

    def _Q_mult(self, Q_matrix, psi, alpha):
        '''
        Multiply Q matrix by vectors related to psi and alpha to use when estimating var(psi + alpha).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            psi (NumPy Array): vector related to psi
            alpha (NumPy Array): vector related to alpha

        Returns:
            (NumPy Array): Q_matrix @ np.concatenate((psi, alpha))
        '''
        return Q_matrix @ np.concatenate((psi, alpha))

class CovPsiAlpha():
    '''
    Generate Q to estimate cov(psi, alpha).

    Arguments:
        category1 (NumPy Array or None): categorical vector of 0s and 1s for left covariance term; None is equivalent to all 1s
        category2 (NumPy Array or None): categorical vector of 0s and 1s for right covariance term; None is equivalent to all 1s
        category1_name (str or None): category 1 string representation; if None, just print 'psi, alpha'
        category2_name (str or None): category 2 string representation; if None, just print 'psi, alpha'
    '''

    def __init__(self, category1=None, category2=None, category1_name=None, category2_name=None):
        self.category1 = category1
        self.category2 = category2
        self.category1_name = category1_name
        self.category2_name = category2_name

    def name(self):
        '''
        Return string representation of cov(psi, alpha).

        Returns:
            (str): string representation of cov(psi, alpha)
        '''
        str_name = 'cov(psi'
        if self.category1_name is not None:
            str_name += f'-{self.category1_name}'
        str_name += ', alpha'
        if self.category2_name is not None:
            str_name += f'-{self.category2_name}'
        str_name += ')'
        return str_name

    def _get_Ql(self, adata, J, W, Dp):
        '''
        Construct Ql matrix (Q-left) to use when estimating cov(psi, alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (left term of Q matrix, weights, psi/alpha) --> (J, Dp, 'psi')
        '''
        if self.category1 is None:
            return (J, Dp, 'psi')
        return (J[self.category1, :], Dp[self.category1], 'psi')

    def _get_Qr(self, adata, J, W, Dp):
        '''
        Construct Qr matrix (Q-right) to use when estimating cov(psi, alpha).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (right term of Q matrix, weights, psi/alpha) --> (W, Dp, 'alpha')
        '''
        if self.category2 is None:
            return (W, Dp, 'alpha')
        return (W[self.category2, :], Dp[self.category2], 'alpha')

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

# class CovAlphaAlpha():
#     '''
#     Generate Q to estimate cov(alpha, alpha).

#     Arguments:
#         category1 (NumPy Array or None): categorical vector of 0s and 1s for left covariance term; None is equivalent to all 1s
#         category2 (NumPy Array or None): categorical vector of 0s and 1s for right covariance term; None is equivalent to all 1s
#         category1_name (str or None): category 1 string representation; if None, just print 'alpha'
#         category2_name (str or None): category 2 string representation; if None, just print 'alpha'
#     '''

#     def __init__(self, category1=None, category2=None, category1_name=None, category2_name=None):
#         self.category1 = category1
#         self.category2 = category2
#         self.category1_name = category1_name
#         self.category2_name = category2_name

#     def name(self):
#         '''
#         Return string representation of cov(alpha, alpha).

#         Returns:
#             (str): string representation of cov(alpha, alpha)
#         '''
#         str_name = 'cov(alpha'
#         if self.category1_name is not None:
#             str_name += f'-{self.category1_name}'
#         str_name += ', alpha'
#         if self.category2_name is not None:
#             str_name += f'-{self.category2_name}'
#         str_name += ')'
#         return str_name

#     def _get_Ql(self, adata, J, W, Dp):
#         '''
#         Construct Ql matrix (Q-left) to use when estimating cov(alpha, alpha).

#         Arguments:
#             adata (BipartiteDataFrame): data
#             J (CSC Sparse Matrix): firm ids matrix representation
#             W (CSC Sparse Matrix): worker ids matrix representation
#             Dp (NumPy Array): weights

#         Returns:
#             (tuple): (left term of Q matrix, weights, psi/alpha) --> (W, Dp, 'alpha')
#         '''
#         if self.category1 is None:
#             return (W, Dp, 'alpha')
#         return (W[self.category1, :], Dp[self.category1], 'alpha')

#     def _get_Qr(self, adata, J, W, Dp):
#         '''
#         Construct Qr matrix (Q-right) to use when estimating cov(alpha, alpha).

#         Arguments:
#             adata (BipartiteDataFrame): data
#             J (CSC Sparse Matrix): firm ids matrix representation
#             W (CSC Sparse Matrix): worker ids matrix representation
#             Dp (NumPy Array): weights

#         Returns:
#             (tuple): (right term of Q matrix, weights, psi/alpha) --> (W, Dp, 'alpha')
#         '''
#         if self.category2 is None:
#             return (W, Dp, 'alpha')
#         return (W[self.category2, :], Dp[self.category2], 'alpha')

#     def _Ql_mult(self, Q_matrix, psi, alpha):
#         '''
#         Multiply Ql matrix by vectors related to psi and alpha to use when estimating cov(alpha, alpha).

#         Arguments:
#             Q_matrix (NumPy Array): Q matrix
#             psi (NumPy Array): vector related to psi
#             alpha (NumPy Array): vector related to alpha

#         Returns:
#             (NumPy Array): Q_matrix @ alpha
#         '''
#         return Q_matrix @ alpha

#     def _Qr_mult(self, Q_matrix, psi, alpha):
#         '''
#         Multiply Qr matrix by vectors related to psi and alpha to use when estimating cov(alpha, alpha).

#         Arguments:
#             Q_matrix (NumPy Array): Q matrix
#             psi (NumPy Array): vector related to psi
#             alpha (NumPy Array): vector related to alpha

#         Returns:
#             (NumPy Array): Q_matrix @ alpha
#         '''
#         return Q_matrix @ alpha

class CovPsiPrevPsiNext():
    '''
    Generate Q to estimate cov(psi_t, psi_{t+1}).

    Arguments:
        category1 (NumPy Array or None): categorical vector of 0s and 1s for left covariance term; None is equivalent to all 1s
        category2 (NumPy Array or None): categorical vector of 0s and 1s for right covariance term; None is equivalent to all 1s
        category1_name (str or None): category 1 string representation; if None, just print 'alpha'
        category2_name (str or None): category 2 string representation; if None, just print 'alpha'
    '''

    def __init__(self, category1=None, category2=None, category1_name=None, category2_name=None):
        self.category1 = category1
        self.category2 = category2
        self.category1_name = category1_name
        self.category2_name = category2_name

    def name(self):
        '''
        Return string representation of cov(psi_t, psi_{t+1}).

        Returns:
            (str): string representation of cov(psi_t, psi_{t+1})
        '''
        str_name = 'cov(psi_t'
        if self.category1_name is not None:
            str_name += f'-{self.category1_name}'
        str_name += ', psi_{t+1}'
        if self.category2_name is not None:
            str_name += f'-{self.category2_name}'
        str_name += ')'
        return str_name

    def _get_Ql(self, adata, J, W, Dp):
        '''
        Construct Ql matrix (Q-left) to use when estimating cov(psi_t, psi_{t+1}).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (left term of Q matrix, weights, psi/alpha) --> (Ql, Dp, 'psi')
        '''
        # Get number of firms
        nf = J.shape[1] + 1
        # Get i for this and next period
        i_col = adata.loc[:, 'i'].to_numpy()
        i_next = fast_shift(i_col, -1, fill_value=-2)
        # Drop the last observation for each worker
        keep_rows = (i_col == i_next)
        n_rows = np.sum(keep_rows)
        # Construct Ql
        Ql = csc_matrix((np.ones(n_rows), (np.arange(n_rows, dtype=int), adata.loc[keep_rows, 'j'].to_numpy())), shape=(n_rows, nf))

        # Normalize one firm to 0
        Ql = Ql[:, 1:]

        # Update weights
        if not isinstance(Dp, (float, int)):
            Dp = Dp[keep_rows]

        if self.category1 is None:
            return (Ql, Dp, 'psi')
        return (Ql[self.category1[keep_rows]], Dp[self.category1[keep_rows]], 'psi')

    def _get_Qr(self, adata, J, W, Dp):
        '''
        Construct Qr matrix (Q-right) to use when estimating cov(psi_t, psi_{t+1}).

        Arguments:
            adata (BipartiteDataFrame): data
            J (CSC Sparse Matrix): firm ids matrix representation
            W (CSC Sparse Matrix): worker ids matrix representation
            Dp (NumPy Array): weights

        Returns:
            (tuple): (right term of Q matrix, weights, psi/alpha) --> (Qr, Dp, 'psi')
        '''
        # Get number of firms
        nf = J.shape[1] + 1
        # Get i for this and last period
        i_col = adata.loc[:, 'i'].to_numpy()
        i_prev = fast_shift(i_col, 1, fill_value=-2)
        # Drop the first observation for each worker
        keep_rows = (i_col == i_prev)
        n_rows = np.sum(keep_rows)
        # Construct Qr
        Qr = csc_matrix((np.ones(n_rows), (np.arange(n_rows, dtype=int), adata.loc[keep_rows, 'j'].to_numpy())), shape=(n_rows, nf))

        # Normalize one firm to 0
        Qr = Qr[:, 1:]

        # Update weights
        if not isinstance(Dp, (float, int)):
            Dp = Dp[keep_rows]

        if self.category2 is None:
            return (Qr, Dp, 'psi')
        return (Qr[self.category2[keep_rows], :], Dp[self.category2[keep_rows]], 'psi')

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
    Generate Q to estimate var(covariate) or var(sum(covariates)) if multiple covariates are listed.

    Arguments:
        cov_names (str or list of str): covariate name or list of covariate names to sum
    '''

    def __init__(self, cov_names):
        cov_names = to_list(cov_names)
        if len(cov_names) != len(set(cov_names)):
            raise ValueError('Cannot include a covariate name multiple times.')

        self.cov_names = cov_names

    def name(self):
        '''
        Return string representation of variance to be estimated.

        Returns:
            (str): string representation of variance to be estimated
        '''
        var_str = f'{self.cov_names[0]}'
        for cov_name in self.cov_names[1:]:
            var_str += f' + {cov_name}'

        return f'var({var_str})'

    def _get_Q(self, adata, A, Dp, cov_indices):
        '''
        Construct Q matrix to use when estimating var(covariate).

        Arguments:
            adata (BipartiteDataFrame): data
            A (CSC Sparse Matrix): matrix of covariates
            Dp (NumPy Array): weights
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (tuple): (half of Q matrix, weights) --> (A[covariate], Dp)
        '''
        idx_start, idx_end = cov_indices[self.cov_names[0]]
        A_indices = np.arange(idx_start, idx_end)

        for cov_name in self.cov_names[1:]:
            idx_start, idx_end = cov_indices[cov_name]
            A_indices_2 = np.arange(idx_start, idx_end)
            A_indices = np.concatenate([A_indices, A_indices_2])

        return (A[:, A_indices], Dp)

    def _Q_mult(self, Q_matrix, Z, cov_indices):
        '''
        Multiply Q matrix by component of Z related to covariate to use when estimating var(covariate).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            Z (NumPy Array): vector to multiply
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (NumPy Array): Q_matrix @ Z_covariate
        '''
        idx_start, idx_end = cov_indices[self.cov_names[0]]
        sub_Z = Z[idx_start: idx_end]
        if isinstance(sub_Z, (float, int)):
            # Continuous covariate
            sub_Z = np.array([sub_Z])

        for cov_name in self.cov_names[1:]:
            idx_start, idx_end = cov_indices[cov_name]
            sub_Z_2 = Z[idx_start: idx_end]
            if isinstance(sub_Z_2, (float, int)):
                # Continuous covariate
                sub_Z_2 = np.array([sub_Z_2])

            # Concatenate sub_Z and sub_Z_2
            sub_Z = np.concatenate([sub_Z, sub_Z_2])

        return Q_matrix @ sub_Z

class CovCovariate():
    '''
    Generate Q to estimate cov(covariate 1, covariate 2).

    Arguments:
        cov_names_1 (str or list of str): covariate name or list of covariate names to sum for first term in covariance
        cov_names_2 (str or list of str): covariate name or list of covariate names to sum for second term in covariance
    '''

    def __init__(self, cov_names_1, cov_names_2):
        cov_names_1 = to_list(cov_names_1)
        cov_names_2 = to_list(cov_names_2)

        if (len(cov_names_1) != len(set(cov_names_1))) or (len(cov_names_2) != len(set(cov_names_2))):
            raise ValueError('Cannot include a covariate name multiple times.')

        self.cov_names_1 = cov_names_1
        self.cov_names_2 = cov_names_2

    def name(self):
        '''
        Return string representation of covariance to be estimated.

        Returns:
            (str): string representation of covariance to be estimated
        '''
        cov_str_1 = f'{self.cov_names_1[0]}'
        for cov_name_1 in self.cov_names_1[1:]:
            cov_str_1 += f' + {cov_name_1}'

        cov_str_2 = f'{self.cov_names_2[0]}'
        for cov_name_2 in self.cov_names_2[1:]:
            cov_str_2 += f' + {cov_name_2}'

        return f'cov({cov_str_1}, {cov_str_2})'

    def _get_Ql(self, adata, A, Dp, cov_indices):
        '''
        Construct Ql matrix (Q-left) to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            adata (BipartiteDataFrame): data
            A (CSC Sparse Matrix): matrix of covariates
            Dp (NumPy Array): weights
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (tuple): (left term of Q matrix, weights) --> (A[covariate 1], Dp)
        '''
        idx_start, idx_end = cov_indices[self.cov_names_1[0]]
        A_indices = np.arange(idx_start, idx_end)

        for cov_name in self.cov_names_1[1:]:
            idx_start, idx_end = cov_indices[cov_name]
            A_indices_2 = np.arange(idx_start, idx_end)
            A_indices = np.concatenate([A_indices, A_indices_2])

        return (A[:, A_indices], Dp)

    def _get_Qr(self, adata, A, Dp, cov_indices):
        '''
        Construct Qr matrix (Q-right) to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            adata (BipartiteDataFrame): data
            A (CSC Sparse Matrix): matrix of covariates
            Dp (NumPy Array): weights
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (tuple): (right term of Q matrix, weights) --> (A[covariate 2], Dp)
        '''
        idx_start, idx_end = cov_indices[self.cov_names_2[0]]
        A_indices = np.arange(idx_start, idx_end)

        for cov_name in self.cov_names_2[1:]:
            idx_start, idx_end = cov_indices[cov_name]
            A_indices_2 = np.arange(idx_start, idx_end)
            A_indices = np.concatenate([A_indices, A_indices_2])

        return (A[:, A_indices], Dp)

    def _Ql_mult(self, Q_matrix, Z, cov_indices):
        '''
        Multiply Ql matrix by component of Z related to covariate 1 to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            Z (NumPy Array): vector to multiply
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (NumPy Array): Q_matrix @ Z_covariate_1
        '''
        idx_start, idx_end = cov_indices[self.cov_names_1[0]]
        sub_Z = Z[idx_start: idx_end]
        if isinstance(sub_Z, (float, int)):
            # Continuous covariate
            sub_Z = np.array([sub_Z])

        for cov_name in self.cov_names_1[1:]:
            idx_start, idx_end = cov_indices[cov_name]
            sub_Z_2 = Z[idx_start: idx_end]
            if isinstance(sub_Z_2, (float, int)):
                # Continuous covariate
                sub_Z_2 = np.array([sub_Z_2])

            # Concatenate sub_Z and sub_Z_2
            sub_Z = np.concatenate([sub_Z, sub_Z_2])

        return Q_matrix @ sub_Z

    def _Qr_mult(self, Q_matrix, Z, cov_indices):
        '''
        Multiply Qr matrix by component of Z related to covariate 2 to use when estimating cov(covariate 1, covariate 2).

        Arguments:
            Q_matrix (NumPy Array): Q matrix
            Z (NumPy Array): vector to multiply
            cov_indices (dict of tuples of ints): dictionary linking each covariate to the range of columns in A where it is stored

        Returns:
            (NumPy Array): Q_matrix @ Z_covariate_2
        '''
        idx_start, idx_end = cov_indices[self.cov_names_2[0]]
        sub_Z = Z[idx_start: idx_end]
        if isinstance(sub_Z, (float, int)):
            # Continuous covariate
            sub_Z = np.array([sub_Z])

        for cov_name in self.cov_names_2[1:]:
            idx_start, idx_end = cov_indices[cov_name]
            sub_Z_2 = Z[idx_start: idx_end]
            if isinstance(sub_Z_2, (float, int)):
                # Continuous covariate
                sub_Z_2 = np.array([sub_Z_2])

            # Concatenate sub_Z and sub_Z_2
            sub_Z = np.concatenate([sub_Z, sub_Z_2])

        return Q_matrix @ sub_Z
