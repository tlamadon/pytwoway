'''
Estimate the interacted model from Bonhomme, Lamadon, & Manresa.
'''
from tqdm.auto import tqdm, trange
import numpy as np
from scipy.sparse import csc_matrix, eye, hstack, tril
from scipy.sparse.linalg import eigs, inv
from qpsolvers import solve_qp
from pyamg import ruge_stuben_solver as rss
import bipartitepandas as bpd
from bipartitepandas.util import ParamsDict, ChainedAssignment
from pytwoway.util import weighted_mean, DxSP, SPxD, diag_of_sp_prod

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq0(a):
    return a >= 0
# def _gteq1(a):
#     return a >= 1

# Define default parameter dictionary
_iblm_params_default = ParamsDict({
    ## All ##
    'estimator': ('linear', 'set', ['linear', 'fixed_point'],
        '''
            (default='linear') If 'linear', use linear estimator; if 'fixed_point', use fixed-point estimator.
        ''', None),
    'weighted': (True, 'type', bool,
        '''
            (default=True) If True, use weighted estimators.
        ''', None),
    'norm_fid': (0, 'type_constrained', (int, _gteq0),
        '''
            (default=0) Firm id to normalize.
        ''', '>= 0'),
    'progress_bars': (False, 'type', bool,
        '''
            (default=False) If True, display progress bars.
        ''', None),
    ## Linear ##
    'method': ('liml_1', 'set', ['liml_1', 'liml_2', 'iv', 'hful', 'linear'],
        '''
            (default='liml_1') (For linear estimator) If 'liml_1', use LIML that allows profiling; if 'liml_2', use traditional LIML that does not allow profiling; if 'iv', use IV; if 'hful' use HFUL of Newey et al. (2012); if 'linear', constrain b1==b2==1 (this like an AKM estimator).
        ''', None),
    'stationary_a': (False, 'type', bool,
        '''
            (default=False) (For linear estimator) If True, constrain a1==a2.
        ''', None),
    'stationary_b': (False, 'type', bool,
        '''
            (default=False) (For linear estimator) If True, constrain b1==b2.
        ''', None),
    'profiling': (False, 'type', bool,
        '''
            (default=False) (For linear estimator) If True, estimate profiled LIML by partialling out period 1 covariates from period 2 covariates.
        ''', None),
    'instrument': ('firm_pairs', 'set', ['firm_pairs', 'worker_ids'],
        '''
            (default='firm_pairs') (For linear estimator) Which instrument to use - either 'firm_pairs' (use interaction of firm in period 1 and firm in period 2) or 'worker_ids' (use worker ids).
        ''', None),
    'coarse': (0, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0) (For linear estimator) Make instrument indicator coarser by dividing ids by `coarse` then rounding.
        ''', '>= 0'),
    'tik': (0, 'type', (float, int),
        '''
            (default=0) (For linear estimator) Add tik * I to instrument before inverting.
        ''', None),
    'C': (1, 'type', (float, int),
        '''
            (default=1) (For linear estimator) HFUL C factor.
        ''', None),
    ## Fixed point ##
    'how_split': ('income', 'set', ['income', 'enter_exit'],
        '''
            (default='income') (For fixed-point estimator) If 'income', split workers by their average income; if 'enter_exit', split workers by whether they enter or exit a given firm.
        ''', None),
    'alternative_estimator': (True, 'type', bool,
        '''
            (default=True) (For fixed-point estimator) If True, estimate using alternative estimator.
        ''', None),
    'weight_firm_pairs': (False, 'type', bool,
        '''
            (default=False) (For fixed-point estimator) If True, weight each pair of firms by the number of movers between them.
        ''', None) # ,
    # 'max_iters': (500, 'type_constrained', (int, _gteq1),
    #     '''
    #         (default=500) (For fixed-point estimator) Maximum number of iterations for fixed-point estimation (used when `alternative_estimator`=True).
    #     ''', '>= 1'),
    # 'threshold': (1e-5, 'type_constrained', ((float, int), _gteq0),
    #     '''
    #         (default=1e-5) (For fixed-point estimator) Threshold maximum absolute percent change between iterations to break fixed-point iterations (used when `alternative_estimator`=True).
    #     ''', '>= 0')
})

def iblm_params(update_dict=None):
    '''
    Dictionary of default iblm_params. Run tw.iblm_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of iblm_params
    '''
    new_dict = _iblm_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

class InteractedBLMEstimator():
    '''
    Class for estimating interacted-BLM.

    Arguments:
        params (ParamsDict or None): dictionary of parameters for interacted-BLM estimation. Run tw.iblm_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.iblm_params().
    '''
    def __init__(self, params=None):
        if params is None:
            params = iblm_params()

        self.params = params

    def fit(self, adata, rng=None):
        '''
        Estimate interacted BLM model.

        Arguments:
            adata (BipartitePandas DataFrame): long or collapsed long format labor data
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (tuple of NumPy Arrays): A, B, and alpha
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Unpack parameters ##
        params = self.params
        # Estimator
        estimator = params['estimator']
        # Whether data is weighted
        weighted = params['weighted']
        if not adata._col_included('w'):
            # Skip weighting if no weight column included
            weighted = False
        self.weighted = weighted
        # Progress bars
        self.no_pbars = (not params['progress_bars'])

        #### Estimate model ####
        ### Keep only movers ###
        jdata = adata.loc[adata.get_worker_m(is_sorted=True), :]

        ### Estimate ###
        if estimator == 'linear':
            jdata = jdata.to_permutedeventstudy(order='income', is_sorted=True, copy=False)
            A1, A2, B1, B2 = self._fit_a_b_linear(jdata)

            ## alpha_j1_j2 ##
            y1 = jdata.loc[:, 'y1'].to_numpy()
            y2 = jdata.loc[:, 'y2'].to_numpy()
            j1 = jdata.loc[:, 'j1'].to_numpy()
            j2 = jdata.loc[:, 'j2'].to_numpy()

            jdata.loc[:, 'alpha_j1_j2'] = (y1 - A1[j1]) / B1[j1] + (y2 - A2[j2]) / B2[j2]
            if weighted:
                jdata.loc[:, 'w'] = jdata.loc[:, 'w1'].to_numpy() * jdata.loc[:, 'w2'].to_numpy()
                jdata.loc[:, 'alpha_j1_j2'] *= jdata.loc[:, 'w'].to_numpy()
                groupby_j = jdata.groupby(['j1', 'j2'])
                alpha_j1_j2 = groupby_j['alpha_j1_j2'].sum() / groupby_j['w'].sum()
                jdata.drop('w', axis=1, inplace=True)
            else:
                alpha_j1_j2 = jdata.groupby(['j1', 'j2'])['alpha_j1_j2'].mean()
            jdata.drop('alpha_j1_j2', axis=1, inplace=True)

            # NOTE: is it correct to multiply by (1 / 2) here?
            alpha_j1_j2 = (1 / 2) * alpha_j1_j2.unstack(fill_value=0).to_numpy()

            ## alpha_j ##
            sdata = adata.loc[~(adata.get_worker_m(is_sorted=True)), :].to_eventstudy(is_sorted=True, copy=False)
            y1 = sdata.loc[:, 'y1'].to_numpy()
            y2 = sdata.loc[:, 'y2'].to_numpy()
            j1 = sdata.loc[:, 'j1'].to_numpy()
            j2 = sdata.loc[:, 'j2'].to_numpy()

            sdata.loc[:, 'alpha_j'] = (y1 - A1[j1]) / B1[j1] + (y2 - A2[j2]) / B2[j2]
            if weighted:
                sdata.loc[:, 'w'] = sdata.loc[:, 'w1'].to_numpy() * sdata.loc[:, 'w2'].to_numpy()
                sdata.loc[:, 'alpha_j'] *= sdata.loc[:, 'w'].to_numpy()
                groupby_j = sdata.groupby('j1')
                alpha_j = groupby_j['alpha_j'].sum() / groupby_j['w'].sum()
                sdata.drop('w', axis=1, inplace=True)
            else:
                alpha_j = sdata.groupby('j1')['alpha_j'].mean()
            sdata.drop('alpha_j', axis=1, inplace=True)

            # NOTE: is it correct to multiply by (1 / 2) here?
            alpha_j = (1 / 2) * alpha_j.to_numpy()

            ## Update parameters ##
            A, B, alpha = A1, B1, alpha_j
        elif estimator == 'fixed_point':
            B = self._fit_b_fixed_point(jdata, rng)
            A, alpha = self._fit_a_alpha_fixed_point(adata, B)

        return A, B, alpha

    def _fit_a_b_linear(self, jdata):
        '''
        Fit A and B using the linear estimator.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers

        Returns:
            (tuple of NumPy Arrays): estimated A1, A2, B1, and B2
        '''
        # Unpack parameters
        weighted = self.weighted
        method, norm_fid, stationary_a, stationary_b, profiling, instrument, coarse, tik, C = self.params.get_multiple(('method', 'norm_fid', 'stationary_a', 'stationary_b', 'profiling', 'instrument', 'coarse', 'tik', 'C'))

        if stationary_b and profiling:
            raise ValueError('Can set either `stationary_b`=True or `profiling`=True, but not both.')

        if stationary_b and (method == 'linear'):
            raise ValueError("Can set either `stationary_b`=True or `method`='linear', but not both.")

        if profiling and (method not in ['liml_1', 'hful']):
            raise ValueError(f"Can only set `profiling`=True if `method` is either 'liml_1' or 'hful', but input specifies {method!r}.")

        if instrument not in ['firm_pairs', 'worker_ids']:
            raise ValueError(f"`instrument` must be either 'firm_pairs' or 'worker_ids', but input specifies {instrument!r}.")

        if method not in ['liml_1', 'liml_2', 'iv', 'hful', 'linear']:
            raise ValueError(f"`method` must be one of 'liml_1', 'liml_2', 'iv', 'hful', or 'linear', but input specifies {method!r}.")

        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)
        if instrument == 'worker_ids':
            nw = jdata.n_workers()

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()
        if instrument == 'worker_ids':
            W = jdata.loc[:, 'i'].to_numpy()
        if weighted:
            Dp = (jdata.loc[:, 'w1'].to_numpy() * jdata.loc[:, 'w2'].to_numpy())
        else:
            Dp = 1

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))
        if method != 'linear':
            YY1 = csc_matrix((Y1, (range(ni), J1)), shape=(ni, nf))
            YY2 = csc_matrix((Y2, (range(ni), J2)), shape=(ni, nf))
        else:
            ## Linear ##
            ## Combine matrices and vectors ##
            if stationary_a:
                XX = (JJ1 - JJ2)[:, 1:]
            else:
                XX = hstack([JJ1[:, 1:], -JJ2])
            YY = Y1 - Y2
            DpXX = DxSP(Dp, XX)

            ## Compute intercepts (linear regression) ##
            ints = rss(DpXX.T @ XX).solve(DpXX.T @ YY, tol=1e-10)

            ## Extract results ##
            B1 = np.ones(nf)
            B2 = B1
            A1 = np.append(0, ints[: nf - 1])
            if stationary_a:
                A2 = A1
            else:
                A2 = ints[nf - 1: 2 * nf - 1]

            return A1, A2, B1, B2

        ## Construct instrument ##
        if coarse == 0:
            if instrument == 'firm_pairs':
                # Joint firm indicator
                KK = J1 + nf * J2

                # Transition probability matrix (in this case, matrix of instruments (j1, j2))
                ZZ = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nf ** 2))
            elif instrument == 'worker_ids':
                ZZ = csc_matrix((np.ones(ni), (range(ni), W)), shape=(ni, nw))
        else:
            if instrument == 'firm_pairs':
                ## Firm 1 ##
                # Joint firm indicator
                J2_KK = (J2 / coarse).astype(int, copy=False)
                KK1 = J1 + nf * J2_KK

                # Transition probability matrix (in this case, matrix of instruments (j1, j2))
                JJ12_1 = csc_matrix((np.ones(ni), (range(ni), KK1)), shape=(ni, nf * (np.max(J2_KK) + 1)))

                ## Firm 2 ##
                # Joint firm indicator
                J1_KK = (J1 / coarse).astype(int, copy=False)
                KK2 = J2 + nf * J1_KK

                # Transition probability matrix (in this case, matrix of instruments (j1, j2))
                JJ12_2 = csc_matrix((np.ones(ni), (range(ni), KK2)), shape=(ni, nf * (np.max(J1_KK) + 1)))

                ## Combine ##
                ZZ = hstack([JJ12_1, JJ12_2])
            elif instrument == 'worker_ids':
                WW_KK = (WW / coarse).astype(int, copy=False)
                ZZ = csc_matrix((np.ones(ni), (range(ni), WW_KK)), shape=(ni, np.max(WW_KK) + 1))

        if instrument == 'firm_pairs':
            # Drop zero columns of ZZ (firm pairs that don't occur in the data)
            ZZ_zeros = (np.asarray(ZZ.sum(axis=0))[0, :] == 0)
            ZZ = ZZ[:, ~(ZZ_zeros)]

        ## Construct matrices ##
        if profiling:
            ## Profiling ##
            if stationary_a:
                Y1J1 = hstack([-YY1, JJ1])
                DpY1J1 = DxSP(Dp, Y1J1)
                Y2J2 = hstack([YY2, -JJ2[:, 1:]])
                # NOTE: csc_matrix(pinv(Y1J1.todense())) == inv((Y1J1.T @ Y1J1).tocsc()) @ Y1J1.T
                XX = Y2J2 - Y1J1 @ inv((DpY1J1.T @ Y1J1).tocsc()) @ (DpY1J1.T @ Y2J2)
            else:
                Y1J1J2 = hstack([-YY1, JJ1[:, 1:], -JJ2])
                DpY1J1J2 = DxSP(Dp, Y1J1J2)
                # NOTE: csc_matrix(pinv(Y1J1J2.todense())) == inv((Y1J1J2.T @ Y1J1J2).tocsc()) @ Y1J1J2.T
                XX = YY2 - Y1J1J2 @ inv((DpY1J1J2.T @ Y1J1J2).tocsc()) @ (DpY1J1J2.T @ YY2)
        else:
            ## Standard ##
            if stationary_b:
                X1 = YY2 - YY1
            else:
                X1 = hstack([-YY1, YY2])
            if stationary_a:
                X2 = (JJ1 - JJ2)[:, 1:]
            else:
                X2 = hstack([JJ1[:, 1:], -JJ2])
            if method not in ['liml_2', 'iv']:
                XX = hstack([X1, X2])

        ## LIML ##
        if method == 'liml_1':
            ## Pre-multiply matrices ##
            DpZZ = DxSP(Dp, ZZ)
            ZZtXX = DpZZ.T @ XX

            ## LIML ##
            Wx = (DxSP(Dp, XX).T @ XX).tocsc()
            Wz = SPxD(ZZtXX.T, 1 / (diag_of_sp_prod(DpZZ.T, ZZ) + tik)) @ ZZtXX
            del DpZZ, ZZtXX

            # Smallest eigenvector
            WW = inv(Wx) @ Wz
            evals, evecs = np.linalg.eig(WW.todense())
            beta_hat = evecs[:, np.argmin(evals)]
            # Flatten and take real component
            beta_hat = np.real(np.asarray(beta_hat)[:, 0])
            # Normalize
            beta_hat = beta_hat / beta_hat[norm_fid]

        elif method in ['liml_2', 'iv']:
            ## Define normalization ##
            Y = -X1[:, norm_fid]
            X1 = X1[:, list(range(norm_fid)) + list(range(norm_fid + 1, X1.shape[1]))]

            ## Construct matrices ##
            XX = hstack([X1, X2])

            ## Pre-multiply matrices ##
            DpXX = DxSP(Dp, XX)
            DpZZ = DxSP(np.sqrt(Dp), ZZ)
            ZZtZZinv = 1 / (diag_of_sp_prod(DxSP(Dp, ZZ).T, ZZ) + tik)
            ZZ_ZZtZZinv = SPxD(DpZZ, ZZtZZinv)
            XXtZZinv = (XX.T @ ZZ_ZZtZZinv)
            del ZZtZZinv

            if method == 'liml_2':
                ## LIML 2 ##
                ## Construct matrices ##
                R = hstack([Y, X1])

                ## Pre-multiply matrices ##
                DpX2 = DxSP(Dp, X2)
                DpR = DxSP(Dp, R)
                RtR = DpR.T @ R

                ## LIML ##
                Wx = RtR - (R.T @ X2) @ inv((DpX2.T @ X2).tocsc()) @ (DpX2.T @ DpR)
                Wz = (RtR - (R.T @ ZZ_ZZtZZinv) @ (DpZZ.T @ DpR)).tocsc()
                del DpX2, RtR

                # Smallest eigenvalue
                WW = Wx @ inv(Wz)
                evals, _ = np.linalg.eig(WW.todense())
                lambda_ = np.min(evals)

                # Construct new matrices
                RR = ((1 - lambda_) * XX.T @ XX + lambda_ * XXtZZinv @ (DpZZ.T @ XX)).tocsc()
                RY = (1 - lambda_) * XX.T @ Y + lambda_ * XXtZZinv @ (DpZZ.T @ Y)
            elif method == 'iv':
                ## IV (lambda_ == 1) ##
                # Construct new matrices
                RR = (XXtZZinv @ (DpZZ.T @ XX)).tocsc()
                RY = XXtZZinv @ (DpZZ.T @ Y)

            del DpZZ, ZZ_ZZtZZinv, XXtZZinv

            ## OLS ##
            RR = np.asarray((RR).todense())
            RY = np.asarray((RY).todense())[:, 0]
            beta_hat = solve_qp(RR, -RY, solver='quadprog')
            beta_hat = np.concatenate([beta_hat[: norm_fid], [1], beta_hat[norm_fid:]])

        elif method == 'hful':
            ## HFUL ##
            ## Pre-multiply matrices ##
            DpZZ = DxSP(np.sqrt(Dp), ZZ)
            XXtDpZZ = XX.T @ DpZZ
            ZZtZZinv = 1 / (diag_of_sp_prod(DxSP(Dp, ZZ).T, ZZ) + tik)

            ## HFUL ##
            Pz_diag = diag_of_sp_prod(SPxD(DpZZ, ZZtZZinv).tocsc(), DpZZ.T)
            Wx = (XX.T @ XX).tocsc()
            Wz = SPxD(XXtDpZZ, ZZtZZinv) @ XXtDpZZ.T - SPxD(XX.T, Pz_diag) @ XX
            del DpZZ, XXtDpZZ, ZZtZZinv, Pz_diag

            # Smallest eigenvalue
            WW = inv(Wx) @ Wz
            evals, _ = np.linalg.eig(WW.todense())
            lambda_ = np.min(evals)

            lambda2 = (lambda_ - (1 - lambda_) * (C / ni)) / (1 - (1 - lambda_) * (C / ni))

            ## OLS ##
            ## Solve (Wx)^{-1} @ Wz @ v = lambda_ @ v, where we normalize v[norm_fid] = 1 ##
            WW.setdiag(WW.diagonal() - lambda2)
            WY = -WW[:, norm_fid]
            WX = WW[:, list(range(norm_fid)) + list(range(norm_fid + 1, WW.shape[1]))]

            ## OLS ##
            DpWX = DxSP(Dp, WX)
            XX = np.asarray((DpWX.T @ WX).todense())
            XY = np.asarray((DpWX.T @ WY).todense())[:, 0]
            beta_hat = solve_qp(XX, -XY, solver='quadprog')
            beta_hat = np.concatenate([beta_hat[: norm_fid], [1], beta_hat[norm_fid:]])

        if profiling and (not stationary_a):
            ## Compute intercepts (linear regression) ##
            XX = hstack([JJ1[:, 1:], -JJ2])
            DpXX = DxSP(Dp, XX)
            YY = (YY1 - YY2) @ beta_hat
            ints = rss(DpXX.T @ XX).solve(DpXX.T @ YY, tol=1e-10)
            # Append to beta_hat
            beta_hat = np.concatenate([beta_hat, ints])

        ## Extract results ##
        idx = 0
        B1 = 1 / beta_hat[idx: nf]
        idx += nf
        if stationary_b or profiling:
            B2 = B1
        else:
            B2 = 1 / beta_hat[idx: idx + nf]
            idx += nf
        A1 = np.append(0, beta_hat[idx: idx + (nf - 1)]) * B1
        idx += (nf - 1)
        if stationary_a:
            A2 = A1 * (B2 / B1)
        else:
            A2 = beta_hat[idx: idx + nf] * B2

        return A1, A2, B1, B2

    def _fit_b_fixed_point(self, jdata, rng=None):
        '''
        Fit B using the fixed-point estimator.

        Arguments:
            jdata (BipartitePandas DataFrame): long or collapsed long format labor data for movers
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): estimated B
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Check that jdata has no returns
        if not jdata.no_returns:
            raise ValueError("Cannot run the fixed-point estimator if there are returns in the data. When cleaning your data, please set the parameter 'drop_returns' to drop returns.")

        # Unpack parameters
        weighted = self.weighted
        how_split, alternative_estimator, weight_firm_pairs = self.params.get_multiple(('how_split', 'alternative_estimator', 'weight_firm_pairs')) # max_iters, threshold

        # ## Estimate using event study format ##
        # if weighted:
        #     # Weight y1 and y2
        #     w1 = jdata.loc[:, 'w1'].to_numpy()
        #     w2 = jdata.loc[:, 'w2'].to_numpy()
        #     y1 = jdata.loc[:, 'y1'].to_numpy().copy()
        #     y2 = jdata.loc[:, 'y2'].to_numpy().copy()
        #     jdata.loc[:, 'y1'] = w1 * y1
        #     jdata.loc[:, 'y2'] = w2 * y2
        # groupby_j1j2 = jdata.groupby(['j1', 'j2'])

        # ## Solve for y1 ##
        # if weighted:
        #     w1_sum = groupby_j1j2['w1'].sum()
        #     y1_mean = groupby_j1j2['y1'].sum() / w1_sum
        # else:
        #     if weight_firm_pairs:
        #         w1_sum = groupby_j1j2['y1'].size()
        #     y1_mean = groupby_j1j2['y1'].mean()
        # row = y1_mean.index.get_level_values(0)
        # col = y1_mean.index.get_level_values(1)
        # M_y1 = csc_matrix((y1_mean, (row, col)))

        # ## Solve for y2 ##
        # if weighted:
        #     w2_sum = groupby_j1j2['w2'].sum()
        #     y2_mean = groupby_j1j2['y2'].sum() / w2_sum
        # else:
        #     if weight_firm_pairs:
        #         w2_sum = groupby_j1j2['y2'].size()
        #     y2_mean = groupby_j1j2['y2'].mean()
        # row = y2_mean.index.get_level_values(0)
        # col = y2_mean.index.get_level_values(1)
        # M_y2 = csc_matrix((y2_mean, (row, col)))

        # ## Combine ##
        # M0 = M_y2.T - M_y1

        # if alternative_estimator:
        #     # Multiply lower triangular part of M0 by -1 (we equivalently subtract twice its value)
        #     M0 -= 2 * tril(M0, k=-1)

        # if weight_firm_pairs:
        #     ## Weight firm pairs ##
        #     # w1_sum
        #     row = w1_sum.index.get_level_values(0)
        #     col = w1_sum.index.get_level_values(1)
        #     w1_sum = csc_matrix((w1_sum, (row, col)))
        #     # w2_sum
        #     row = w2_sum.index.get_level_values(0)
        #     col = w2_sum.index.get_level_values(1)
        #     w2_sum = csc_matrix((w2_sum, (row, col)))
        #     # Weight
        #     M0 = M0.multiply(w1_sum + w2_sum.T)

        # if weighted:
        #     # Restore original y1 and y2
        #     jdata.loc[:, 'y1'] = y1
        #     jdata.loc[:, 'y2'] = y2

        if how_split == 'income':
            # Sort workers by average income
            with ChainedAssignment():
                if weighted:
                    jdata.loc[:, 'weighted_y'] = jdata.loc[:, 'w'].to_numpy() * jdata.loc[:, 'y'].to_numpy()
                    jdata['mean_y'] = jdata.groupby('i')['weighted_y'].transform('sum') / jdata.groupby('i')['w'].transform('sum')
                    jdata.drop('weighted_y', axis=1, inplace=True)
                else:
                    jdata['mean_y'] = jdata.groupby('i')['y'].transform('mean')
                jdata.sort_values('mean_y', inplace=True)
                jdata.drop('mean_y', axis=1, inplace=True)

        # Initial data construction
        nf = jdata.n_firms()
        j = jdata.loc[:, 'j'].to_numpy()
        y = jdata.loc[:, 'y'].to_numpy()
        if weighted:
            w = jdata.loc[:, 'w'].to_numpy()
        # Construct graph
        G, _ = jdata._construct_graph(connectedness='leave_out_observation', is_sorted=True, copy=False)
        # Construct lists to store results
        M0_data = []
        M0_row = []
        M0_col = []

        for j1 in trange(nf, disable=self.no_pbars):
            ### Iterate over all firms ###
            # Find workers who worked at firm j1
            obs_in_j1 = (j == j1)
            with ChainedAssignment():
                jdata.loc[:, 'obs_in_j1'] = obs_in_j1
            i_in_j1 = jdata.groupby('i', sort=False)['obs_in_j1'].transform('max').to_numpy()
            # Take subset of data for workers who worked at firm j1
            jdata_j1 = jdata.loc[i_in_j1, :]
            j_j1 = j[i_in_j1]
            y_j1 = y[i_in_j1]
            if weighted:
                w_j1 = w[i_in_j1]
            # For each firm, find its neighboring firms
            j1_neighbors = G.neighborhood(j1, order=2, mindist=2)
            for j2 in tqdm(j1_neighbors, disable=self.no_pbars):
                ### Iterate over all neighbors ###
                if j2 > j1:
                    ## Account for symmetry by estimating only if j2 > j1 ##
                    # Find workers who worked at both firms j1 and j2
                    obs_in_j2 = (j_j1 == j2)
                    with ChainedAssignment():
                        jdata_j1.loc[:, 'obs_in_j2'] = obs_in_j2
                    i_in_j2 = jdata_j1.groupby('i', sort=False)['obs_in_j2'].transform('max').to_numpy()
                    # Take subsets of data for workers who worked at both firms j1 and j2
                    j_j12 = j_j1[i_in_j2]
                    y_j12 = y_j1[i_in_j2]
                    if weighted:
                        w_j12 = w_j1[i_in_j2]
                    # Take subsets of data specifically for firms j1 and j2
                    is_j12 = (j_j12 == j1) | (j_j12 == j2)
                    j_j12 = j_j12[is_j12]
                    y_j12 = y_j12[is_j12]
                    if weighted:
                        w_j12 = w_j12[is_j12]
                    if len(j_j12) >= 4:
                        ## If there are at least two workers with observations at both firms ##
                        # Split data for j1 and j2
                        y_j11 = y_j12[j_j12 == j1]
                        y_j22 = y_j12[j_j12 == j2]
                        if weighted:
                            w_j11 = w_j12[j_j12 == j1]
                            w_j22 = w_j12[j_j12 == j2]
                        # Split observations into entering/exiting groups
                        if how_split == 'enter_exit':
                            j_j12_first = j_j12[np.arange(len(j_j12)) % 2 == 0]
                            entering = (j_j12_first == j2)
                            exiting = (j_j12_first == j1)
                        elif how_split == 'income':
                            if len(y_j11) % 2 == 0:
                                halfway = len(y_j11) // 2
                            else:
                                halfway = len(y_j11) // 2 + rng.binomial(n=1, p=0.5)
                            entering = (np.arange(len(y_j11)) < halfway)
                            exiting = (np.arange(len(y_j11)) >= halfway)
                        else:
                            raise ValueError(f"`how_split` must be one of 'enter_exit' or 'income', but input specifies {how_split!r}.")
                        if (np.sum(entering) > 0) and (np.sum(exiting) > 0):
                            # Need workers to both enter and exit
                            if weighted:
                                entering_w1 = np.sum(w_j11[entering])
                                entering_w2 = np.sum(w_j22[entering])
                                exiting_w1 = np.sum(w_j11[exiting])
                                exiting_w2 = np.sum(w_j22[exiting])
                                w1 = entering_w1 + exiting_w1
                                w2 = entering_w2 + exiting_w2
                                entering_y1 = np.sum(w_j11[entering] * y_j11[entering]) / entering_w1
                                entering_y2 = np.sum(w_j22[entering] * y_j22[entering]) / entering_w2
                                exiting_y1 = np.sum(w_j11[exiting] * y_j11[exiting]) / exiting_w1
                                exiting_y2 = np.sum(w_j22[exiting] * y_j22[exiting]) / exiting_w2
                            else:
                                w1 = len(y_j11)
                                w2 = len(y_j22)
                                entering_y1 = np.mean(y_j11[entering])
                                entering_y2 = np.mean(y_j22[entering])
                                exiting_y1 = np.mean(y_j11[exiting])
                                exiting_y2 = np.mean(y_j22[exiting])
                            if not weight_firm_pairs:
                                w1 = 1
                                w2 = 1
                            ## Compute M0 (use symmetry) ##
                            # For M0[j1, j2] = (entering_y1 - exiting_y1)
                            M0_data.append(w1 * (entering_y1 - exiting_y1))
                            M0_row.append(j1)
                            M0_col.append(j2)
                            # For M0[j2, j1] = (exiting_y2 - entering_y2)
                            M0_data.append(w2 * (exiting_y2 - entering_y2))
                            if alternative_estimator:
                                M0_data[-1] *= -1
                            M0_row.append(j2)
                            M0_col.append(j1)

        if how_split == 'income':
            # Sort jdata
            jdata.sort_rows(copy=False)

        # Convert to sparse matrix
        M0 = csc_matrix((M0_data, (M0_row, M0_col)), shape=(nf, nf))
        S0 = - np.asarray(M0.sum(axis=0))[0, :]
        if alternative_estimator:
            S0 *= -1

        ## Solve for B ##
        lhs = DxSP(1 / S0, M0)

        if False: # alternative_estimator:
            ## Fixed point guaranteed ##
            # NOTE: for some reason this sometimes converges to the wrong fixed point, so we comment it out for now
            prev_guess = np.ones(nf)
            for _ in range(max_iters):
                new_guess = lhs @ prev_guess
                new_guess /= new_guess[0]
                if np.max(np.abs((new_guess - prev_guess) / prev_guess)) <= threshold:
                    break
                prev_guess = new_guess
            evec = new_guess
        else:
            ## Fixed point not guaranteed ##
            try:
                _, evec = eigs(lhs, k=1, sigma=1)
            except RuntimeError:
                # If scipy.linalg.eigs doesn't work, fall back to NumPy
                evals, evecs = np.linalg.eig(lhs.todense())
                evec = np.asarray(evecs[:, np.argmin(np.abs(evals - 1))])

            # Flatten and take real component
            evec = np.real(evec[:, 0])

            # Normalize
            evec /= evec[0]

        return evec

    def _fit_a_alpha_fixed_point(self, adata, B):
        '''
        Fit A and alpha using the fixed-point estimator.

        Arguments:
            adata (BipartitePandas DataFrame): long or collapsed long format labor data
            B (NumPy Array): estimated B

        Returns:
            (tuple of NumPy Arrays): estimated A and alpha
        '''
        ### Prepare matrices ###
        nn = len(adata)
        nf = adata.n_firms()
        nw = adata.n_workers()

        ## Y (income) ##
        Y = adata.loc[:, 'y'].to_numpy()

        ## J (firms) ##
        J = csc_matrix((np.ones(nn), (adata.index.to_numpy(), adata.loc[:, 'j'].to_numpy())), shape=(nn, nf))

        # Normalize one firm to 0
        J = J[:, range(nf - 1)]

        ## W (workers) ##
        W = csc_matrix((B[adata.loc[:, 'j'].to_numpy()], (adata.index.to_numpy(), adata.loc[:, 'i'].to_numpy())), shape=(nn, nw))

        if self.weighted:
            ### Weighted ###
            ## Dp (weight) ##
            Dp = adata.loc[:, 'w'].to_numpy()

            ## Weighted J and W ##
            DpJ = DxSP(Dp, J)
            DpW = DxSP(Dp, W)
        else:
            ## Unweighted ##
            ## Dp (weight) ##
            Dp = 1

            ## Weighted J and W ##
            DpJ = J
            DpW = W

        ## Dwinv ##
        Dwinv = 1 / diag_of_sp_prod(W.T, DpW)

        ## Dwinv @ W.T @ Dp @ J ##
        WtDpJ = W.T @ DpJ
        DwinvWtDpJ = DxSP(Dwinv, WtDpJ.tocsc())

        ## M ##
        M = rss(J.T @ DpJ - WtDpJ.T @ DwinvWtDpJ)

        ## Estimate A and alpha ##
        A = M.solve(DpJ.T @ Y - DwinvWtDpJ.T @ DpW.T @ Y, tol=1e-10)

        alpha = Dwinv * (DpW.T @ Y) - DwinvWtDpJ @ A

        ## Add 0 for normalized firm ##
        A = np.append(A, 0)

        return A, alpha
