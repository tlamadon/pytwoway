'''
Implement the interacted estimator from Bonhomme, Lamadon, & Manresa.
'''
from tqdm.auto import tqdm, trange
import numpy as np
from scipy.sparse import csc_matrix, eye, hstack
from scipy.sparse.linalg import eigs, inv
from scipy.linalg import pinv
from pyamg import ruge_stuben_solver as rss
from bipartitepandas.util import ChainedAssignment
import pytwoway as tw

class InteractedBLMModel():
    '''
    Class for estimating interacted-BLM.
    '''
    def __init__(self):
        pass

    def fit_b_fixed_point(self, jdata, how_split='income', rng=None):
        '''
        Fit fixed-point estimator for b.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            how_split (str): if 'income', split workers by their average income; if 'enter_exit', split workers by whether they enter or exit a given firm
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): estimated b
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Weighted
        weighted = jdata._col_included('w')

        if how_split == 'income':
            # Sort workers by average income
            if weighted:
                jdata['weighted_y'] = jdata['w'] * jdata['y']
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

        for j1 in trange(nf):
            ### Iterate over all firms ###
            # Find workers who worked at firm j1
            obs_in_j1 = (j == j1)
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
            for j2 in tqdm(j1_neighbors):
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
                                entering_y1 = np.sum(w_j11[entering] * y_j11[entering]) / np.sum(w_j11[entering])
                                entering_y2 = np.sum(w_j22[entering] * y_j22[entering]) / np.sum(w_j22[entering])
                                exiting_y1 = np.sum(w_j11[exiting] * y_j11[exiting]) / np.sum(w_j11[exiting])
                                exiting_y2 = np.sum(w_j22[exiting] * y_j22[exiting]) / np.sum(w_j22[exiting])
                            else:
                                entering_y1 = np.mean(y_j11[entering])
                                entering_y2 = np.mean(y_j22[entering])
                                exiting_y1 = np.mean(y_j11[exiting])
                                exiting_y2 = np.mean(y_j22[exiting])
                            ## Compute M0 (use symmetry) ##
                            # For M0[j1, j2] = (entering_y1 - exiting_y1)
                            M0_data.append(entering_y1 - exiting_y1)
                            M0_row.append(j1)
                            M0_col.append(j2)
                            # For M0[j2, j1] = (exiting_y2 - entering_y2)
                            M0_data.append(exiting_y2 - entering_y2)
                            M0_row.append(j2)
                            M0_col.append(j1)

        if how_split == 'income':
            # Sort jdata
            jdata.sort_rows(copy=False)

        # Convert to sparse matrix
        M0 = csc_matrix((M0_data, (M0_row, M0_col)), shape=(nf, nf))
        S0 = - np.asarray(M0.sum(axis=0))[0, :]
        if False: # compute_exp_b:
            M0_data = np.array(M0_data)
            M0 = csc_matrix((np.exp(M0_data), (M0_row, M0_col)), shape=(nf, nf))
            S0 = np.asarray(csc_matrix((np.exp(-M0_data), (M0_row, M0_col)), shape=(nf, nf)).sum(axis=0))[0, :]

        ## Solve for B ##
        lhs = tw.util.DxSP(1 / S0, M0)
        # NOTE: fixed point doesn't work
        # B = np.random.normal(size=nf)
        # B /= B[0]
        # for _ in range(1000):
        #     B = lhs @ B
        #     B /= B[0]
        _, evec = eigs(lhs, k=1, sigma=1)

        # Flatten and take real component
        evec = np.real(evec[:, 0])

        ## Check eigenvector ##
        # NOTE: can check if eigenvector has a repeated value, but too costly to do it properly
        evec2 = lhs @ evec
        # Check that it's actually an eigenvector
        if not (np.var(evec2 / evec) < 1e-10):
            # If it's not actually an eigenvector
            return np.ones(nf)
            # raise ValueError('Fixed point cannot be found.')

        # Normalize
        evec /= evec[0]

        return evec

    def fit_b_liml_regular(self, jdata, norm_fid=0, coarse=0, tik=0):
        '''
        Fit b using regular LIML.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize
            coarse (float): make joint firm indicator coarser by dividing the second firm id by `coarse`
            tik (float): add tik * I to instrument before inverting to make full rank; used when coarse != 0

        Returns:
            (NumPy Array): estimated b
        '''
        if (coarse != 0) and (tik == 0):
            raise ValueError('If `coarse` != 0, then must also set `tik` != 0.')

        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))
        YY1 = csc_matrix((Y1, (range(ni), J1)), shape=(ni, nf))
        YY2 = csc_matrix((Y2, (range(ni), J2)), shape=(ni, nf))

        if coarse == 0:
            # Joint firm indicator
            KK = J1 + nf * J2

            # Transition probability matrix (in this case, matrix of instruments (j1, j2))
            JJ12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nf ** 2))
        else:
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
            JJ12 = hstack([JJ12_1, JJ12_2])

        # Drop zero columns of JJ12
        JJ12_zeros = (np.asarray(JJ12.sum(axis=0))[0, :] == 0)
        JJ12 = JJ12[:, ~(JJ12_zeros)]

        ## Combine matrices ##
        XX = hstack([YY2, -YY1, JJ1[:, 1:], -JJ2])

        ## LIML ##
        JJtXX = JJ12.T @ XX
        Wz = JJtXX.T @ inv((JJ12.T @ JJ12 + tik * eye(JJ12.shape[1])).tocsc()) @ JJtXX
        Wx = (XX.T @ XX).tocsc()
        del JJtXX

        # Smallest eigenvector (eigenvalues bounded between 0 and 1)
        WW = inv(Wx) @ Wz
        _, evec = eigs(WW, k=1, sigma=0)
        evec = np.real(evec[:, 0])

        ## Extract results ##
        b_liml = evec / evec[norm_fid]
        B2 = 1 / b_liml[: nf]
        B1 = 1 / b_liml[nf: 2 * nf]
        A1 = np.concatenate([[0], b_liml[2 * nf: 3 * nf - 1]]) * B1
        A2 = b_liml[3 * nf - 1: 4 * nf - 1] * B2

        return B1, B2

    def fit_b_liml_regular_stationary_b(self, jdata, norm_fid=0, profiling=False):
        '''
        Fit b using regular LIML with the constraint b1=b2.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize
            profiling (bool): if True, estimate profiled LIML

        Returns:
            (NumPy Array): estimated b
        '''
        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))
        YY1 = csc_matrix((Y1, (range(ni), J1)), shape=(ni, nf))
        YY2 = csc_matrix((Y2, (range(ni), J2)), shape=(ni, nf))

        # Joint firm indicator
        KK = J1 + nf * J2

        # Transition probability matrix (in this case, matrix of instruments (j1, j2))
        JJ12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nf ** 2))

        # Drop zero columns of JJ12
        JJ12_zeros = (np.asarray(JJ12.sum(axis=0))[0, :] == 0)
        JJ12 = JJ12[:, ~(JJ12_zeros)]

        if profiling:
            ## Profiling ##
            Y1J1 = hstack([-YY1, JJ1])
            Y2J2 = hstack([YY2, -JJ2[:, 1:]])
            XX = Y2J2 - Y1J1 @ csc_matrix(pinv(Y1J1.todense())) @ Y2J2
        else:
            ## Combine matrices ##
            XX = hstack([YY2 - YY1, JJ1[:, 1:], -JJ2])

        ## LIML ##
        JJtXX = JJ12.T @ XX
        Wz = JJtXX.T @ inv((JJ12.T @ JJ12).tocsc()) @ JJtXX
        Wx = (XX.T @ XX).tocsc()
        del JJtXX

        # Smallest eigenvector (eigenvalues bounded between 0 and 1)
        WW = inv(Wx) @ Wz
        _, evec = eigs(WW, k=1, sigma=0)
        evec = np.real(evec[:, 0])

        ## Extract results ##
        b_liml = evec / evec[norm_fid]
        B2 = 1 / b_liml[: nf]
        B1 = B2
        A1 = np.concatenate([[0], b_liml[nf: 2 * nf - 1]]) * B1
        if profiling:
            # FIXME I'm not completely sure this code is correct for profiling
            A2 = A1
        else:
            A2 = b_liml[2 * nf - 1: 3 * nf - 1] * B2

        return B1, B2

    def fit_b_liml_regular_profiling(self, jdata, norm_fid=0):
        '''
        Fit b using regular LIML where the b terms are profiled out in period 2 first, then they are used to recover the intercepts in both periods.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize

        Returns:
            (NumPy Array): estimated b
        '''
        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))
        YY1 = csc_matrix((Y1, (range(ni), J1)), shape=(ni, nf))
        YY2 = csc_matrix((Y2, (range(ni), J2)), shape=(ni, nf))

        # Joint firm indicator
        KK = J1 + nf * J2

        # Transition probability matrix (in this case, matrix of instruments (j1, j2))
        JJ12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nf ** 2))

        # Drop zero columns of JJ12
        JJ12_zeros = (np.asarray(JJ12.sum(axis=0))[0, :] == 0)
        JJ12 = JJ12[:, ~(JJ12_zeros)]

        ## Combine matrices ##
        XX1 = hstack([JJ1[:, 1:], -JJ2])

        ## Profiling ##
        Y1J1J2 = hstack([-YY1, XX1])
        XX2 = YY2 - Y1J1J2 @ csc_matrix(pinv(Y1J1J2.todense())) @ YY2

        ## LIML ##
        JJtXX = JJ12.T @ XX2
        Wz = JJtXX.T @ inv((JJ12.T @ JJ12).tocsc()) @ JJtXX
        Wx = (XX2.T @ XX2).tocsc()
        del JJtXX

        # Smallest eigenvector (eigenvalues bounded between 0 and 1)
        WW = inv(Wx) @ Wz
        _, evec = eigs(WW, k=1, sigma=0)
        evec = np.real(evec[:, 0])

        ## Compute intercepts (linear regression) ##
        YY = (YY1 - YY2) @ evec
        ints = rss(XX1.T @ XX1).solve(XX1.T @ YY, tol=1e-10)

        ## Extract results ##
        b_liml = np.concatenate([evec, ints])
        b_liml = b_liml / b_liml[norm_fid]
        B2 = 1 / b_liml[: nf]
        B1 = B2
        A1 = np.concatenate([[0], b_liml[nf: 2 * nf - 1]]) * B1
        A2 = b_liml[2 * nf - 1: 3 * nf - 1] * B2

        return B1, B2

    def fit_b_liml_regular_HFUL(self, jdata, norm_fid=0, C=1):
        '''
        Fit b using regular LIML with HFUL of Hausman Newey et al. 2012 QE.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize
            C (float): HFUL factor

        Returns:
            (NumPy Array): estimated b
        '''
        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))
        YY1 = csc_matrix((Y1, (range(ni), J1)), shape=(ni, nf))
        YY2 = csc_matrix((Y2, (range(ni), J2)), shape=(ni, nf))

        # Joint firm indicator
        KK = J1 + nf * J2

        # Transition probability matrix (in this case, matrix of instruments (j1, j2))
        JJ12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nf ** 2))

        # Drop zero columns of JJ12
        JJ12_zeros = (np.asarray(JJ12.sum(axis=0))[0, :] == 0)
        JJ12 = JJ12[:, ~(JJ12_zeros)]

        ## Combine matrices ##
        XX = hstack([YY2, -YY1, JJ1[:, 1:], -JJ2])

        ## LIML ##
        Pz = JJ12 @ inv((JJ12.T @ JJ12).tocsc()) @ JJ12.T
        Pz.setdiag(0)
        Wz = XX.T @ Pz @ XX
        Wx = (XX.T @ XX).tocsc()

        # Smallest eigenvalue (eigenvalues NOT bounded between 0 and 1)
        WW = inv(Wx) @ Wz
        evals, _ = np.linalg.eig(WW.todense())
        eval = np.min(evals)

        ## HFUL ##
        lambda_ = (eval - (C / ni) * (1 - eval)) / (1 - (C / ni) * (1 - eval))

        ## Solve (Wx)^{-1} @ Wz @ v = lambda_ @ v, where we normalize v[0] = 1 ##
        M = WW - lambda_ * eye(WW.shape[0])
        MX = M[:, 1:]
        b_liml = rss(MX.T @ MX).solve(-(MX.T @ M[:, 0]).todense())

        ## Extract results ##
        b_liml = np.concatenate([[1], b_liml])
        B2 = 1 / b_liml[: nf]
        B1 = 1 / b_liml[nf: 2 * nf]
        A1 = np.concatenate([[0], b_liml[2 * nf: 3 * nf - 1]]) * B1
        A2 = b_liml[3 * nf - 1: 4 * nf - 1] * B2

        return B1, B2

    def fit_b_liml_single_iv(self, jdata, norm_fid=0):
        '''
        Fit b using LIML on model in difference, single equation IV.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize

        Returns:
            (NumPy Array): estimated b
        '''
        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))
        YY1 = csc_matrix((Y1, (range(ni), J1)), shape=(ni, nf))
        YY2 = csc_matrix((Y2, (range(ni), J2)), shape=(ni, nf))

        # Joint firm indicator
        KK = J1 + nf * J2

        # Transition probability matrix (in this case, matrix of instruments (j1, j2))
        JJ12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nf ** 2))

        # Drop zero columns of JJ12
        JJ12_zeros = (np.asarray(JJ12.sum(axis=0))[0, :] == 0)
        JJ12 = JJ12[:, ~(JJ12_zeros)]

        ## Combine matrices ##
        X1 = hstack([YY1, YY2])
        X2 = hstack([JJ1, JJ2])

        ## Define normalization ##
        Y = -X1[:, norm_fid]
        X1 = X1[:, list(range(norm_fid)) + list(range(norm_fid + 1, 2 * nf))]
        X2 = X2[:, range(2 * nf - 1)]

        ## Combine matrices ##
        R = hstack([Y, X1])
        XX = hstack([X1, X2])

        ## LIML ##
        JJtJJinv = inv((JJ12.T @ JJ12).tocsc())
        RtR = R.T @ R
        JJtR = JJ12.T @ R
        X2tR = X2.T @ R
        Wz = (RtR - JJtR.T @ JJtJJinv @ JJtR).tocsc()
        Wx = RtR - X2tR.T @ inv((X2.T @ X2).tocsc()) @ X2tR
        del RtR, JJtR, X2tR

        # Smallest eigenvalue (eigenvalues bounded between 0 and 1)
        WW = Wx @ inv(Wz)
        eval, _ = eigs(WW, k=1, sigma=0)

        JJtXX = JJ12.T @ XX
        RR = ((1 - eval) * XX.T @ XX + eval * JJtXX.T @ JJtJJinv @ JJtXX).tocsc()
        RY = (1 - eval) * XX.T @ Y + eval * JJtXX.T @ JJtJJinv @ JJ12.T @ Y

        ## Extract results ##
        b_liml = np.real(np.asarray((inv(RR) @ RY).todense()).flatten())
        tau = np.ones(nf)
        tau[: norm_fid] = b_liml[: norm_fid]
        tau[norm_fid + 1:] = b_liml[norm_fid: nf - 1]
        B1 = 1 / tau
        B2 = - 1 / b_liml[nf - 1: 2 * nf - 1]
        A1 = - b_liml[2 * nf - 1: 3 * nf - 1] * B1
        A2 = np.zeros(nf)
        A2[: nf - 1] = b_liml[3 * nf - 1: 4 * nf - 2] * B2[: nf - 1]

        return B1, B2

    def fit_b_linear(self, jdata, norm_fid=0):
        '''
        Fit b using regular LIML but with the constraint b1=b2=1 (this is like an AKM estimator).

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize

        Returns:
            (NumPy Array): estimated b
        '''
        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))

        ## Combine matrices and vectors ##
        XX = hstack([JJ1[:, list(range(0, norm_fid)) + list(range(norm_fid + 1, nf))], -JJ2])
        YY = Y1 - Y2

        ## Compute intercepts (linear regression) ##
        ints = rss(XX.T @ XX).solve(XX.T @ YY, tol=1e-10)

        ## Extract results ##
        B2 = np.ones(nf)
        B1 = B2
        A1 = np.concatenate([ints[: norm_fid], [0], ints[norm_fid: nf - 1]])
        A2 = ints[nf - 1: 2 * nf - 1]

        return B1, B2

    def fit_b_linear_stationary_a(self, jdata, norm_fid=0):
        '''
        Fit b using regular LIML but with the constraints b1=b2=1 and a1=a2 (this is like an AKM estimator).

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize

        Returns:
            (NumPy Array): estimated b
        '''
        # Parameters
        nf, ni = jdata.n_firms(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()

        ## Sparse matrix representations ##
        JJ1 = csc_matrix((np.ones(ni), (range(ni), J1)), shape=(ni, nf))
        JJ2 = csc_matrix((np.ones(ni), (range(ni), J2)), shape=(ni, nf))

        ## Combine matrices and vectors ##
        XX = (JJ1 - JJ2)[:, list(range(0, norm_fid)) + list(range(norm_fid + 1, nf))]
        YY = Y1 - Y2

        ## Compute intercepts (linear regression) ##
        ints = rss(XX.T @ XX).solve(XX.T @ YY, tol=1e-10)

        ## Extract results ##
        B2 = np.ones(nf)
        B1 = B2
        A1 = np.concatenate([ints[: norm_fid], [0], ints[norm_fid: nf - 1]])
        A2 = A1

        return B1, B2
