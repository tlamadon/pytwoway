'''
Implement the interacted estimator from Bonhomme, Lamadon, & Manresa.
'''
from tqdm.auto import tqdm, trange
import numpy as np
from scipy.sparse import csc_matrix, hstack
from scipy.sparse.linalg import eigs, inv
from bipartitepandas.util import ChainedAssignment
import pytwoway as tw

# import bipartitepandas as bpd
# rng = np.random.default_rng(1234)
a = bpd.BipartiteDataFrame(bpd.SimBipartite(bpd.sim_params({'n_workers': 1000, 'w_sig': 0})).simulate()).clean().collapse(level='match').construct_artificial_time()
jdata = a[a.get_worker_m()].clean()
# jdata = jdata.cluster().to_eventstudy()
# # jdata = jdata.cluster()
# # jdata['j'] = jdata['g'].copy()
# # jdata = bpd.BipartiteDataFrame(jdata[['i', 'j', 'y']])
# # jdata = jdata.clean().collapse(level='match').construct_artificial_time()
# # jdata = a[a.get_worker_m()].clean()


class InteractedBLMModel():
    '''
    Class for estimating interacted-BLM.
    '''
    def __init__(self):
        pass

    def fit_b_fixed_point(self, jdata, rng=None):
        '''
        Fit fixed-point estimator for b.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): estimated b
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Initial data construction
        nf = jdata.n_firms()
        j = jdata.loc[:, 'j'].to_numpy()
        y = jdata.loc[:, 'y'].to_numpy()
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
                    # Take subsets of data specifically for firms j1 and j2
                    is_j12 = (j_j12 == j1) | (j_j12 == j2)
                    j_j12 = j_j12[is_j12]
                    y_j12 = y_j12[is_j12]
                    if len(j_j12) >= 4:
                        ## If there are at least two workers with observations at both firms ##
                        # Split data for j1 and j2
                        y_j11 = y_j12[j_j12 == j1]
                        y_j22 = y_j12[j_j12 == j2]
                        # Split observations into entering/exiting groups
                        j_j12_first = j_j12[np.arange(len(j_j12)) % 2 == 0]
                        entering = (j_j12_first == j2)
                        exiting = (j_j12_first == j1)
                        if (np.sum(entering) > 0) and (np.sum(exiting) > 0):
                            # Need workers to both enter and exit
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
        evec = np.real(evec.flatten())

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

    def fit_b_linear(self, jdata, norm_fid=0, rng=None):
        '''
        Fit linear estimator for b using LIML.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            norm_fid (int): firm id to normalize
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): estimated b
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Parameters
        nk, ni = jdata.n_clusters(), len(jdata)

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=False)

        ## Sparse matrix representations ##
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        YY1 = csc_matrix((Y1, (range(ni), G1)), shape=(ni, nk))
        YY2 = csc_matrix((Y2, (range(ni), G2)), shape=(ni, nk))

        # Joint firm indicator
        KK = G1 + nk * G2

        # Transition probability matrix (in this case, matrix of instruments (j1, j2))
        GG12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nk ** 2))

        ## Combine matrices ##
        X1 = hstack([YY1, YY2])
        X2 = hstack([GG1, GG2])

        ## Define normalization ##
        Y = -X1[:, norm_fid]
        X1 = X1[:, list(range(norm_fid)) + list(range(norm_fid + 1, 2 * nk))]
        X2 = X2[:, range(2 * nk - 1)]

        ## Combine matrices ##
        R = hstack([Y, X1])
        XX = hstack([X1, X2])

        ## LIML ##
        GGtGG = (GG12.T @ GG12).tocsc()
        RtR = R.T @ R
        RtGG = R.T @ GG12
        RtX2 = R.T @ X2
        Wz = (RtR - RtGG @ inv(GGtGG) @ RtGG.T).tocsc()
        Wx = RtR - RtX2 @ inv((X2.T @ X2).tocsc()) @ RtX2.T
        del RtR, RtGG, RtX2

        # Smallest eigenvalue
        WW = Wx @ inv(Wz)
        evals, evecs = eigs(WW)
        lambda_ = min(evals)

        XXtGG = XX.T @ GG12
        RR = ((1 - lambda_) * XX.T @ XX + lambda_ * XXtGG @ inv(GGtGG) @ XXtGG.T).tocsc()
        RY = (1 - lambda_) * XX.T @ Y + lambda_ * XXtGG @ inv(GGtGG) @ GG12.T @ Y

        ## Extract results ##
        b_liml = np.real(np.asarray((inv(RR) @ RY).todense()).flatten())
        tau = np.ones(nk)
        tau[: norm_fid] = b_liml[: norm_fid]
        tau[norm_fid + 1:] = b_liml[norm_fid: nk - 1]
        B1 = 1 / tau
        B2 = - 1 / b_liml[nk - 1: 2 * nk - 1]
        A1 = - b_liml[2 * nk - 1: 3 * nk - 1] * B1
        A2 = np.zeros(nk)
        A2[: nk - 1] = b_liml[3 * nk - 1: 4 * nk - 2] * B2[: nk - 1]

        return B1, B2

# import bipartitepandas as bpd

# # Instantiate
# model = InteractedBLMModel()

# # Prepare parameters
# n_loops = 100
# B1_err = np.zeros(n_loops)
# B2_err = np.zeros(n_loops)
# evec_err = np.zeros(n_loops)

# for iter in range(n_loops):
#     # Simulate some data
#     n_workers = 10000
#     n_firms = 200
#     nk = 10
#     a = np.random.normal(size=nk)
#     b = np.random.normal(size=nk)
#     # Make sure b values aren't too small
#     b[abs(b) < 0.01] *= 50
#     # Normalize b
#     b = b / b[0]
#     # Link firms to firm types
#     firm_types = np.random.choice(range(nk), size=n_firms, replace=True)
#     # Simulate data
#     i = np.repeat(range(n_workers), 2)
#     j = np.random.choice(range(n_firms), size=2 * n_workers, replace=True)
#     t = np.tile(range(2), n_workers)
#     g = firm_types[j]
#     alpha_i = np.repeat(np.random.normal(size=n_workers), 2)
#     eps_i = 0.1 * np.random.normal(size=2 * n_workers)
#     # Simulate wages
#     y = a[g] + b[g] * alpha_i + eps_i

#     # Prepare data
#     cp = bpd.clean_params({'verbose': False})
#     bdf = bpd.BipartiteDataFrame(i=i, j=j, y=y, t=t, g=g).clean(cp).collapse()
#     jdata = bdf[bdf.get_worker_m()].clean(cp).to_eventstudy()

#     B1, B2 = model.fit_b_linear(jdata)

#     bdf = bpd.BipartiteDataFrame(i=i, j=g, y=y, t=t).clean(cp).collapse()
#     jdata = bdf[bdf.get_worker_m()].clean(cp)

#     evec = model.fit_b_fixed_point(jdata)

#     # if abs(np.mean((evec - b) / b)) > 500:
#     #     stop

#     B1_err[iter] = np.mean((B1 - b) / b)
#     B2_err[iter] = np.mean((B2 - b) / b)
#     evec_err[iter] = np.mean((evec - b) / b)

# # Plot
# from matplotlib import pyplot as plt
# axis = np.arange(n_loops)
# plt.plot(axis, B1_err, label='B1')
# plt.plot(axis, B2_err, label='B2')
# plt.plot(axis, evec_err, label='Fixed point')
# plt.legend()
# plt.show()

# # Histogram
# # Eliminate outliers (so we can actually see things)
# B1_err_nooutliers = B1_err[abs(B1_err) < 20]
# B2_err_nooutliers = B2_err[abs(B2_err) < 20]
# evec_err_nooutliers = evec_err[abs(evec_err) < 20]
# bins = np.arange(-20, 20 + 0.25, 0.25)
# plt.hist(B1_err_nooutliers, alpha=0.75, bins=bins, label='B1')
# plt.hist(B2_err_nooutliers, alpha=0.75, bins=bins, label='B2')
# plt.hist(evec_err_nooutliers, alpha=0.5, bins=bins, label='Fixed point')
# plt.legend()
# plt.show()

# # Kernel density plot
# from scipy.stats import gaussian_kde
# bins = np.arange(-20, 20 + 0.25, 0.25)
# B1_err_nooutliers = B1_err[abs(B1_err) < 100]
# B2_err_nooutliers = B2_err[abs(B2_err) < 100]
# evec_err_nooutliers = evec_err[abs(evec_err) < 100]
# density_B1 = gaussian_kde(B1_err_nooutliers)
# # density_B1.covariance_factor = lambda : .01
# # density_B1._compute_covariance()
# density_B2 = gaussian_kde(B2_err_nooutliers)
# # density_B2.covariance_factor = lambda : .01
# # density_B2._compute_covariance()
# density_evec = gaussian_kde(evec_err_nooutliers)
# # density_evec.covariance_factor = lambda : .01
# # density_evec._compute_covariance()
# plt.plot(bins, density_B1(bins), label='B1')
# plt.plot(bins, density_B2(bins), label='B2')
# plt.plot(bins, density_evec(bins), label='Fixed point')
# plt.legend()
# plt.show()
