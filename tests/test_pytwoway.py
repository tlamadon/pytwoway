'''
Tests for pytwoway
'''
'''
TODO:
    -Check that for FE, uncollapsed and collapsed estimators give identical results
    -Update tests that stopped working
'''
import pytest
import copy
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw
from pytwoway import constraints as cons
from scipy.sparse import csc_matrix

##############
##### FE #####
##############

def test_fe_ols():
    # Test that OLS is estimating fixed effects correctly.
    worker_data = []
    # alpha0 = 3, psi0 = 5 --> 8
    worker_data.append({'i': 0, 'j': 0, 'y': 8, 't': 1})
    # alpha0 = 3, psi1 = 3 --> 6
    worker_data.append({'i': 0, 'j': 1, 'y': 6, 't': 2})
    # alpha1 = 2, psi1 = 3 --> 5
    worker_data.append({'i': 1, 'j': 1, 'y': 5, 't': 1})
    # alpha1 = 2, psi3 = 4 --> 6
    worker_data.append({'i': 1, 'j': 3, 'y': 6, 't': 2})
    # alpha2 = 4, psi3 = 4 --> 8
    worker_data.append({'i': 2, 'j': 3, 'y': 8, 't': 1})
    # alpha2 = 4, psi3 = 4 --> 8
    worker_data.append({'i': 2, 'j': 3, 'y': 8, 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteDataFrame(df, log=False).clean(bpd.clean_params({'verbose': False})).collapse()

    fe_params = tw.fe_params({'feonly': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEEstimator(bdf, fe_params)
    fe_solver.fit()

    assert np.all(np.isclose(bdf.loc[:, 'psi_hat'].to_numpy() + bdf.loc[:, 'alpha_hat'].to_numpy(), bdf.loc[:, 'y'].to_numpy()))

def test_fe_estimator_full_novar():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE with var(eps) = 0.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 0})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1234))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEEstimator(b, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    # True parameters
    true_sigma_2 = np.var(a.loc[:, 'y'] - a.loc[:, 'psi'].to_numpy() - a.loc[:, 'alpha'].to_numpy(), ddof=0)
    true_var_psi = np.var(a.loc[:, 'psi'].to_numpy(), ddof=0)
    true_cov_psi_alpha = np.cov(a.loc[:, 'psi'].to_numpy(), a.loc[:, 'alpha'].to_numpy(), ddof=0)[0, 1]

    # Estimated parameters
    ## Plug-in ##
    est_pi_sigma_2 = fe_solver.sigma_2_pi
    est_pi_var_psi = fe_solver.var_fe
    est_pi_cov_psi_alpha = fe_solver.cov_fe
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['eps_var_ho']
    est_ho_var_psi = fe_solver.res['var_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['eps_var_he']
    est_he_var_psi = fe_solver.res['var_he']
    est_he_cov_psi_alpha = fe_solver.res['cov_he']

    # sigma^2
    assert np.isclose(true_sigma_2, est_pi_sigma_2)
    assert np.isclose(true_sigma_2, est_ho_sigma_2)
    assert np.isclose(true_sigma_2, est_he_sigma_2)
    # var(psi)
    assert np.isclose(true_var_psi, est_pi_var_psi)
    assert np.isclose(true_var_psi, est_ho_var_psi)
    assert np.isclose(true_var_psi, est_he_var_psi)
    # cov(psi, alpha)
    assert np.isclose(true_cov_psi_alpha, est_pi_cov_psi_alpha)
    assert np.isclose(true_cov_psi_alpha, est_ho_cov_psi_alpha)
    assert np.isclose(true_cov_psi_alpha, est_he_cov_psi_alpha)

    # y
    assert np.all(np.isclose(b.loc[:, 'psi_hat'].to_numpy() + b.loc[:, 'alpha_hat'].to_numpy(), b.loc[:, 'y'].to_numpy()))

def test_fe_estimator_full_var_uncollapsed():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE with var(eps) = 1 for uncollapsed data.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1235))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a # .collapse()

    fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEEstimator(b, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    # True parameters
    true_sigma_2 = np.var(a.loc[:, 'y'] - a.loc[:, 'psi'].to_numpy() - a.loc[:, 'alpha'].to_numpy(), ddof=0)
    true_var_psi = np.var(a.loc[:, 'psi'].to_numpy(), ddof=0)
    true_cov_psi_alpha = np.cov(a.loc[:, 'psi'].to_numpy(), a.loc[:, 'alpha'].to_numpy(), ddof=0)[0, 1]

    # Estimated parameters
    ## Plug-in ##
    est_pi_sigma_2 = fe_solver.sigma_2_pi
    est_pi_var_psi = fe_solver.var_fe
    est_pi_cov_psi_alpha = fe_solver.cov_fe
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['eps_var_ho']
    est_ho_var_psi = fe_solver.res['var_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['eps_var_he']
    est_he_var_psi = fe_solver.res['var_he']
    est_he_cov_psi_alpha = fe_solver.res['cov_he']

    # sigma^2
    # assert np.abs((est_pi_sigma_2 - true_sigma_2) / true_sigma_2) < 1e-2
    assert np.abs((est_ho_sigma_2 - true_sigma_2) / true_sigma_2) < 1e-2
    assert np.abs((est_he_sigma_2 - true_sigma_2) / true_sigma_2) < 1e-2
    # var(psi)
    # assert np.abs((est_pi_var_psi - true_var_psi) / true_var_psi) < 1e-2
    assert np.abs((est_ho_var_psi - true_var_psi) / true_var_psi) < 0.025
    assert np.abs((est_he_var_psi - true_var_psi) / true_var_psi) < 0.025
    # cov(psi, alpha)
    # assert np.abs((est_pi_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-2
    assert np.abs((est_ho_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-2
    assert np.abs((est_he_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-2

    # y
    assert np.sum(np.isclose(b.loc[:, 'psi_hat'].to_numpy() + b.loc[:, 'alpha_hat'].to_numpy(), b.loc[:, 'y'].to_numpy(), atol=1)) / len(b) > 0.75

def test_fe_estimator_full_var_collapsed():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE with var(eps) = 1 for collapsed data.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEEstimator(b, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    # True parameters
    true_sigma_2 = np.var(a.loc[:, 'y'] - a.loc[:, 'psi'].to_numpy() - a.loc[:, 'alpha'].to_numpy(), ddof=0)
    true_var_psi = np.var(a.loc[:, 'psi'].to_numpy(), ddof=0)
    true_cov_psi_alpha = np.cov(a.loc[:, 'psi'].to_numpy(), a.loc[:, 'alpha'].to_numpy(), ddof=0)[0, 1]

    # Estimated parameters
    ## Plug-in ##
    est_pi_sigma_2 = fe_solver.sigma_2_pi
    est_pi_var_psi = fe_solver.var_fe
    est_pi_cov_psi_alpha = fe_solver.cov_fe
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['eps_var_ho']
    est_ho_var_psi = fe_solver.res['var_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['eps_var_he']
    est_he_var_psi = fe_solver.res['var_he']
    est_he_cov_psi_alpha = fe_solver.res['cov_he']

    # sigma^2
    # assert np.abs((est_pi_sigma_2 - true_sigma_2) / true_sigma_2) < 1e-2
    assert np.abs((est_ho_sigma_2 - true_sigma_2) / true_sigma_2) < 0.02
    assert np.abs((est_he_sigma_2 - true_sigma_2) / true_sigma_2) < 0.055
    # var(psi)
    # assert np.abs((est_pi_var_psi - true_var_psi) / true_var_psi) < 1e-2
    assert np.abs((est_ho_var_psi - true_var_psi) / true_var_psi) < 0.15
    assert np.abs((est_he_var_psi - true_var_psi) / true_var_psi) < 0.15
    # cov(psi, alpha)
    # assert np.abs((est_pi_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-2
    assert np.abs((est_ho_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 0.05
    assert np.abs((est_he_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 0.05

    # y
    assert np.sum(np.isclose(b.loc[:, 'psi_hat'].to_numpy() + b.loc[:, 'alpha_hat'].to_numpy(), b.loc[:, 'y'].to_numpy(), atol=1)) / len(b) > 0.8

def test_fe_estimator_full_approx_analytical_non_collapsed():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE for analytical vs. approximate estimators, with non-collapsed data.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))

    fe_params_a = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True})
    fe_solver_a = tw.FEEstimator(a, fe_params_a)
    fe_solver_a.fit(np.random.default_rng(1237))
    fe_params_b = tw.fe_params({'he': True, 'exact_trace_sigma_2': False, 'exact_trace_ho': False, 'exact_trace_he': False, 'exact_lev_he': False})
    fe_solver_b = tw.FEEstimator(a, fe_params_b)
    fe_solver_b.fit(np.random.default_rng(1238))

    ### Analytical ###
    ## Plug-in ##
    est_pi_sigma_2_a = fe_solver_a.sigma_2_pi
    est_pi_var_psi_a = fe_solver_a.var_fe
    est_pi_cov_psi_alpha_a = fe_solver_a.cov_fe
    ## HO ##
    est_ho_sigma_2_a = fe_solver_a.res['eps_var_ho']
    est_ho_var_psi_a = fe_solver_a.res['var_ho']
    est_ho_cov_psi_alpha_a = fe_solver_a.res['cov_ho']
    ## HE ##
    est_he_sigma_2_a = fe_solver_a.res['eps_var_he']
    est_he_var_psi_a = fe_solver_a.res['var_he']
    est_he_cov_psi_alpha_a = fe_solver_a.res['cov_he']
    ### Approximate ###
    ## Plug-in ##
    est_pi_sigma_2_b = fe_solver_b.sigma_2_pi
    est_pi_var_psi_b = fe_solver_b.var_fe
    est_pi_cov_psi_alpha_b = fe_solver_b.cov_fe
    ## HO ##
    est_ho_sigma_2_b = fe_solver_b.res['eps_var_ho']
    est_ho_var_psi_b = fe_solver_b.res['var_ho']
    est_ho_cov_psi_alpha_b = fe_solver_b.res['cov_ho']
    ## HE ##
    est_he_sigma_2_b = fe_solver_b.res['eps_var_he']
    est_he_var_psi_b = fe_solver_b.res['var_he']
    est_he_cov_psi_alpha_b = fe_solver_b.res['cov_he']

    # sigma^2
    assert np.abs((est_pi_sigma_2_b - est_pi_sigma_2_a) / est_pi_sigma_2_a) == 0
    assert np.abs((est_ho_sigma_2_b - est_ho_sigma_2_a) / est_ho_sigma_2_a) == 0
    assert np.abs((est_he_sigma_2_b - est_he_sigma_2_a) / est_he_sigma_2_a) < 1e-2
    # var(psi)
    assert np.abs((est_pi_var_psi_b - est_pi_var_psi_a) / est_pi_var_psi_a) == 0
    assert np.abs((est_ho_var_psi_b - est_ho_var_psi_a) / est_ho_var_psi_a) < 1e-2
    assert np.abs((est_he_var_psi_b - est_he_var_psi_a) / est_he_var_psi_a) < 1e-2
    # cov(psi, alpha)
    assert np.abs((est_pi_cov_psi_alpha_b - est_pi_cov_psi_alpha_a) / est_pi_cov_psi_alpha_a) == 0
    assert np.abs((est_ho_cov_psi_alpha_b - est_ho_cov_psi_alpha_a) / est_ho_cov_psi_alpha_a) < 1e-2
    assert np.abs((est_he_cov_psi_alpha_b - est_he_cov_psi_alpha_a) / est_he_cov_psi_alpha_a) < 0.015

def test_fe_estimator_full_approx_analytical_collapsed():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE for analytical vs. approximate estimators, with collapsed data.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False})).collapse()

    fe_params_a = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True})
    fe_solver_a = tw.FEEstimator(a, fe_params_a)
    fe_solver_a.fit(np.random.default_rng(1237))
    fe_params_b = tw.fe_params({'he': True, 'exact_trace_sigma_2': False, 'exact_trace_ho': False, 'exact_trace_he': False, 'exact_lev_he': False})
    fe_solver_b = tw.FEEstimator(a, fe_params_b)
    fe_solver_b.fit(np.random.default_rng(1238))

    ### Analytical ###
    ## Plug-in ##
    est_pi_sigma_2_a = fe_solver_a.sigma_2_pi
    est_pi_var_psi_a = fe_solver_a.var_fe
    est_pi_cov_psi_alpha_a = fe_solver_a.cov_fe
    ## HO ##
    est_ho_sigma_2_a = fe_solver_a.res['eps_var_ho']
    est_ho_var_psi_a = fe_solver_a.res['var_ho']
    est_ho_cov_psi_alpha_a = fe_solver_a.res['cov_ho']
    ## HE ##
    est_he_sigma_2_a = fe_solver_a.res['eps_var_he']
    est_he_var_psi_a = fe_solver_a.res['var_he']
    est_he_cov_psi_alpha_a = fe_solver_a.res['cov_he']
    ### Approximate ###
    ## Plug-in ##
    est_pi_sigma_2_b = fe_solver_b.sigma_2_pi
    est_pi_var_psi_b = fe_solver_b.var_fe
    est_pi_cov_psi_alpha_b = fe_solver_b.cov_fe
    ## HO ##
    est_ho_sigma_2_b = fe_solver_b.res['eps_var_ho']
    est_ho_var_psi_b = fe_solver_b.res['var_ho']
    est_ho_cov_psi_alpha_b = fe_solver_b.res['cov_ho']
    ## HE ##
    est_he_sigma_2_b = fe_solver_b.res['eps_var_he']
    est_he_var_psi_b = fe_solver_b.res['var_he']
    est_he_cov_psi_alpha_b = fe_solver_b.res['cov_he']

    # sigma^2
    assert np.abs((est_pi_sigma_2_b - est_pi_sigma_2_a) / est_pi_sigma_2_a) == 0
    assert np.abs((est_ho_sigma_2_b - est_ho_sigma_2_a) / est_ho_sigma_2_a) < 1e-3
    assert np.abs((est_he_sigma_2_b - est_he_sigma_2_a) / est_he_sigma_2_a) < 0.025
    # var(psi)
    assert np.abs((est_pi_var_psi_b - est_pi_var_psi_a) / est_pi_var_psi_a) == 0
    assert np.abs((est_ho_var_psi_b - est_ho_var_psi_a) / est_ho_var_psi_a) < 1e-3
    assert np.abs((est_he_var_psi_b - est_he_var_psi_a) / est_he_var_psi_a) < 1e-2
    # cov(psi, alpha)
    assert np.abs((est_pi_cov_psi_alpha_b - est_pi_cov_psi_alpha_a) / est_pi_cov_psi_alpha_a) == 0
    assert np.abs((est_ho_cov_psi_alpha_b - est_ho_cov_psi_alpha_a) / est_ho_cov_psi_alpha_a) < 1e-2
    assert np.abs((est_he_cov_psi_alpha_b - est_he_cov_psi_alpha_a) / est_he_cov_psi_alpha_a) < 1e-2

def test_fe_estimator_full_Q():
    # Test that FE estimates custom Q correctly for plug-in, HO, and HE estimators for Q.VarAlpha() and Q.CovPsiPrevPsiNext().
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 0})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True, 'attach_fe_estimates': True, 'Q_var': tw.Q.VarAlpha(), 'Q_cov': tw.Q.CovPsiPrevPsiNext()})
    fe_solver = tw.FEEstimator(b, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    # True parameters
    # psi, alpha
    psi = b.loc[:, 'psi'].to_numpy()
    alpha = b.loc[:, 'alpha'].to_numpy()
    true_var_psi = np.var(b.loc[:, 'psi'].to_numpy(), ddof=0)
    true_var_alpha = np.var(b.loc[:, 'alpha'].to_numpy(), ddof=0)
    # Get i for this, last, and next period
    i_col = b.loc[:, 'i'].to_numpy()
    i_prev = bpd.util.fast_shift(i_col, 1, fill_value=-2)
    i_next = bpd.util.fast_shift(i_col, -1, fill_value=-2)
    ## Drop the first/last observation for each worker ##
    # Drop the last observation for each worker
    prev_rows = (i_col == i_next)
    # Drop the first observation for each worker
    next_rows = (i_col == i_prev)
    # Weights
    w_col = b.loc[:, 'w'].to_numpy()
    true_cov_psi_prev_psi_next = tw.fe._weighted_cov(psi[prev_rows], psi[next_rows], w_col[prev_rows], w_col[next_rows])
    true_cov_psi_alpha = tw.fe._weighted_cov(psi, alpha, w_col, w_col)

    # Estimated parameters
    ## Plug-in ##
    est_pi_var_alpha = fe_solver.var_fe
    est_pi_cov_psi_prev_psi_next = fe_solver.cov_fe
    ## HO ##
    est_ho_var_alpha = fe_solver.res['var_ho']
    est_ho_cov_psi_prev_psi_next = fe_solver.res['cov_ho']
    ## HE ##
    est_he_var_alpha = fe_solver.res['var_he']
    est_he_cov_psi_prev_psi_next = fe_solver.res['cov_he']

    # var(alpha)
    assert np.abs((est_pi_var_alpha - true_var_alpha) / true_var_alpha) < 1e-2
    assert np.abs((est_ho_var_alpha - true_var_alpha) / true_var_alpha) < 1e-2
    assert np.abs((est_he_var_alpha - true_var_alpha) / true_var_alpha) < 1e-2
    # cov(psi_t, psi_{t+1})
    assert np.abs((est_pi_cov_psi_prev_psi_next - true_cov_psi_prev_psi_next) / true_cov_psi_prev_psi_next) < 1e-10
    assert np.abs((est_ho_cov_psi_prev_psi_next - true_cov_psi_prev_psi_next) / true_cov_psi_prev_psi_next) < 1e-10
    assert np.abs((est_he_cov_psi_prev_psi_next - true_cov_psi_prev_psi_next) / true_cov_psi_prev_psi_next) < 1e-10
    # Make sure cov(psi_t, psi_{t+1}) isn't just similar to var(psi) or cov(psi, alpha)
    assert np.abs((true_var_psi - true_cov_psi_prev_psi_next) / true_cov_psi_prev_psi_next) > 0.8
    assert np.abs((true_cov_psi_alpha - true_cov_psi_prev_psi_next) / true_cov_psi_prev_psi_next) > 0.05

# def test_fe_estimator_full_Q_2():
#     # Test that FE estimates custom Q correctly for plug-in, HO, and HE estimators for Q.VarGamma().
#     sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 0})
#     a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
#     a = bpd.BipartiteDataFrame(a, log=False).clean()
#     b = a.collapse()

#     fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': False, 'exact_trace_he': False, 'exact_lev_he': True, 'attach_fe_estimates': True, 'Q_var': tw.Q.VarGamma(), 'Q_cov': tw.Q.CovPsiAlpha()})
#     fe_solver = tw.FEEstimator(b, fe_params)
#     fe_solver.fit(np.random.default_rng(1234))

#     # True parameters
#     # psi, alpha
#     true_var_gamma = np.var(a.loc[:, ['psi', 'alpha']].to_numpy(), ddof=0)
#     true_var_psi = np.var(a.loc[:, 'psi'].to_numpy(), ddof=0)
#     true_var_alpha = np.var(a.loc[:, 'alpha'].to_numpy(), ddof=0)

#     # Estimated parameters
#     ## Plug-in ##
#     est_pi_var_gamma = fe_solver.var_fe
#     ## HO ##
#     est_ho_var_gamma = fe_solver.res['var_ho']
#     ## HE ##
#     est_he_var_gamma = fe_solver.res['var_he']

#     # var(gamma)
#     assert np.abs((est_pi_var_gamma - true_var_gamma) / true_var_gamma) < 1e-2
#     assert np.abs((est_ho_var_gamma - true_var_gamma) / true_var_gamma) < 1e-2
#     assert np.abs((est_he_var_gamma - true_var_gamma) / true_var_gamma) < 1e-2
#     # Make sure var(gamma) isn't just similar to var(psi) or var(alpha)
#     assert np.abs((true_var_psi - true_var_gamma) / true_var_gamma) > 0.9
#     assert np.abs((true_var_alpha - true_var_gamma) / true_var_gamma) > 0.05

def test_fe_Pii():
    # Test that HE Pii are equivalent when computed using M^{-1} explicitly or computing each observation one at a time using multi-grid solver
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1239))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False})).collapse()

    fe_params_a = tw.fe_params({'ho': False, 'he': True, 'exact_trace_he': True, 'exact_lev_he': True})
    fe_solver_a = tw.FEEstimator(a, fe_params_a)
    fe_params_b = tw.fe_params({'ho': False, 'he': True, 'exact_trace_he': False, 'exact_lev_he': False})
    fe_solver_b = tw.FEEstimator(a, fe_params_b)

    # Run estimators
    ## Analytical ##
    fe_solver_a._prep_vars()
    fe_solver_a._prep_matrices()
    fe_solver_a._construct_AAinv_components_full()
    Pii_a, _ = fe_solver_a._estimate_exact_leverages()
    ## Approximate ##
    fe_solver_b._prep_vars()
    fe_solver_b._prep_matrices()
    fe_solver_b._construct_AAinv_components_partial()
    Pii_b, _ = fe_solver_b._estimate_exact_leverages()

    # Don't take percent because some values equal 0
    assert np.sum(np.abs(Pii_b - Pii_a)) < 1e-4

def test_fe_weights():
    # Test that FE estimates are identical for weighted and unweighted (estimate parameters sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO and HE.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1240))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': True, 'exact_trace_he': True, 'exact_lev_he': True, 'attach_fe_estimates': True})
    fe_solver_a = tw.FEEstimator(a, fe_params)
    fe_solver_a.fit(np.random.default_rng(1241))
    fe_solver_b = tw.FEEstimator(b, fe_params)
    fe_solver_b.fit(np.random.default_rng(1242))

    # Estimated parameters
    ### Unweighted ###
    ## Plug-in ##
    est_pi_sigma_2_a = fe_solver_a.sigma_2_pi
    est_pi_var_psi_a = fe_solver_a.var_fe
    est_pi_cov_psi_alpha_a = fe_solver_a.cov_fe
    ## HO ##
    est_ho_sigma_2_a = fe_solver_a.res['eps_var_ho']
    est_ho_var_psi_a = fe_solver_a.res['var_ho']
    est_ho_cov_psi_alpha_a = fe_solver_a.res['cov_ho']
    ## HE ##
    est_he_sigma_2_a = fe_solver_a.res['eps_var_he']
    est_he_var_psi_a = fe_solver_a.res['var_he']
    est_he_cov_psi_alpha_a = fe_solver_a.res['cov_he']
    ### Weighted ###
    ## Plug-in ##
    est_pi_sigma_2_b = fe_solver_b.sigma_2_pi
    est_pi_var_psi_b = fe_solver_b.var_fe
    est_pi_cov_psi_alpha_b = fe_solver_b.cov_fe
    ## HO ##
    est_ho_sigma_2_b = fe_solver_b.res['eps_var_ho']
    est_ho_var_psi_b = fe_solver_b.res['var_ho']
    est_ho_cov_psi_alpha_b = fe_solver_b.res['cov_ho']
    ## HE ##
    est_he_sigma_2_b = fe_solver_b.res['eps_var_he']
    est_he_var_psi_b = fe_solver_b.res['var_he']
    est_he_cov_psi_alpha_b = fe_solver_b.res['cov_he']

    # psi_hat, alpha_hat
    assert np.all(np.isclose(fe_solver_a.psi_hat, fe_solver_b.psi_hat))
    assert np.all(np.isclose(fe_solver_a.alpha_hat, fe_solver_b.alpha_hat))
    # Plug-in
    # assert np.abs((est_pi_sigma_2_b - est_pi_sigma_2_a) / est_pi_sigma_2_a) < 0.4
    assert np.isclose(est_pi_var_psi_a, est_pi_var_psi_b)
    assert np.isclose(est_pi_cov_psi_alpha_a, est_pi_cov_psi_alpha_b)
    # HO
    assert np.abs((est_ho_sigma_2_b - est_ho_sigma_2_a) / est_ho_sigma_2_a) < 0.025
    assert np.abs((est_ho_var_psi_b - est_ho_var_psi_a) / est_ho_var_psi_a) < 1e-3
    assert np.abs((est_ho_cov_psi_alpha_b - est_ho_cov_psi_alpha_a) / est_ho_cov_psi_alpha_a) < 1e-3
    # HE
    assert np.abs((est_he_sigma_2_b - est_he_sigma_2_a) / est_he_sigma_2_a) < 0.25
    assert np.abs((est_he_var_psi_b - est_he_var_psi_a) / est_he_var_psi_a) < 1e-2
    assert np.abs((est_he_cov_psi_alpha_b - est_he_cov_psi_alpha_a) / est_he_cov_psi_alpha_a) < 1e-2

#######################
##### Monte Carlo #####
#######################

def test_monte_carlo():
    # Use Monte Carlo to test CRE, FE, FE-HO, and FE-HE estimators.
    twmc_net = tw.MonteCarlo()
    twmc_net.monte_carlo(N=50, ncore=4, rng=np.random.default_rng(1240))

    # Extract results
    true_psi_var = twmc_net.res['true_psi_var']
    true_psi_alpha_cov = twmc_net.res['true_psi_alpha_cov']
    cre_psi_var = twmc_net.res['cre_psi_var']
    cre_psi_alpha_cov = twmc_net.res['cre_psi_alpha_cov']
    fe_psi_var = twmc_net.res['fe_psi_var']
    fe_psi_alpha_cov = twmc_net.res['fe_psi_alpha_cov']
    ho_psi_var = twmc_net.res['ho_psi_var']
    ho_psi_alpha_cov = twmc_net.res['ho_psi_alpha_cov']
    he_psi_var = twmc_net.res['he_psi_var']
    he_psi_alpha_cov = twmc_net.res['he_psi_alpha_cov']

    # Compute mean percent differences from truth
    cre_psi_diff = abs(np.mean((cre_psi_var - true_psi_var) / true_psi_var))
    cre_psi_alpha_diff = abs(np.mean((cre_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    fe_psi_diff = abs(np.mean((fe_psi_var - true_psi_var) / true_psi_var))
    fe_psi_alpha_diff = abs(np.mean((fe_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    ho_psi_diff = abs(np.mean((ho_psi_var - true_psi_var) / true_psi_var))
    ho_psi_alpha_diff = abs(np.mean((ho_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    he_psi_diff = abs(np.mean((he_psi_var - true_psi_var) / true_psi_var))
    he_psi_alpha_diff = abs(np.mean((he_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))

    assert cre_psi_diff < 0.03
    assert cre_psi_alpha_diff < 1e-2
    assert fe_psi_diff < 0.02
    assert fe_psi_alpha_diff < 0.015
    assert ho_psi_diff < 1e-2
    assert ho_psi_alpha_diff < 1e-3
    assert he_psi_diff < 1e-2
    assert he_psi_alpha_diff < 1e-3

#####################
##### Attrition #####
#####################

# def test_attrition_increasing_1():
#     # Test attrition_increasing() and attrition_decreasing(), by checking that the fraction of movers remaining is approximately equal to what `subsets` specifies.

#     # First test increasing, then decreasing
#     subsets_lst = [np.linspace(0.1, 0.5, 5), np.linspace(0.5, 0.1, 5)]
#     attrition_fn = [tw.attrition.attrition_increasing, tw.attrition.attrition_decreasing]

#     for i in range(2):
#         # Non-collapsed
#         bdf = bpd.BipartiteLong(bpd.SimBipartite({'seed': 1234}).sim_network(), track_id_changes=True).clean_data().get_es()

#         orig_n_movers = len(bdf.loc[bdf['m'] == 1, 'i'].unique())
#         n_movers = []

#         for j in attrition_fn[i](bdf, subsets_lst[i], rng=np.random.default_rng(1234)):
#             n_movers.append(len(j.loc[j['m'] == 1, 'i'].unique()))

#         n_movers_vs_subsets = abs((np.array(n_movers) / orig_n_movers) - subsets_lst[i])

#         assert np.max(n_movers_vs_subsets) < 2e-4

#         # Collapsed
#         bdf = bpd.BipartiteLong(bpd.SimBipartite({'seed': 1234}).sim_network(), track_id_changes=True).clean_data().get_collapsed_long().get_es()

#         orig_n_movers = len(bdf.loc[bdf['m'] == 1, 'i'].unique())
#         n_movers = []

#         for j in attrition_fn[i](bdf, subsets_lst[i], rng=np.random.default_rng(1234)):
#             n_movers.append(len(j.loc[j['m'] == 1, 'i'].unique()))

#         n_movers_vs_subsets = abs((np.array(n_movers) / orig_n_movers) - subsets_lst[i])

#         assert np.max(n_movers_vs_subsets) < 2e-4

###############
##### BLM #####
###############

def test_blm_monotonic_1():
    # Test whether BLM likelihoods are monotonic, using default fit.
    rng = np.random.default_rng(1234)
    ## Set parameters ##
    nl = 3
    nk = 4
    n_control = 2
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True
    })
    sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False
    })
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cts_tv_wi_params = tw.sim_continuous_control_params({
        'worker_type_interaction': True
    })
    sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
        'worker_type_interaction': True,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cts_tv_params = tw.sim_continuous_control_params({
        'worker_type_interaction': False
    })
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        'worker_type_interaction': False,
        'stationary_A': True, 'stationary_S': True
    })
    blm_sim_params = tw.sim_params({
        'nl': nl,
        'nk': nk,
        'firm_size': 10,
        'NNm': np.ones(shape=(nk, nk)).astype(int, copy=False),
        'NNs': np.ones(shape=nk).astype(int, copy=False),
        'mmult': 1000, 'smult': 1000,
        'a1_sig': 1, 'a2_sig': 1, 's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {
            'cat_tv_wi_control': sim_cat_tv_wi_params,
            'cat_tnv_wi_control': sim_cat_tnv_wi_params,
            'cat_tv_control': sim_cat_tv_params,
            'cat_tnv_control': sim_cat_tnv_params
        },
        'continuous_controls': {
            'cts_tv_wi_control': sim_cts_tv_wi_params,
            'cts_tnv_wi_control': sim_cts_tnv_wi_params,
            'cts_tv_control': sim_cts_tv_params,
            'cts_tnv_control': sim_cts_tnv_params
        }
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True
    })
    cat_tnv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False
    })
    cat_tnv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    cts_tv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True
    })
    cts_tnv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    cts_tv_params = tw.continuous_control_params({
        'worker_type_interaction': False
    })
    cts_tnv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    blm_params = tw.blm_params({
        'nl': nl,
        'nk': nk,
        'n_iters_movers': 150,
        'threshold_movers': 1e-6,
        'categorical_controls': {
            'cat_tv_wi_control': cat_tv_wi_params,
            'cat_tnv_wi_control': cat_tnv_wi_params,
            'cat_tv_control': cat_tv_params,
            'cat_tnv_control': cat_tnv_params
        },
        'continuous_controls': {
            'cts_tv_wi_control': cts_tv_wi_params,
            'cts_tnv_wi_control': cts_tnv_wi_params,
            'cts_tv_control': cts_tv_params,
            'cts_tnv_control': cts_tnv_params
        }
    })
    ## Simulate data ##
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM model
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Fit BLM model
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.min(np.diff(blm_fit.liks1)) > 0
    assert np.min(np.diff(blm_fit.liks0)) > 0

# NOTE: this is commented out because it takes so long to run
# def test_blm_monotonic_2():
#     # Test whether BLM likelihoods are monotonic, using constrained-unconstrained fit.
#     rng = np.random.default_rng(1235)
#     ## Set parameters ##
#     nl = 3
#     nk = 4
#     n_control = 2
#     sim_cat_tv_wi_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True
#     })
#     sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True,
#         'stationary_A': True, 'stationary_S': True
#     })
#     sim_cat_tv_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False
#     })
#     sim_cat_tnv_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False,
#         'stationary_A': True, 'stationary_S': True
#     })
#     sim_cts_tv_wi_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': True
#     })
#     sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': True,
#         'stationary_A': True, 'stationary_S': True
#     })
#     sim_cts_tv_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': False
#     })
#     sim_cts_tnv_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': False,
#         'stationary_A': True, 'stationary_S': True
#     })
#     blm_sim_params = tw.sim_params({
#         'nl': nl,
#         'nk': nk,
#         'firm_size': 10,
#         'NNm': np.ones(shape=(nk, nk)).astype(int, copy=False),
#         'NNs': np.ones(shape=nk).astype(int, copy=False),
#         'mmult': 1000, 'smult': 1000,
#         'a1_sig': 1, 'a2_sig': 1, 's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
#         'categorical_controls': {
#             'cat_tv_wi_control': sim_cat_tv_wi_params,
#             'cat_tnv_wi_control': sim_cat_tnv_wi_params,
#             'cat_tv_control': sim_cat_tv_params,
#             'cat_tnv_control': sim_cat_tnv_params
#         },
#         'continuous_controls': {
#             'cts_tv_wi_control': sim_cts_tv_wi_params,
#             'cts_tnv_wi_control': sim_cts_tnv_wi_params,
#             'cts_tv_control': sim_cts_tv_params,
#             'cts_tnv_control': sim_cts_tnv_params
#         }
#     })
#     cat_tv_wi_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True
#     })
#     cat_tnv_wi_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     cat_tv_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False
#     })
#     cat_tnv_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     cts_tv_wi_params = tw.continuous_control_params({
#         'worker_type_interaction': True
#     })
#     cts_tnv_wi_params = tw.continuous_control_params({
#         'worker_type_interaction': True,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     cts_tv_params = tw.continuous_control_params({
#         'worker_type_interaction': False
#     })
#     cts_tnv_params = tw.continuous_control_params({
#         'worker_type_interaction': False,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     blm_params = tw.blm_params({
#         'nl': nl,
#         'nk': nk,
#         'n_iters_movers': 150,
#         'categorical_controls': {
#             'cat_tv_wi_control': cat_tv_wi_params,
#             'cat_tnv_wi_control': cat_tnv_wi_params,
#             'cat_tv_control': cat_tv_params,
#             'cat_tnv_control': cat_tnv_params
#         },
#         'continuous_controls': {
#             'cts_tv_wi_control': cts_tv_wi_params,
#             'cts_tnv_wi_control': cts_tnv_wi_params,
#             'cts_tv_control': cts_tv_params,
#             'cts_tnv_control': cts_tnv_params
#         }
#     })
#     ## Simulate data ##
#     blm_true = tw.SimBLM(blm_sim_params)
#     sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
#     jdata, sdata = sim_data['jdata'], sim_data['sdata']
#     jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
#     sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
#     # Initialize BLM estimator
#     blm_fit = tw.BLMEstimator(blm_params)
#     # Fit BLM estimator
#     blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)

#     assert np.min(np.diff(blm_fit.model.liks1)[:83]) > 0
#     assert np.min(np.diff(blm_fit.model.liks0)) > 0

def test_blm_qi():
    # Test whether BLM posterior probabilities are giving the most weight to the correct type.
    rng = np.random.default_rng(1234)
    nl = 3
    nk = 4
    # Define parameter dictionaries
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 1, 's2_low': 0, 's2_high': 1
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'return_qi': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 1, 's2_low': 0, 's2_high': 1
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata = sim_data['jdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    # Estimate qi matrix
    qi_estimate = blm_fit._fit_movers(jdata=jdata)
    max_qi_col = np.argmax(qi_estimate, axis=1)
    n_correct_qi = np.sum(max_qi_col == jdata['l'])

    assert (n_correct_qi / len(max_qi_col)) >= 0.95

def test_blm_start_at_truth_no_controls():
    # Test whether BLM estimator works when starting at truth with no controls.
    rng = np.random.default_rng(1234)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.max(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) < 0.045
    assert np.max(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) < 0.03
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.03
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.045

def test_blm_full_estimation_no_controls():
    # Test whether BLM estimator works for full estimation with no controls.
    rng = np.random.default_rng(1235)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.max(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) < 0.02
    assert np.max(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) < 0.025
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.02
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cat_tv_wi():
    # Test whether BLM estimator works when starting at truth for categorical, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1236)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Normalize
    blm_fit.A1 = (blm_fit.A1.T - sim_params['A1'][:, 0]).T
    blm_fit.A2 = (blm_fit.A2.T - sim_params['A2'][:, 0]).T
    blm_fit.A1_cat['cat_tv_wi_control'] = (blm_fit.A1_cat['cat_tv_wi_control'].T + sim_params['A1'][:, 0]).T
    blm_fit.A2_cat['cat_tv_wi_control'] = (blm_fit.A2_cat['cat_tv_wi_control'].T + sim_params['A2'][:, 0]).T
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 1] ** 2)

    # Normalization alters worker type order - adjust estimates to make comparable to simulated parameters
    adj_order = [1, 0]
    A1_sum_0_fit = A1_sum_0_fit[:, adj_order]
    A1_sum_1_fit = A1_sum_1_fit[:, adj_order]
    A2_sum_0_fit = A2_sum_0_fit[:, adj_order]
    A2_sum_1_fit = A2_sum_1_fit[:, adj_order]
    S1_sum_0_fit = S1_sum_0_fit[:, adj_order]
    S1_sum_1_fit = S1_sum_1_fit[:, adj_order]
    S2_sum_0_fit = S2_sum_0_fit[:, adj_order]
    S2_sum_1_fit = S2_sum_1_fit[:, adj_order]
    sorted_pk1 = np.reshape(np.reshape(blm_fit.pk1, (nk, nk, nl))[:, :, adj_order], (nk * nk, nl))
    sorted_pk0 = blm_fit.pk0[:, adj_order]

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    # FIXME figure out why this is so high (it seems like it's just the seed)
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 0.05
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.6
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.65
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.6
    assert np.prod(np.abs((sorted_pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.03
    assert np.prod(np.abs((sorted_pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.025

def test_blm_full_estimation_cat_tv_wi():
    # Test whether BLM estimator works for full estimation for categorical, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1237)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = (sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 0])
    A1_sum_1_sim = (sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 1])
    A2_sum_0_sim = (sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 0])
    A2_sum_1_sim = (sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 1])
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 1]
    S1_sum_0_sim = (np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 0] ** 2))
    S1_sum_1_sim = (np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 1] ** 2))
    S2_sum_0_sim = (np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 0] ** 2))
    S2_sum_1_sim = (np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 1] ** 2))
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.4
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.65
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.55
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 1e-2
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.03

def test_blm_start_at_truth_cat_tnv_wi():
    # Test whether BLM estimator works when starting at truth for categorical, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1238)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_wi_control': sim_cat_tnv_wi_params}
    })
    cat_tnv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_wi_control': cat_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 1] ** 2)

    # NOTE: these numbers are larger because of cons.MonotonicMean(), disable it for better results
    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 0.02
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 4.05
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 1.8
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 1.7
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.02
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.04
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_wi_control'], blm_fit.A2_cat['cat_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_wi_control'], blm_fit.S2_cat['cat_tnv_wi_control']))

def test_blm_full_estimation_cat_tnv_wi():
    # Test whether BLM estimator works for full estimation for categorical, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1239)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_wi_control': sim_cat_tnv_wi_params}
    })
    cat_tnv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_wi_control': cat_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 1] ** 2)

    # Normalization alters worker type order - adjust estimates to make comparable to simulated parameters
    adj_order = [1, 0]
    A1_sum_0_fit = A1_sum_0_fit[:, adj_order]
    A1_sum_1_fit = A1_sum_1_fit[:, adj_order]
    A2_sum_0_fit = A2_sum_0_fit[:, adj_order]
    A2_sum_1_fit = A2_sum_1_fit[:, adj_order]
    S1_sum_0_fit = S1_sum_0_fit[:, adj_order]
    S1_sum_1_fit = S1_sum_1_fit[:, adj_order]
    S2_sum_0_fit = S2_sum_0_fit[:, adj_order]
    S2_sum_1_fit = S2_sum_1_fit[:, adj_order]
    sorted_pk1 = np.reshape(np.reshape(blm_fit.pk1, (nk, nk, nl))[:, :, adj_order], (nk * nk, nl))
    sorted_pk0 = blm_fit.pk0[:, adj_order]

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-2
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 1.05
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.55
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.3
    assert np.prod(np.abs((sorted_pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((sorted_pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_wi_control'], blm_fit.A2_cat['cat_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_wi_control'], blm_fit.S2_cat['cat_tnv_wi_control']))

def test_blm_start_at_truth_cat_tv():
    # Test whether BLM estimator works when starting at truth for categorical, time-varying control variables.
    rng = np.random.default_rng(1240)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][1] ** 2)

    # NOTE: don't worry about larger values here - they are mostly driven by a single parameter estimating incorrectly
    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 0.04
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 0.02
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 0.41
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 0.35
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 8.15
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 7.9
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 2.7
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 1.75
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.09
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.35

def test_blm_full_estimation_cat_tv():
    # Test whether BLM estimator works for full estimation for categorical, time-varying control variables.
    # NOTE: n_init increased to 40
    rng = np.random.default_rng(1241)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=40, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 0.035
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 1.15
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.65
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cat_tnv():
    # Test whether BLM estimator works when starting at truth for categorical, time non-varying control variables.
    rng = np.random.default_rng(1242)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_control': sim_cat_tnv_params}
    })
    cat_tnv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_control': cat_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Normalize
    blm_fit.A1 -= sim_params['A1'][0, 0]
    blm_fit.A2 -= sim_params['A1'][0, 0]
    blm_fit.A1_cat['cat_tnv_control'] += sim_params['A1'][0, 0]
    blm_fit.A2_cat['cat_tnv_control'] += sim_params['A1'][0, 0]
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-4
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_control'], blm_fit.A2_cat['cat_tnv_control'], atol=1e-3))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_control'], blm_fit.S2_cat['cat_tnv_control'], atol=1e-3))

def test_blm_full_estimation_cat_tnv():
    # Test whether BLM estimator works for full estimation for categorical, time non-varying control variables.
    rng = np.random.default_rng(1243)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_control': sim_cat_tnv_params}
    })
    cat_tnv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_control': cat_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 0.05
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.5
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.5
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 1e-2
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_control'], blm_fit.A2_cat['cat_tnv_control'], atol=1e-4))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_control'], blm_fit.S2_cat['cat_tnv_control'], atol=1e-3))

def test_blm_start_at_truth_cts_tv_wi():
    # Test whether BLM estimator works when starting at truth for continuous, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1244)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_wi_control': sim_cts_tv_wi_params}
    })
    cts_tv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_wi_control': cts_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 0.35
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.55
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_wi_control'] - sim_params['A1_cts']['cts_tv_wi_control']) / sim_params['A1_cts']['cts_tv_wi_control'])) < 1e-2
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_wi_control'] - sim_params['A2_cts']['cts_tv_wi_control']) / sim_params['A2_cts']['cts_tv_wi_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_wi_control'] - sim_params['S1_cts']['cts_tv_wi_control']) / sim_params['S1_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_wi_control'] - sim_params['S2_cts']['cts_tv_wi_control']) / sim_params['S2_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S2_cts']['cts_tv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.03
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.025

def test_blm_full_estimation_cts_tv_wi():
    # Test whether BLM estimator works for full estimation for continuous, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1245)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_wi_control': sim_cts_tv_wi_params}
    })
    cts_tv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_wi_control': cts_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 0.9
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.06
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_wi_control'] - sim_params['A1_cts']['cts_tv_wi_control']) / sim_params['A1_cts']['cts_tv_wi_control'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_wi_control'] - sim_params['A2_cts']['cts_tv_wi_control']) / sim_params['A2_cts']['cts_tv_wi_control'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_wi_control'] - sim_params['S1_cts']['cts_tv_wi_control']) / sim_params['S1_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_wi_control'] - sim_params['S2_cts']['cts_tv_wi_control']) / sim_params['S2_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S2_cts']['cts_tv_wi_control'].size) < 0.9
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 1e-2
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cts_tnv_wi():
    # Test whether BLM estimator works when starting at truth for continuous, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1246)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_wi_control': sim_cts_tnv_wi_params}
    })
    cts_tnv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_wi_control': cts_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.2
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.5
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_wi_control'] - sim_params['A1_cts']['cts_tnv_wi_control']) / sim_params['A1_cts']['cts_tnv_wi_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_wi_control'] - sim_params['S1_cts']['cts_tnv_wi_control']) / sim_params['S1_cts']['cts_tnv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_wi_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_wi_control'], blm_fit.A2_cts['cts_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_wi_control'], blm_fit.S2_cts['cts_tnv_wi_control']))

def test_blm_full_estimation_cts_tnv_wi():
    # Test whether BLM estimator works for full estimation for continuous, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1247)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_wi_control': sim_cts_tnv_wi_params}
    })
    cts_tnv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_wi_control': cts_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.15
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.95
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_wi_control'] - sim_params['A1_cts']['cts_tnv_wi_control']) / sim_params['A1_cts']['cts_tnv_wi_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_wi_control'] - sim_params['S1_cts']['cts_tnv_wi_control']) / sim_params['S1_cts']['cts_tnv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_wi_control'], blm_fit.A2_cts['cts_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_wi_control'], blm_fit.S2_cts['cts_tnv_wi_control']))

def test_blm_start_at_truth_cts_tv():
    # Test whether BLM estimator works when starting at truth for continuous, time-varying control variables.
    rng = np.random.default_rng(1248)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_control': sim_cts_tv_params}
    })
    cts_tv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_control': cts_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.35
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 1.7
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_control'] - sim_params['A1_cts']['cts_tv_control']) / sim_params['A1_cts']['cts_tv_control'])) < 1e-5
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_control'] - sim_params['A2_cts']['cts_tv_control']) / sim_params['A2_cts']['cts_tv_control'])) < 1e-5
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_control'] - sim_params['S1_cts']['cts_tv_control']) / sim_params['S1_cts']['cts_tv_control'])) ** (1 / sim_params['S1_cts']['cts_tv_control'].size) < 0.9
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_control'] - sim_params['S2_cts']['cts_tv_control']) / sim_params['S2_cts']['cts_tv_control'])) ** (1 / sim_params['S2_cts']['cts_tv_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 1e-2
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02

def test_blm_full_estimation_cts_tv():
    # Test whether BLM estimator works for full estimation for continuous, time-varying control variables.
    rng = np.random.default_rng(1249)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_control': sim_cts_tv_params}
    })
    cts_tv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_control': cts_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.6
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.8
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_control'] - sim_params['A1_cts']['cts_tv_control']) / sim_params['A1_cts']['cts_tv_control'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_control'] - sim_params['A2_cts']['cts_tv_control']) / sim_params['A2_cts']['cts_tv_control'])) < 1e-5
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_control'] - sim_params['S1_cts']['cts_tv_control']) / sim_params['S1_cts']['cts_tv_control'])) ** (1 / sim_params['S1_cts']['cts_tv_control'].size) < 0.9
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_control'] - sim_params['S2_cts']['cts_tv_control']) / sim_params['S2_cts']['cts_tv_control'])) ** (1 / sim_params['S2_cts']['cts_tv_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cts_tnv():
    # Test whether BLM estimator works when starting at truth for continuous, time non-varying control variables.
    rng = np.random.default_rng(1250)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_control': sim_cts_tnv_params}
    })
    cts_tnv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_control': cts_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 0.75
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.75
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_control'] - sim_params['A1_cts']['cts_tnv_control']) / sim_params['A1_cts']['cts_tnv_control'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_control'] - sim_params['S1_cts']['cts_tnv_control']) / sim_params['S1_cts']['cts_tnv_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.035
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.025
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_control'], blm_fit.A2_cts['cts_tnv_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_control'], blm_fit.S2_cts['cts_tnv_control']))

def test_blm_full_estimation_cts_tnv():
    # Test whether BLM estimator works for full estimation for continuous, time non-varying control variables.
    # NOTE: n_init increased to 40
    rng = np.random.default_rng(1251)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_control': sim_cts_tnv_params}
    })
    cts_tnv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_control': cts_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=40, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.25
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 1.35
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_control'] - sim_params['A1_cts']['cts_tnv_control']) / sim_params['A1_cts']['cts_tnv_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_control'] - sim_params['S1_cts']['cts_tnv_control']) / sim_params['S1_cts']['cts_tnv_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 1e-2
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_control'], blm_fit.A2_cts['cts_tnv_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_control'], blm_fit.S2_cts['cts_tnv_control']))

def test_blm_control_constraints_linear():
    # Test whether Linear() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(1252)
    nl = 3 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Linear(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params},
        'd_mean_worker_effect': 0.025
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs(np.diff(np.diff(blm_fit.A1_cat['cat_tv_wi_control'], axis=0), axis=0))) < 1e-11
    assert np.max(np.abs(np.diff(np.diff(blm_fit.A2_cat['cat_tv_wi_control'], axis=0), axis=0))) < 1e-15

    # Make sure normalization also works properly
    assert np.all(blm_fit.A1[:, 1] == 0)
    assert np.all(blm_fit.A2[:, 1] == 0)

    # Make sure monotonic mean also works properly
    assert np.all(np.isclose(np.diff(np.mean(blm_fit.A1, axis=1)), 0.025))

def test_blm_control_constraints_linear_additive():
    # Test whether LinearAdditive() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(12522)
    nl = 3 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.LinearAdditive(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params},
        'd_mean_worker_effect': 0.025
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    assert np.var(np.diff(blm_fit.A1_cat['cat_tv_wi_control'], axis=0)) < 1e-28
    assert np.var(np.diff(blm_fit.A2_cat['cat_tv_wi_control'], axis=0)) < 1e-30

    # Make sure normalization also works properly
    assert np.all(blm_fit.A1[:, 1] == 0)
    assert np.all(blm_fit.A2[:, 1] == 0)

    # Make sure monotonic mean also works properly
    assert np.all(np.isclose(np.diff(np.mean(blm_fit.A1, axis=1)), 0.025))

def test_blm_control_constraints_monotonic():
    # Test whether Monotonic() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(1253)
    nl = 3 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Monotonic(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'cons_a': None,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    assert np.min(np.diff(sim_params['A1_cat']['cat_tv_wi_control'], axis=0)) < 0
    assert np.min(np.diff(blm_fit.A1_cat['cat_tv_wi_control'], axis=0)) >= 0
    assert np.min(np.diff(sim_params['A2_cat']['cat_tv_wi_control'], axis=0)) < 0
    assert np.min(np.diff(blm_fit.A2_cat['cat_tv_wi_control'], axis=0)) >= 0

    # Make sure normalization also works properly
    assert np.all(blm_fit.A1[:, 0] == 0)
    assert np.all(blm_fit.A2[:, 0] == 0)

def test_blm_control_constraints_stationary_firm_type_variation():
    # Test whether StationaryFirmTypeVariation() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(1254)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.StationaryFirmTypeVariation(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params},
        'force_min_firm_type': True
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    A1 = blm_fit.A1_cat['cat_tv_control']
    A2 = blm_fit.A2_cat['cat_tv_control']

    assert np.max(np.abs((A2.T - np.mean(A2, axis=1)) - (A1.T - np.mean(A1, axis=1)))) < 1e-14

def test_blm_control_constraints_lb_ub():
    # Test whether BoundedBelow() and BoundedAbove() constraints for control variables work for BLM estimator.
    rng = np.random.default_rng(1255)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': -0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_s': [cons.BoundedBelow(lb=3e-5), cons.BoundedAbove(ub=4e-5)],
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': -0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    assert np.min(blm_fit.S1_cat['cat_tv_control'] ** 2) >= 3e-5
    assert np.min(blm_fit.S2_cat['cat_tv_control'] ** 2) >= 3e-5
    assert np.max(blm_fit.S1_cat['cat_tv_control'] ** 2) <= 4e-5
    assert np.max(blm_fit.S2_cat['cat_tv_control'] ** 2) <= 4e-5

    # Make sure simulated parameters fall outside range
    assert np.min(sim_params['S1_cat']['cat_tv_control'] ** 2) <= 3e-5
    assert np.min(sim_params['S2_cat']['cat_tv_control'] ** 2) <= 3e-5
    assert np.max(sim_params['S1_cat']['cat_tv_control'] ** 2) >= 4e-5
    assert np.max(sim_params['S2_cat']['cat_tv_control'] ** 2) >= 4e-5

def test_blm_control_normalization():
    # Test whether normalization for categorical control variables works for BLM estimator.
    rng = np.random.default_rng(1256)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_params_one = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cat_params_two = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cts_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_control_one': sim_cat_params_one, 'cat_control_two': sim_cat_params_two},
        'continuous_controls': {'cts_control': sim_cts_params}
    })
    cat_params_one = tw.categorical_control_params({
        'n': n_control,
        'cons_a': None, 'cons_s': None,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cat_params_two = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cts_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_control_one': cat_params_one, 'cat_control_two': cat_params_two},
        'continuous_controls': {'cts_control': cts_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_0_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A1_sum_1_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_1_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A2_sum_0_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_0_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A2_sum_1_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_1_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A1_sum_0_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_0_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A1_sum_1_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_1_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A2_sum_0_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_0_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 1]
    A2_sum_1_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_1_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 1]

    assert np.max(np.abs((A1_sum_0_0_fit - A1_sum_0_0_sim) / A1_sum_0_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_0_1_fit - A1_sum_0_1_sim) / A1_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_0_fit - A1_sum_1_0_sim) / A1_sum_1_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_1_1_fit - A1_sum_1_1_sim) / A1_sum_1_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_0_fit - A2_sum_0_0_sim) / A2_sum_0_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_1_fit - A2_sum_0_1_sim) / A2_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_0_fit - A2_sum_1_0_sim) / A2_sum_1_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_1_fit - A2_sum_1_1_sim) / A2_sum_1_1_sim)) < 1e-3
    assert np.all(blm_fit.A1[:, 0] == 0)
    assert blm_fit.A2[0, 0] == 0
    assert np.all(blm_fit.A1_cat['cat_control_two'] == blm_fit.A2_cat['cat_control_two'])

def test_blm_control_normalization_primary_period_second():
    # Test whether normalization for categorical control variables for primary period == 'second' works for BLM estimator.
    # NOTE: rng set to 1256 instead of 1257 to avoid sorting parameters at end
    rng = np.random.default_rng(1256)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_params_one = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cat_params_two = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cts_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_control_one': sim_cat_params_one, 'cat_control_two': sim_cat_params_two},
        'continuous_controls': {'cts_control': sim_cts_params}
    })
    cat_params_one = tw.categorical_control_params({
        'n': n_control,
        'cons_a': None, 'cons_s': None,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cat_params_two = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cts_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_control_one': cat_params_one, 'cat_control_two': cat_params_two},
        'continuous_controls': {'cts_control': cts_params},
        'primary_period': 'second'
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_0_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A1_sum_1_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_1_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A2_sum_0_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_0_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A2_sum_1_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_1_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A1_sum_0_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_0_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A1_sum_1_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_1_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A2_sum_0_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_0_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 1]
    A2_sum_1_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_1_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 1]

    assert np.max(np.abs((A1_sum_0_0_fit - A1_sum_0_0_sim) / A1_sum_0_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_0_1_fit - A1_sum_0_1_sim) / A1_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_0_fit - A1_sum_1_0_sim) / A1_sum_1_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_1_1_fit - A1_sum_1_1_sim) / A1_sum_1_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_0_fit - A2_sum_0_0_sim) / A2_sum_0_0_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_0_1_fit - A2_sum_0_1_sim) / A2_sum_0_1_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_1_0_fit - A2_sum_1_0_sim) / A2_sum_1_0_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_1_1_fit - A2_sum_1_1_sim) / A2_sum_1_1_sim)) < 1e-2
    assert np.all(blm_fit.A2[:, 2] == 0)
    assert blm_fit.A1[0, 2] == 0
    assert np.all(blm_fit.A1_cat['cat_control_two'] == blm_fit.A2_cat['cat_control_two'])

def test_blm_control_normalization_primary_period_all():
    # Test whether normalization for categorical control variables for primary period == 'all' works for BLM estimator.
    # NOTE: rng set to 1260 instead of 1258 to avoid sorting parameters at end
    rng = np.random.default_rng(1260)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_params_one = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cat_params_two = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cts_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_control_one': sim_cat_params_one, 'cat_control_two': sim_cat_params_two},
        'continuous_controls': {'cts_control': sim_cts_params}
    })
    cat_params_one = tw.categorical_control_params({
        'n': n_control,
        'cons_a': None, 'cons_s': None,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cat_params_two = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cts_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_control_one': cat_params_one, 'cat_control_two': cat_params_two},
        'continuous_controls': {'cts_control': cts_params},
        'primary_period': 'all',
        'force_min_firm_type': True
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = copy.deepcopy(sim_params['A1'])
    blm_fit.A2 = copy.deepcopy(sim_params['A2'])
    blm_fit.S1 = copy.deepcopy(sim_params['S1'])
    blm_fit.S2 = copy.deepcopy(sim_params['S2'])
    blm_fit.A1_cat = copy.deepcopy(sim_params['A1_cat'])
    blm_fit.A2_cat = copy.deepcopy(sim_params['A2_cat'])
    blm_fit.S1_cat = copy.deepcopy(sim_params['S1_cat'])
    blm_fit.S2_cat = copy.deepcopy(sim_params['S2_cat'])
    blm_fit.A1_cts = copy.deepcopy(sim_params['A1_cts'])
    blm_fit.A2_cts = copy.deepcopy(sim_params['A2_cts'])
    blm_fit.S1_cts = copy.deepcopy(sim_params['S1_cts'])
    blm_fit.S2_cts = copy.deepcopy(sim_params['S2_cts'])
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    # blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_0_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A1_sum_1_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_1_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A2_sum_0_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_0_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A2_sum_1_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_1_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A1_sum_0_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_0_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A1_sum_1_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_1_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A2_sum_0_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_0_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 1]
    A2_sum_1_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_1_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 1]

    assert np.max(np.abs((A1_sum_0_0_fit - A1_sum_0_0_sim) / A1_sum_0_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_0_1_fit - A1_sum_0_1_sim) / A1_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_0_fit - A1_sum_1_0_sim) / A1_sum_1_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_1_fit - A1_sum_1_1_sim) / A1_sum_1_1_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_0_fit - A2_sum_0_0_sim) / A2_sum_0_0_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_1_fit - A2_sum_0_1_sim) / A2_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_0_fit - A2_sum_1_0_sim) / A2_sum_1_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_1_fit - A2_sum_1_1_sim) / A2_sum_1_1_sim)) < 1e-3
    assert np.all((blm_fit.A1 + blm_fit.A2)[:, 0] == 0)
    assert blm_fit.A1[0, 0] == 0
    assert blm_fit.A2[0, 0] == 0
    assert np.all(blm_fit.A1_cat['cat_control_two'] == blm_fit.A2_cat['cat_control_two'])
