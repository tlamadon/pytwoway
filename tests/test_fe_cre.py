'''
Tests FE and CRE estimators.
'''
'''
TODO:
    -Check that for FE, uncollapsed and collapsed estimators give identical results
    -Update tests that stopped working
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw

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
    est_pi_var_psi = fe_solver.var_fe['var(psi)']
    est_pi_cov_psi_alpha = fe_solver.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['var(eps)_ho']
    est_ho_var_psi = fe_solver.res['var(psi)_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['var(eps)_he']
    est_he_var_psi = fe_solver.res['var(psi)_he']
    est_he_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_he']

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
    est_pi_var_psi = fe_solver.var_fe['var(psi)']
    est_pi_cov_psi_alpha = fe_solver.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['var(eps)_ho']
    est_ho_var_psi = fe_solver.res['var(psi)_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['var(eps)_he']
    est_he_var_psi = fe_solver.res['var(psi)_he']
    est_he_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_he']

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
    est_pi_var_psi = fe_solver.var_fe['var(psi)']
    est_pi_cov_psi_alpha = fe_solver.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['var(eps)_ho']
    est_ho_var_psi = fe_solver.res['var(psi)_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['var(eps)_he']
    est_he_var_psi = fe_solver.res['var(psi)_he']
    est_he_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_he']

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
    est_pi_var_psi_a = fe_solver_a.var_fe['var(psi)']
    est_pi_cov_psi_alpha_a = fe_solver_a.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2_a = fe_solver_a.res['var(eps)_ho']
    est_ho_var_psi_a = fe_solver_a.res['var(psi)_ho']
    est_ho_cov_psi_alpha_a = fe_solver_a.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2_a = fe_solver_a.res['var(eps)_he']
    est_he_var_psi_a = fe_solver_a.res['var(psi)_he']
    est_he_cov_psi_alpha_a = fe_solver_a.res['cov(psi, alpha)_he']
    ### Approximate ###
    ## Plug-in ##
    est_pi_sigma_2_b = fe_solver_b.sigma_2_pi
    est_pi_var_psi_b = fe_solver_b.var_fe['var(psi)']
    est_pi_cov_psi_alpha_b = fe_solver_b.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2_b = fe_solver_b.res['var(eps)_ho']
    est_ho_var_psi_b = fe_solver_b.res['var(psi)_ho']
    est_ho_cov_psi_alpha_b = fe_solver_b.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2_b = fe_solver_b.res['var(eps)_he']
    est_he_var_psi_b = fe_solver_b.res['var(psi)_he']
    est_he_cov_psi_alpha_b = fe_solver_b.res['cov(psi, alpha)_he']

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
    est_pi_var_psi_a = fe_solver_a.var_fe['var(psi)']
    est_pi_cov_psi_alpha_a = fe_solver_a.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2_a = fe_solver_a.res['var(eps)_ho']
    est_ho_var_psi_a = fe_solver_a.res['var(psi)_ho']
    est_ho_cov_psi_alpha_a = fe_solver_a.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2_a = fe_solver_a.res['var(eps)_he']
    est_he_var_psi_a = fe_solver_a.res['var(psi)_he']
    est_he_cov_psi_alpha_a = fe_solver_a.res['cov(psi, alpha)_he']
    ### Approximate ###
    ## Plug-in ##
    est_pi_sigma_2_b = fe_solver_b.sigma_2_pi
    est_pi_var_psi_b = fe_solver_b.var_fe['var(psi)']
    est_pi_cov_psi_alpha_b = fe_solver_b.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2_b = fe_solver_b.res['var(eps)_ho']
    est_ho_var_psi_b = fe_solver_b.res['var(psi)_ho']
    est_ho_cov_psi_alpha_b = fe_solver_b.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2_b = fe_solver_b.res['var(eps)_he']
    est_he_var_psi_b = fe_solver_b.res['var(psi)_he']
    est_he_cov_psi_alpha_b = fe_solver_b.res['cov(psi, alpha)_he']

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
    true_cov_psi_prev_psi_next = tw.util.weighted_cov(psi[prev_rows], psi[next_rows], w_col[prev_rows], w_col[next_rows])
    true_cov_psi_alpha = tw.util.weighted_cov(psi, alpha, w_col, w_col)

    # Estimated parameters
    ## Plug-in ##
    est_pi_var_alpha = fe_solver.var_fe['var(alpha)']
    est_pi_cov_psi_prev_psi_next = fe_solver.cov_fe['cov(psi_t, psi_{t+1})']
    ## HO ##
    est_ho_var_alpha = fe_solver.res['var(alpha)_ho']
    est_ho_cov_psi_prev_psi_next = fe_solver.res['cov(psi_t, psi_{t+1})_ho']
    ## HE ##
    est_he_var_alpha = fe_solver.res['var(alpha)_he']
    est_he_cov_psi_prev_psi_next = fe_solver.res['cov(psi_t, psi_{t+1})_he']

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

def test_fe_estimator_full_Q_2():
    # Test that FE estimates custom Q correctly for plug-in, HO, and HE estimators for Q.VarPsiPlusAlpha().
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 0})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean()
    b = a.collapse()

    fe_params = tw.fe_params({'he': True, 'exact_trace_sigma_2': True, 'exact_trace_ho': False, 'exact_trace_he': False, 'exact_lev_he': True, 'attach_fe_estimates': True, 'Q_var': tw.Q.VarPsiPlusAlpha(), 'Q_cov': tw.Q.CovPsiAlpha()})
    fe_solver = tw.FEEstimator(b, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    # True parameters
    # psi, alpha
    true_var_psi_plus_alpha = np.var(a.loc[:, 'psi'].to_numpy() + a.loc[:, 'alpha'].to_numpy(), ddof=0)
    true_var_psi = np.var(a.loc[:, 'psi'].to_numpy(), ddof=0)
    true_var_alpha = np.var(a.loc[:, 'alpha'].to_numpy(), ddof=0)

    # Estimated parameters
    ## Plug-in ##
    est_pi_var_psi_plus_alpha = fe_solver.var_fe['var(psi + alpha)']
    ## HO ##
    est_ho_var_psi_plus_alpha = fe_solver.res['var(psi + alpha)_ho']
    ## HE ##
    est_he_var_psi_plus_alpha = fe_solver.res['var(psi + alpha)_he']

    # var(psi + alpha)
    assert np.abs((est_pi_var_psi_plus_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) < 1e-10
    assert np.abs((est_ho_var_psi_plus_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) < 1e-10
    assert np.abs((est_he_var_psi_plus_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) < 1e-10
    # Make sure var(psi + alpha) isn't just similar to var(psi) or var(alpha)
    assert np.abs((true_var_psi - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) > 0.6
    assert np.abs((true_var_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) > 0.7

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
    est_pi_var_psi_a = fe_solver_a.var_fe['var(psi)']
    est_pi_cov_psi_alpha_a = fe_solver_a.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2_a = fe_solver_a.res['var(eps)_ho']
    est_ho_var_psi_a = fe_solver_a.res['var(psi)_ho']
    est_ho_cov_psi_alpha_a = fe_solver_a.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2_a = fe_solver_a.res['var(eps)_he']
    est_he_var_psi_a = fe_solver_a.res['var(psi)_he']
    est_he_cov_psi_alpha_a = fe_solver_a.res['cov(psi, alpha)_he']
    ### Weighted ###
    ## Plug-in ##
    est_pi_sigma_2_b = fe_solver_b.sigma_2_pi
    est_pi_var_psi_b = fe_solver_b.var_fe['var(psi)']
    est_pi_cov_psi_alpha_b = fe_solver_b.cov_fe['cov(psi, alpha)']
    ## HO ##
    est_ho_sigma_2_b = fe_solver_b.res['var(eps)_ho']
    est_ho_var_psi_b = fe_solver_b.res['var(psi)_ho']
    est_ho_cov_psi_alpha_b = fe_solver_b.res['cov(psi, alpha)_ho']
    ## HE ##
    est_he_sigma_2_b = fe_solver_b.res['var(eps)_he']
    est_he_var_psi_b = fe_solver_b.res['var(psi)_he']
    est_he_cov_psi_alpha_b = fe_solver_b.res['cov(psi, alpha)_he']

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
