'''
Tests FE with controls estimators.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw
from pytwoway.util import weighted_var, weighted_cov

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

    fe_params = tw.fecontrol_params({'feonly': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEControlEstimator(bdf, fe_params)
    fe_solver.fit()

    assert np.all(np.isclose(bdf.loc[:, 'psi_hat'].to_numpy() + bdf.loc[:, 'alpha_hat'].to_numpy(), bdf.loc[:, 'y'].to_numpy()))

def test_fe_estimator_full_novar():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE with var(eps) = 0.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 0})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1234))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fecontrol_params({'he': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEControlEstimator(b, fe_params)
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

    fe_params = tw.fecontrol_params({'he': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEControlEstimator(b, fe_params)
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
    assert np.abs((est_he_sigma_2 - true_sigma_2) / true_sigma_2) < 0.015
    # var(psi)
    # assert np.abs((est_pi_var_psi - true_var_psi) / true_var_psi) < 1e-2
    assert np.abs((est_ho_var_psi - true_var_psi) / true_var_psi) < 0.025
    assert np.abs((est_he_var_psi - true_var_psi) / true_var_psi) < 0.025
    # cov(psi, alpha)
    # assert np.abs((est_pi_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-2
    assert np.abs((est_ho_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-2
    assert np.abs((est_he_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 0.015

    # y
    assert np.sum(np.isclose(b.loc[:, 'psi_hat'].to_numpy() + b.loc[:, 'alpha_hat'].to_numpy(), b.loc[:, 'y'].to_numpy(), atol=1)) / len(b) > 0.75

def test_fe_estimator_full_var_collapsed():
    # Test that FE estimates parameters correctly (sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO, and HE with var(eps) = 1 for collapsed data.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fecontrol_params({'he': True, 'attach_fe_estimates': True})
    fe_solver = tw.FEControlEstimator(b, fe_params)
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
    assert np.abs((est_he_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 0.09

    # y
    assert np.sum(np.isclose(b.loc[:, 'psi_hat'].to_numpy() + b.loc[:, 'alpha_hat'].to_numpy(), b.loc[:, 'y'].to_numpy(), atol=1)) / len(b) > 0.8

def test_fe_estimator_full_Q():
    # Test that FE estimates custom Q correctly for plug-in, HO, and HE estimators for the variances: Q.VarCovariate('psi'), Q.VarCovariate('alpha'), and Q.VarCovariate(['psi', 'alpha']); and covariances: Q.CovCovariate('psi', 'alpha') and Q.CovCovariate(['psi', 'alpha'], 'alpha').
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 0})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1236))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fecontrol_params({'he': True, 'attach_fe_estimates': True, 'Q_var': [tw.Q.VarCovariate('psi'), tw.Q.VarCovariate('alpha'), tw.Q.VarCovariate(['psi', 'alpha'])], 'Q_cov': [tw.Q.CovCovariate('psi', 'alpha'), tw.Q.CovCovariate(['psi', 'alpha'], 'alpha')]})
    fe_solver = tw.FEControlEstimator(b, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    ### True parameters ###
    w_col = b.loc[:, 'w'].to_numpy()
    # psi, alpha
    psi = b.loc[:, 'psi'].to_numpy()
    alpha = b.loc[:, 'alpha'].to_numpy()
    true_var_psi = tw.util.weighted_var(psi, w_col, dof=0)
    true_var_alpha = tw.util.weighted_var(alpha, w_col, dof=0)
    true_var_psi_plus_alpha = tw.util.weighted_var(psi + alpha, w_col, dof=0)
    true_cov_psi_alpha = tw.util.weighted_cov(psi, alpha, w_col, w_col)
    true_cov_psi_plus_alpha_alpha = tw.util.weighted_cov(psi + alpha, alpha, w_col, w_col)

    ### Estimated parameters ###
    ## Plug-in ##
    est_pi_var_psi = fe_solver.res['var(psi)_fe']
    est_pi_var_alpha = fe_solver.res['var(alpha)_fe']
    est_pi_var_psi_plus_alpha = fe_solver.res['var(psi + alpha)_fe']
    est_pi_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_fe']
    est_pi_cov_psi_plus_alpha_alpha = fe_solver.res['cov(psi + alpha, alpha)_fe']
    ## HO ##
    est_ho_var_psi = fe_solver.res['var(psi)_ho']
    est_ho_var_alpha = fe_solver.res['var(alpha)_ho']
    est_ho_var_psi_plus_alpha = fe_solver.res['var(psi + alpha)_ho']
    est_ho_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_ho']
    est_ho_cov_psi_plus_alpha_alpha = fe_solver.res['cov(psi + alpha, alpha)_ho']
    ## HE ##
    est_he_var_psi = fe_solver.res['var(psi)_he']
    est_he_var_alpha = fe_solver.res['var(alpha)_he']
    est_he_var_psi_plus_alpha = fe_solver.res['var(psi + alpha)_he']
    est_he_cov_psi_alpha = fe_solver.res['cov(psi, alpha)_he']
    est_he_cov_psi_plus_alpha_alpha = fe_solver.res['cov(psi + alpha, alpha)_he']

    # var(psi)
    assert np.abs((est_pi_var_psi - true_var_psi) / true_var_psi) < 1e-8
    assert np.abs((est_ho_var_psi - true_var_psi) / true_var_psi) < 1e-8
    assert np.abs((est_he_var_psi - true_var_psi) / true_var_psi) < 1e-8
    # var(alpha)
    assert np.abs((est_pi_var_alpha - true_var_alpha) / true_var_alpha) < 1e-8
    assert np.abs((est_ho_var_alpha - true_var_alpha) / true_var_alpha) < 1e-8
    assert np.abs((est_he_var_alpha - true_var_alpha) / true_var_alpha) < 1e-8
    # var(psi + alpha)
    assert np.abs((est_pi_var_psi_plus_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) < 1e-9
    assert np.abs((est_ho_var_psi_plus_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) < 1e-9
    assert np.abs((est_he_var_psi_plus_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) < 1e-9
    # cov(psi, alpha)
    assert np.abs((est_pi_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-8
    assert np.abs((est_ho_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-8
    assert np.abs((est_he_cov_psi_alpha - true_cov_psi_alpha) / true_cov_psi_alpha) < 1e-8
    # cov(psi + alpha, alpha)
    assert np.abs((est_pi_cov_psi_plus_alpha_alpha - true_cov_psi_plus_alpha_alpha) / true_cov_psi_plus_alpha_alpha) < 1e-8
    assert np.abs((est_ho_cov_psi_plus_alpha_alpha - true_cov_psi_plus_alpha_alpha) / true_cov_psi_plus_alpha_alpha) < 1e-8
    assert np.abs((est_he_cov_psi_plus_alpha_alpha - true_cov_psi_plus_alpha_alpha) / true_cov_psi_plus_alpha_alpha) < 1e-8
    
    # Make sure var(psi + alpha) isn't just similar to var(psi) or var(alpha)
    assert np.abs((true_var_psi - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) > 0.6
    assert np.abs((true_var_alpha - true_var_psi_plus_alpha) / true_var_psi_plus_alpha) > 0.7
    # Make sure cov(psi + alpha, alpha) isn't just similar to var(psi) or var(alpha) or var(psi + alpha)
    assert np.abs((true_var_psi - true_cov_psi_plus_alpha_alpha) / true_cov_psi_plus_alpha_alpha) > 0.15
    assert np.abs((true_var_alpha - true_cov_psi_plus_alpha_alpha) / true_cov_psi_plus_alpha_alpha) > 0.4
    assert np.abs((true_var_psi_plus_alpha - true_cov_psi_plus_alpha_alpha) / true_cov_psi_plus_alpha_alpha) > 1.2

def test_fe_weights():
    # Test that FE estimates are identical for weighted and unweighted (estimate parameters sigma^2, var(psi), and cov(psi, alpha)) for plug-in, HO and HE.
    sim_params = bpd.sim_params({'n_workers': 1000, 'w_sig': 1})
    a = bpd.SimBipartite(sim_params).simulate(np.random.default_rng(1240))
    a = bpd.BipartiteDataFrame(a, log=False).clean(bpd.clean_params({'verbose': False}))
    b = a.collapse()

    fe_params = tw.fecontrol_params({'he': True, 'attach_fe_estimates': True})
    fe_solver_a = tw.FEControlEstimator(a, fe_params)
    fe_solver_a.fit(np.random.default_rng(1241))
    fe_solver_b = tw.FEControlEstimator(b, fe_params)
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

    # gamma_hat
    assert np.all(np.isclose(fe_solver_a.gamma_hat, fe_solver_b.gamma_hat))
    # Plug-in
    # assert np.abs((est_pi_sigma_2_b - est_pi_sigma_2_a) / est_pi_sigma_2_a) < 0.4
    assert np.isclose(est_pi_var_psi_a, est_pi_var_psi_b)
    assert np.isclose(est_pi_cov_psi_alpha_a, est_pi_cov_psi_alpha_b)
    # HO
    assert np.abs((est_ho_sigma_2_b - est_ho_sigma_2_a) / est_ho_sigma_2_a) < 0.025
    assert np.abs((est_ho_var_psi_b - est_ho_var_psi_a) / est_ho_var_psi_a) < 1e-2
    assert np.abs((est_ho_cov_psi_alpha_b - est_ho_cov_psi_alpha_a) / est_ho_cov_psi_alpha_a) < 1e-2
    # HE
    assert np.abs((est_he_sigma_2_b - est_he_sigma_2_a) / est_he_sigma_2_a) < 0.15
    assert np.abs((est_he_var_psi_b - est_he_var_psi_a) / est_he_var_psi_a) < 0.025
    assert np.abs((est_he_cov_psi_alpha_b - est_he_cov_psi_alpha_a) / est_he_cov_psi_alpha_a) < 0.03

def test_fe_controls():
    # Test FE with controls
    rng = np.random.default_rng(2451)
    ## Set parameters ##
    nl = 3
    nk = 4
    n_control = 2
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        's1_low': 0, 's1_high': 0, 's2_low': 0, 's2_high': 0,
        'worker_type_interaction': False,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        's1_low': 0, 's1_high': 0, 's2_low': 0, 's2_high': 0,
        'worker_type_interaction': False,
        'stationary_A': True, 'stationary_S': True
    })
    blm_sim_params = tw.sim_params({
        'nl': nl,
        'nk': nk,
        'firm_size': 40,
        'NNm': np.ones(shape=(nk, nk)).astype(int, copy=False),
        'NNs': np.ones(shape=nk).astype(int, copy=False),
        'mmult': 10, 'smult': 10,
        'a1_sig': 1, 'a2_sig': 1, 's1_low': 0, 's1_high': 0, 's2_low': 0, 's2_high': 0,
        'categorical_controls': {
            'cat_tnv_control': sim_cat_tnv_params
        },
        'continuous_controls': {
            'cts_tnv_control': sim_cts_tnv_params
        },
        'stationary_A': True, 'stationary_S': True,
        'linear_additive': True
    })
    ## Simulate data ##
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    a = bpd.BipartiteDataFrame(pd.concat([jdata, sdata]).rename({'g': 'j', 'j': 'g'}, axis=1, allow_optional=True, allow_required=True), custom_long_es_split_dict={'l': False}).construct_artificial_time(copy=False).to_long(is_sorted=True, copy=False)

    ## Estimate model ##
    fe_params = tw.fecontrol_params(
        {
            'categorical_controls': 'cat_tnv_control',
            'continuous_controls': 'cts_tnv_control',
            'he': True,
            'ndraw_trace_he': 50,
            'attach_fe_estimates': 'all',
            'Q_var': [
                tw.Q.VarCovariate(['psi', 'alpha']),
                tw.Q.VarCovariate('cat_tnv_control'),
                tw.Q.VarCovariate('cts_tnv_control'),
                tw.Q.VarCovariate(['cat_tnv_control', 'cts_tnv_control'])
                ],
            'Q_cov': [
                tw.Q.CovCovariate('cat_tnv_control', 'cts_tnv_control'),
                tw.Q.CovCovariate(['psi', 'alpha'], ['cat_tnv_control', 'cts_tnv_control'])
                ]
        }
    )
    fe_solver = tw.FEControlEstimator(a, fe_params)
    fe_solver.fit(np.random.default_rng(1234))

    ### Estimated parameters ###
    ## Plug-in ##
    est_pi_sigma_2 = fe_solver.sigma_2_pi
    est_pi_var_psi_alpha = fe_solver.res['var(psi + alpha)_fe']
    est_pi_var_cat = fe_solver.res['var(cat_tnv_control)_fe']
    est_pi_var_cts = fe_solver.res['var(cts_tnv_control)_fe']
    est_pi_var_cat_cts = fe_solver.res['var(cat_tnv_control + cts_tnv_control)_fe']
    est_pi_cov_cat_cts = fe_solver.res['cov(cat_tnv_control, cts_tnv_control)_fe']
    est_pi_cov_psi_alpha_cat_cts = fe_solver.res['cov(psi + alpha, cat_tnv_control + cts_tnv_control)_fe']
    ## HO ##
    est_ho_sigma_2 = fe_solver.res['var(eps)_ho']
    est_ho_var_psi_alpha = fe_solver.res['var(psi + alpha)_ho']
    est_ho_var_cat = fe_solver.res['var(cat_tnv_control)_ho']
    est_ho_var_cts = fe_solver.res['var(cts_tnv_control)_ho']
    est_ho_var_cat_cts = fe_solver.res['var(cat_tnv_control + cts_tnv_control)_ho']
    est_ho_cov_cat_cts = fe_solver.res['cov(cat_tnv_control, cts_tnv_control)_ho']
    est_ho_cov_psi_alpha_cat_cts = fe_solver.res['cov(psi + alpha, cat_tnv_control + cts_tnv_control)_ho']
    ## HE ##
    est_he_sigma_2 = fe_solver.res['var(eps)_he']
    est_he_var_psi_alpha = fe_solver.res['var(psi + alpha)_he']
    est_he_var_cat = fe_solver.res['var(cat_tnv_control)_he']
    est_he_var_cts = fe_solver.res['var(cts_tnv_control)_he']
    est_he_var_cat_cts = fe_solver.res['var(cat_tnv_control + cts_tnv_control)_he']
    est_he_cov_cat_cts = fe_solver.res['cov(cat_tnv_control, cts_tnv_control)_he']
    est_he_cov_psi_alpha_cat_cts = fe_solver.res['cov(psi + alpha, cat_tnv_control + cts_tnv_control)_he']

    ### True parameters ###
    # true_psi = sim_params['A1'][0, :] - sim_params['A1'][0, nk - 1]
    # true_alpha = (sim_params['A1'] - true_psi)[:, 0]
    l = a.loc[:, 'l'].to_numpy()
    g = a.loc[:, 'j'].to_numpy()
    psi_alpha = sim_params['A1'][l, g] # true_psi[g] + true_alpha[l]
    cat = sim_params['A1_cat']['cat_tnv_control'][a.loc[:, 'cat_tnv_control'].to_numpy()]
    cts = sim_params['A1_cts']['cts_tnv_control'][0] * a.loc[:, 'cts_tnv_control'].to_numpy()

    true_sigma_2 = weighted_var(a.loc[:, 'y'].to_numpy() - psi_alpha - cat - cts)
    true_var_psi_alpha = weighted_var(psi_alpha)
    true_var_cat = weighted_var(cat)
    true_var_cts = weighted_var(cts)
    true_var_cat_cts = weighted_var(cat + cts)
    true_cov_cat_cts = weighted_cov(cat, cts)
    true_cov_psi_alpha_cat_cts = weighted_cov(psi_alpha, cat + cts)

    # Plug-in
    assert np.isclose(est_pi_sigma_2, true_sigma_2)
    assert np.abs((est_pi_var_psi_alpha - true_var_psi_alpha) / true_var_psi_alpha) < 1e-9
    assert np.abs((est_pi_var_cat - true_var_cat) / true_var_cat) < 1e-8
    assert np.abs((est_pi_var_cts - true_var_cts) / true_var_cts) < 1e-9
    assert np.abs((est_pi_var_cat_cts - true_var_cat_cts) / true_var_cat_cts) < 1e-9
    assert np.abs((est_pi_cov_cat_cts - true_cov_cat_cts) / true_cov_cat_cts) < 1e-9
    assert np.abs((est_pi_cov_psi_alpha_cat_cts - true_cov_psi_alpha_cat_cts) / true_cov_psi_alpha_cat_cts) < 1e-7
    # HO
    assert np.isclose(est_ho_sigma_2, true_sigma_2)
    assert np.abs((est_ho_var_psi_alpha - true_var_psi_alpha) / true_var_psi_alpha) < 1e-9
    assert np.abs((est_ho_var_cat - true_var_cat) / true_var_cat) < 1e-8
    assert np.abs((est_ho_var_cts - true_var_cts) / true_var_cts) < 1e-9
    assert np.abs((est_ho_var_cat_cts - true_var_cat_cts) / true_var_cat_cts) < 1e-9
    assert np.abs((est_ho_cov_cat_cts - true_cov_cat_cts) / true_cov_cat_cts) < 1e-9
    assert np.abs((est_ho_cov_psi_alpha_cat_cts - true_cov_psi_alpha_cat_cts) / true_cov_psi_alpha_cat_cts) < 1e-7
    # HE
    assert np.isclose(est_he_sigma_2, true_sigma_2)
    assert np.abs((est_he_var_psi_alpha - true_var_psi_alpha) / true_var_psi_alpha) < 1e-8
    assert np.abs((est_he_var_cat - true_var_cat) / true_var_cat) < 1e-8
    assert np.abs((est_he_var_cts - true_var_cts) / true_var_cts) < 1e-9
    assert np.abs((est_he_var_cat_cts - true_var_cat_cts) / true_var_cat_cts) < 1e-9
    assert np.abs((est_he_cov_cat_cts - true_cov_cat_cts) / true_cov_cat_cts) < 1e-9
    assert np.abs((est_he_cov_psi_alpha_cat_cts - true_cov_psi_alpha_cat_cts) / true_cov_psi_alpha_cat_cts) < 1e-7

    # y
    assert np.sum(np.isclose(a.loc[:, 'psi_hat'].to_numpy() + a.loc[:, 'alpha_hat'].to_numpy() + a.loc[:, 'cat_tnv_control_hat'].to_numpy() + a.loc[:, 'cts_tnv_control_hat'].to_numpy(), a.loc[:, 'y'].to_numpy(), atol=1)) / len(a) == 1
