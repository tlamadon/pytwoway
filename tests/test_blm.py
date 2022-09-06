'''
Tests for BLM estimator.
'''
import pytest
import copy
import numpy as np
import pytwoway as tw
from pytwoway import constraints as cons

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
    blm_sim_params = tw.sim_blm_params({
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
#     blm_sim_params = tw.sim_blm_params({
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
#     blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)

#     assert np.min(np.diff(blm_fit.model.liks1)[:83]) > 0
#     assert np.min(np.diff(blm_fit.model.liks0)) > 0

def test_blm_qi():
    # Test whether BLM posterior probabilities are giving the most weight to the correct type.
    rng = np.random.default_rng(1234)
    nl = 3
    nk = 4
    # Define parameter dictionaries
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.max(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) < 0.025
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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
    # NOTE: don't normalize
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
    blm_sim_params = tw.sim_blm_params({
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
        'categorical_controls': {'cat_tv_control': cat_tv_params},
        'normalize': False
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

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-4
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.6
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.35
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.035

def test_blm_full_estimation_cat_tv():
    # Test whether BLM estimator works for full estimation for categorical, time-varying control variables.
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-2
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
    blm_sim_params = tw.sim_blm_params({
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
    # NOTE: n_init increased to 25
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=25, n_best=5, ncore=8, rng=rng)
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_sim_params = tw.sim_blm_params({
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
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=8, rng=rng)
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
