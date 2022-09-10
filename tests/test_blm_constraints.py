'''
Tests for BLM estimator with constraints.
'''
import pytest
import copy
import numpy as np
import pytwoway as tw
from pytwoway import constraints as cons

###############
##### BLM #####
###############

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
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_sim_params = tw.sim_blm_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Linear(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
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
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_sim_params = tw.sim_blm_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.LinearAdditive(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
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
    blm_sim_params = tw.sim_blm_params({
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
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_sim_params = tw.sim_blm_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.StationaryFirmTypeVariation(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
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
        's1_low': 0.005, 's1_high': 0.5, 's2_low': 0.005, 's2_high': 0.5
    })
    blm_sim_params = tw.sim_blm_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_s': [cons.BoundedBelow(lb=0.08), cons.BoundedAbove(ub=0.09)],
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': -0.5, 'a2_sig': 2.5,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.1, 's2_high': 0.1
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
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

    assert np.min(blm_fit.S1_cat['cat_tv_control'] ** 2) >= 0.08
    assert np.min(blm_fit.S2_cat['cat_tv_control'] ** 2) >= 0.08
    assert np.max(blm_fit.S1_cat['cat_tv_control'] ** 2) <= 0.09
    assert np.max(blm_fit.S2_cat['cat_tv_control'] ** 2) <= 0.09

    # Make sure simulated parameters fall outside range
    assert np.min(sim_params['S1_cat']['cat_tv_control'] ** 2) <= 0.08
    assert np.min(sim_params['S2_cat']['cat_tv_control'] ** 2) <= 0.08
    assert np.max(sim_params['S1_cat']['cat_tv_control'] ** 2) >= 0.09
    assert np.max(sim_params['S2_cat']['cat_tv_control'] ** 2) >= 0.09

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
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    sim_cat_params_two = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    sim_cts_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_sim_params = tw.sim_blm_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
        'categorical_controls': {'cat_control_one': sim_cat_params_one, 'cat_control_two': sim_cat_params_two},
        'continuous_controls': {'cts_control': sim_cts_params}
    })
    cat_params_one = tw.categorical_control_params({
        'n': n_control,
        'cons_a': None, 'cons_s': None,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    cat_params_two = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    cts_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
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
    assert np.max(np.abs((A1_sum_1_1_fit - A1_sum_1_1_sim) / A1_sum_1_1_sim)) < 1e-2
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
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    sim_cat_params_two = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    sim_cts_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_sim_params = tw.sim_blm_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
        'categorical_controls': {'cat_control_one': sim_cat_params_one, 'cat_control_two': sim_cat_params_two},
        'continuous_controls': {'cts_control': sim_cts_params}
    })
    cat_params_one = tw.categorical_control_params({
        'n': n_control,
        'cons_a': None, 'cons_s': None,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    cat_params_two = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    cts_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0.02, 's1_high': 0.02, 's2_low': 0.02, 's2_high': 0.02,
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
    assert np.max(np.abs((A1_sum_0_1_fit - A1_sum_0_1_sim) / A1_sum_0_1_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_1_0_fit - A1_sum_1_0_sim) / A1_sum_1_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_1_1_fit - A1_sum_1_1_sim) / A1_sum_1_1_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_0_0_fit - A2_sum_0_0_sim) / A2_sum_0_0_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_0_1_fit - A2_sum_0_1_sim) / A2_sum_0_1_sim)) < 0.015
    assert np.max(np.abs((A2_sum_1_0_fit - A2_sum_1_0_sim) / A2_sum_1_0_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_1_1_fit - A2_sum_1_1_sim) / A2_sum_1_1_sim)) < 0.015
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
    blm_sim_params = tw.sim_blm_params({
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
