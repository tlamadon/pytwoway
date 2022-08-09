'''
Tests Borovickova and Shimer estimator.
'''
import pytest
import numpy as np
import pytwoway as tw

##################################
##### Borovickova and Shimer #####
##################################

def test_bs():
    # Test Borovickova and Shimer estimator.
    sigma_lambda, sigma_mu, rho = 1.25, 0.75, 0.25
    sigma_lambda_sq, sigma_mu_sq = sigma_lambda ** 2, sigma_mu ** 2
    sim_params = tw.sim_bs_params(
        {
            'sigma_lambda_sq': sigma_lambda ** 2,
            'sigma_mu_sq': sigma_mu ** 2,
            'sigma_wages': (sigma_lambda ** 2 + sigma_mu ** 2 - 2 * rho * sigma_lambda * sigma_mu) / (1 - rho ** 2),
            'rho': rho
        }
    )
    sim_data = tw.SimBS(sim_params).simulate(rng=np.random.default_rng(4619))

    bs_estimator = tw.BSEstimator()
    bs_estimator.fit(sim_data, alternative_estimator=False, weighted=False)
    res_1 = bs_estimator.res

    bs_estimator.fit(sim_data, alternative_estimator=False, weighted=True)
    res_2 = bs_estimator.res

    bs_estimator.fit(sim_data, alternative_estimator=True, weighted=False)
    res_3 = bs_estimator.res

    bs_estimator.fit(sim_data, alternative_estimator=True, weighted=True)
    res_4 = bs_estimator.res

    # sigma_lambda_sq
    assert np.abs((res_1['var(lambda)'] - sigma_lambda_sq) / sigma_lambda_sq) < 0.05
    assert np.abs((res_2['var(lambda)'] - sigma_lambda_sq) / sigma_lambda_sq) < 0.05
    assert np.abs((res_3['var(lambda)'] - sigma_lambda_sq) / sigma_lambda_sq) < 0.05
    assert np.abs((res_4['var(lambda)'] - sigma_lambda_sq) / sigma_lambda_sq) < 0.05
    # sigma_mu_sq
    assert np.abs((res_1['var(mu)'] - sigma_mu_sq) / sigma_mu_sq) < 1e-2
    assert np.abs((res_2['var(mu)'] - sigma_mu_sq) / sigma_mu_sq) < 1e-2
    assert np.abs((res_3['var(mu)'] - sigma_mu_sq) / sigma_mu_sq) < 1e-2
    assert np.abs((res_4['var(mu)'] - sigma_mu_sq) / sigma_mu_sq) < 1e-2
    # rho
    assert np.abs((res_1['corr(mu, lambda)'] - rho) / rho) < 0.05
    assert np.abs((res_2['corr(mu, lambda)'] - rho) / rho) < 0.05
    assert np.abs((res_3['corr(mu, lambda)'] - rho) / rho) < 0.05
    assert np.abs((res_4['corr(mu, lambda)'] - rho) / rho) < 0.05
