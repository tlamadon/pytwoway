Using from python 
=================

 
To run in Python:

- If you have data

.. code-block:: python

    import pytwoway as tw
    # Create TwoWay object
    tw_net = tw.TwoWay(data, formatting, col_dict)
    # Fit the FE estimators:
    fe_res = tw_net.fit_fe(fe_params)
    # Fit the CRE estimator
    cre_res = tw_net.fit_cre(cre_params, cluster_params)

- If you want to simulate data

.. code-block:: python

    import pytwoway as tw
    # Create SimTwoWay object
    stw_net = tw.SimTwoWay(sim_params)
    # Generate data
    sim_data = stw_net.sim_network()

- If you want to run Monte Carlo on simulated data

.. code-block:: python

    import pytwoway as tw
    # Create TwoWayMonteCarlo object
    twmc_net = tw.TwoWayMonteCarlo(sim_params)
    # Run Monte Carlo
    twmc_net.twfe_monte_carlo(N, ncore, akm_params, cre_params, cluster_params)
    # Plot results
    twmc_net.plot_monte_carlo()
