Using from python 
=================

 
To run in Python:

- If you have data

.. code-block:: python

    from pytwoway import TwoWay as tw
    # Create TwoWay object
    tw_net = tw(data, formatting, col_dict)
    # Fit the FE estimators:
    tw_net.fit_fe(fe_params)
    # Fit the CRE estimator
    tw_net.fit_cre(cre_params, cluster_params)

- If you want to simulate data

.. code-block:: python

    from pytwoway import SimTwoWay as stw
    # Create SimTwoWay object
    stw_net = stw(sim_params)
    # Generate data
    sim_data = stw_net.sim_network()

- If you want to run Monte Carlo on simulated data

.. code-block:: python

    from pytwoway import TwoWayMonteCarlo as twmc
    # Create simulated Monte Carlo object
    twmc_net = twmc(sim_params)
    # Run Monte Carlo
    twmc_net.twfe_monte_carlo(N, ncore, akm_params, cre_params, cluster_params)
    # Plot results
    twmc_net.plot_monte_carlo()
