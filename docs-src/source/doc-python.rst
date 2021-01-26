Using from python
=================

To install from pip, from the command line run::

   pip install pytwoway

Sample data: :download:`download <twoway_sample_data.csv>`

To run in Python:

- If you want to run estimators on your own data:

.. code-block:: python

   import pytwoway as tw
   # Create TwoWay object
   tw_net = tw.TwoWay(data)
   # Fit the FE estimators
   tw_net.fit_fe()
   # Save the FE results
   fe_res = tw_net.summary_fe()
   # Fit the CRE estimator
   tw_net.fit_cre()
   # Save the CRE results
   cre_res = tw_net.summary_cre()

.. note::
   Your data must include the following columns:
    - ``wid``: the worker identifier
    - ``fid``: the firm identifier
    - ``year``: the time
    - ``comp``: the outcome variable, in our case compensation
   .. list-table:: Example data
      :widths: 25 25 25 25
      :header-rows: 1
      :align: center

      * - wid
        - fid
        - year
        - comp

      * - 1
        - 1
        - 2019
        - 1000
      * - 1
        - 2
        - 2020
        - 1500
      * - 2
        - 3
        - 2019
        - 500
      * - 2
        - 3
        - 2020
        - 550

- If you want to run estimators on simulated data:

.. code-block:: python

   import pytwoway as tw
   # Create SimTwoWay object
   stw_net = tw.SimTwoWay()
   # Generate data
   sim_data = stw_net.sim_network()
   # Below is identical to first example, except we are now using the simulated data
   # Create TwoWay object
   tw_net = tw.TwoWay(sim_data)
   # Fit the FE estimators:
   tw_net.fit_fe()
   # Save the FE results
   fe_res = tw_net.summary_fe()
   # Fit the CRE estimator
   tw_net.fit_cre()
   # Save the CRE results
   cre_res = tw_net.summary_cre()

- If you want to run Monte Carlo on simulated data:

.. code-block:: python

   import pytwoway as tw
   # Create TwoWayMonteCarlo object
   twmc_net = tw.TwoWayMonteCarlo()
   # Run Monte Carlo
   twmc_net.twfe_monte_carlo()
   # Plot results
   twmc_net.plot_monte_carlo()
