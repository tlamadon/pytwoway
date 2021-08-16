Using from Python
=================

To install via pip, from the command line run::

   pip install pytwoway

Sample data: :download:`download <twoway_sample_data.csv>`

To run in Python:

- If you want to run estimators on your own data:

.. code-block:: python

   import pytwoway as tw
   # Create TwoWay object
   tw_net = tw.TwoWay(data)
   # Clean data
   tw_net.prep_data()
   # Fit the FE estimators
   tw_net.fit_fe()
   # Save the FE results
   fe_res = tw_net.summary_fe()
   # Cluster to prepare for CRE
   tw_net.cluster()
   # Fit the CRE estimator
   tw_net.fit_cre()
   # Save the CRE results
   cre_res = tw_net.summary_cre()

.. note::
   Your data must include the following columns:
    - ``i``: worker identifier
    - ``j``: firm identifier
    - ``y``: compensation
    - ``t``: time
   .. list-table:: Example data
      :widths: 25 25 25 25
      :header-rows: 1
      :align: center

      * - i
        - j
        - y
        - t

      * - 1
        - 1
        - 1000
        - 2019
      * - 1
        - 2
        - 1500
        - 2020
      * - 2
        - 3
        - 500
        - 2019
      * - 2
        - 3
        - 550
        - 2020

- If you want to run estimators on simulated data:

.. code-block:: python

   import pytwoway as tw
   from bipartitepandas import SimBipartite
   # Create SimTwoWay object
   sbp_net = SimBipartite()
   # Generate data
   sim_data = sbp_net.sim_network()
   # Below is identical to first example, except we are now using the simulated data
   # Create TwoWay object
   tw_net = tw.TwoWay(sim_data)
   # Clean data
   tw_net.prep_data()
   # Fit the FE estimators:
   tw_net.fit_fe()
   # Save the FE results
   fe_res = tw_net.summary_fe()
   # Cluster to prepare for CRE
   tw_net.cluster()
   # Fit the CRE estimator
   tw_net.fit_cre()
   # Save the CRE results
   cre_res = tw_net.summary_cre()

- If you want to run a Monte Carlo estimation on simulated data:

.. code-block:: python

   import pytwoway as tw
   # Create TwoWayMonteCarlo object
   twmc_net = tw.TwoWayMonteCarlo()
   # Run Monte Carlo
   twmc_net.twfe_monte_carlo()
   # Plot results
   twmc_net.plot_monte_carlo()
