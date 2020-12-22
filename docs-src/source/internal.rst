==========
Python API
==========

Overview
---------

The main pytwoway API is split into four modules:

* ``pytwoway.cre``: CRE estimator. Canonically imported
  using

  .. code-block:: python

    from python import cre

* ``pytwoway.fe_approximate_correction_full.FEsolver``: FE esimators. Canonically imported
  using

  .. code-block:: python

    from python import fe_approximate_correction_full
    fe = fe_approximate_correction_full.FEsolver

* ``pytwoway.twfe_network.twfe_network``: Class to format labor data. Canonically imported using

  .. code-block:: python

    from pytwoway import twfe_network
    tn = twfe_network.twfe_network

* ``pytwoway.sim_twfe_network.sim_twfe_network``: Class to simulate labor data and run Monte Carlo simulations. Canonically imported using

  .. code-block:: python

    from pytwoway import sim_twfe_network
    sn = sim_twfe_network.sim_twfe_network

.. The API focuses on models and the most frequently used statistical test, and tools.
.. :ref:`api-structure:Import Paths and Structure` explains the design of the two API modules and how
.. importing from the API differs from directly importing from the module where the
.. model is defined. See the detailed topic pages in the :ref:`user-guide:User Guide` for a complete
.. list of available models, statistics, and tools.

Modules and Methods
-------------------

``pytwoway.cre``
~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.cre.CREsolver
   ~pytwoway.cre.CREsolver.compute_posterior_var
   ~pytwoway.cre.CREsolver.estimate_between_cluster
   ~pytwoway.cre.CREsolver.estimate_within_cluster
   ~pytwoway.cre.CREsolver.estimate_within_parameters
   ~pytwoway.cre.expand_grid
   ~pytwoway.cre.main
   ~pytwoway.cre.pd_to_np
   ~pytwoway.cre.pipe_qcov

``pytwoway.fe_approximate_correction_full``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autosummary::

   ~pytwoway.fe_approximate_correction_full.FEsolver
   ~pytwoway.fe_approximate_correction_full.FEsolver.collect_res
   ~pytwoway.fe_approximate_correction_full.FEsolver.compute_early_stats
   ~pytwoway.fe_approximate_correction_full.FEsolver.compute_leverages_Pii
   ~pytwoway.fe_approximate_correction_full.FEsolver.compute_trace_approximation_fe
   ~pytwoway.fe_approximate_correction_full.FEsolver.compute_trace_approximation_he
   ~pytwoway.fe_approximate_correction_full.FEsolver.construct_Q
   ~pytwoway.fe_approximate_correction_full.FEsolver.create_fe_solver
   ~pytwoway.fe_approximate_correction_full.FEsolver.get_akm_estimates
   ~pytwoway.fe_approximate_correction_full.FEsolver.init_prepped_adata
   ~pytwoway.fe_approximate_correction_full.FEsolver.leverage_approx
   ~pytwoway.fe_approximate_correction_full.FEsolver.load
   ~pytwoway.fe_approximate_correction_full.FEsolver.mult_A
   ~pytwoway.fe_approximate_correction_full.FEsolver.mult_AAinv
   ~pytwoway.fe_approximate_correction_full.FEsolver.mult_Atranspose
   ~pytwoway.fe_approximate_correction_full.FEsolver.prep_data
   ~pytwoway.fe_approximate_correction_full.FEsolver.proj
   ~pytwoway.fe_approximate_correction_full.FEsolver.run_1
   ~pytwoway.fe_approximate_correction_full.FEsolver.run_2
   ~pytwoway.fe_approximate_correction_full.FEsolver.save
   ~pytwoway.fe_approximate_correction_full.FEsolver.save_early_stats
   ~pytwoway.fe_approximate_correction_full.FEsolver.save_res
   ~pytwoway.fe_approximate_correction_full.FEsolver.solve
   ~pytwoway.fe_approximate_correction_full.FEsolver.weighted_quantile
   ~pytwoway.fe_approximate_correction_full.FEsolver.weighted_var


``pytwoway.twfe_network``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.twfe_network.twfe_network
   ~pytwoway.twfe_network.twfe_network.approx_cdfs
   ~pytwoway.twfe_network.twfe_network.clean_data
   ~pytwoway.twfe_network.twfe_network.cluster
   ~pytwoway.twfe_network.twfe_network.conset
   ~pytwoway.twfe_network.twfe_network.contiguous_fids
   ~pytwoway.twfe_network.twfe_network.data_validity
   ~pytwoway.twfe_network.twfe_network.n_firms
   ~pytwoway.twfe_network.twfe_network.n_workers
   ~pytwoway.twfe_network.twfe_network.refactor_es
   ~pytwoway.twfe_network.twfe_network.run_akm_corrected
   ~pytwoway.twfe_network.twfe_network.run_cre
   ~pytwoway.twfe_network.twfe_network.update_cols
   ~pytwoway.twfe_network.twfe_network.update_dict

``pytwoway.sim_twfe_network``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.sim_twfe_network.sim_twfe_network
   ~pytwoway.sim_twfe_network.sim_twfe_network.plot_monte_carlo
   ~pytwoway.sim_twfe_network.sim_twfe_network.sim_network
   ~pytwoway.sim_twfe_network.sim_twfe_network.sim_network_draw_fids
   ~pytwoway.sim_twfe_network.sim_twfe_network.sim_network_gen_fe
   ~pytwoway.sim_twfe_network.sim_twfe_network.twfe_monte_carlo
   ~pytwoway.sim_twfe_network.sim_twfe_network.twfe_monte_carlo_interior
   ~pytwoway.sim_twfe_network.sim_twfe_network.update_dict
