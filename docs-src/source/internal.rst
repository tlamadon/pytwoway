==========
Python API
==========

Overview
---------

The main pytwoway API is split into five modules:

* ``pytwoway.cre``: Module for CRE estimator. Canonically imported
  using

  .. code-block:: python

    from pytwoway import cre

* ``pytwoway.fe``: Module for FE estimators. Canonically imported
  using

  .. code-block:: python

    from pytwoway import fe

* ``pytwoway.bipartite_network``: Module for formatting bipartite networks (usually of firms and workers). Canonically imported using

  .. code-block:: python

    from pytwoway import bipartite_network as bn

* ``pytwoway.twfe_network``: Module for formatting data and run CRE and FE estimators. Canonically imported using

  .. code-block:: python

    from pytwoway import twfe_network as tn

* ``pytwoway.sim_twfe_network``: Class to simulate labor data and run Monte Carlo simulations. Canonically imported using

  .. code-block:: python

    from pytwoway import sim_twfe_network as sn

Modules and Methods
-------------------

``pytwoway.cre``
~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.cre.CRESolver
   ~pytwoway.cre.CRESolver.compute_posterior_var
   ~pytwoway.cre.CRESolver.estimate_between_cluster
   ~pytwoway.cre.CRESolver.estimate_within_cluster
   ~pytwoway.cre.CRESolver.estimate_within_parameters
   ~pytwoway.cre.expand_grid
   ~pytwoway.cre.main
   ~pytwoway.cre.pd_to_np
   ~pytwoway.cre.pipe_qcov

``pytwoway.fe``
~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.fe.FESolver
   ~pytwoway.fe.FESolver.collect_res
   ~pytwoway.fe.FESolver.compute_early_stats
   ~pytwoway.fe.FESolver.compute_leverages_Pii
   ~pytwoway.fe.FESolver.compute_trace_approximation_fe
   ~pytwoway.fe.FESolver.compute_trace_approximation_he
   ~pytwoway.fe.FESolver.construct_Q
   ~pytwoway.fe.FESolver.create_fe_solver
   ~pytwoway.fe.FESolver.fit_1
   ~pytwoway.fe.FESolver.fit_2
   ~pytwoway.fe.FESolver.get_fe_estimates
   ~pytwoway.fe.FESolver.init_prepped_adata
   ~pytwoway.fe.FESolver.leverage_approx
   ~pytwoway.fe.FESolver.load
   ~pytwoway.fe.FESolver.mult_A
   ~pytwoway.fe.FESolver.mult_AAinv
   ~pytwoway.fe.FESolver.mult_Atranspose
   ~pytwoway.fe.FESolver.prep_data
   ~pytwoway.fe.FESolver.proj
   ~pytwoway.fe.FESolver.save
   ~pytwoway.fe.FESolver.save_early_stats
   ~pytwoway.fe.FESolver.save_res
   ~pytwoway.fe.FESolver.solve
   ~pytwoway.fe.FESolver.weighted_quantile
   ~pytwoway.fe.FESolver.weighted_var

``pytwoway.bipartite_network``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.bipartite_network.BipartiteData
   ~pytwoway.bipartite_network.BipartiteData.approx_cdfs
   ~pytwoway.bipartite_network.BipartiteData.clean_data
   ~pytwoway.bipartite_network.BipartiteData.cluster
   ~pytwoway.bipartite_network.BipartiteData.conset
   ~pytwoway.bipartite_network.BipartiteData.contiguous_fids
   ~pytwoway.bipartite_network.BipartiteData.data_validity
   ~pytwoway.bipartite_network.BipartiteData.n_firms
   ~pytwoway.bipartite_network.BipartiteData.n_workers
   ~pytwoway.bipartite_network.BipartiteData.refactor_es
   ~pytwoway.bipartite_network.BipartiteData.refactor_pseudo_long
   ~pytwoway.bipartite_network.BipartiteData.update_cols
   ~pytwoway.bipartite_network.BipartiteData.update_dict

``pytwoway.twfe_network``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.twfe_network.TwoWay
   ~pytwoway.twfe_network.TwoWay.fit_cre
   ~pytwoway.twfe_network.TwoWay.fit_fe
   ~pytwoway.twfe_network.TwoWay.prep_cre
   ~pytwoway.twfe_network.TwoWay.prep_fe

``pytwoway.sim_twfe_network``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.sim_twfe_network.SimTwoWay
   ~pytwoway.sim_twfe_network.SimTwoWay.sim_network
   ~pytwoway.sim_twfe_network.SimTwoWay.sim_network_draw_fids
   ~pytwoway.sim_twfe_network.SimTwoWay.sim_network_gen_fe
   ~pytwoway.sim_twfe_network.SimTwoWay.update_dict
   ~pytwoway.sim_twfe_network.TwoWayMonteCarlo
   ~pytwoway.sim_twfe_network.TwoWayMonteCarlo.plot_monte_carlo
   ~pytwoway.sim_twfe_network.TwoWayMonteCarlo.twfe_monte_carlo
   ~pytwoway.sim_twfe_network.TwoWayMonteCarlo.twfe_monte_carlo_interior
