.. pytwoway documentation master file, created by
   sphinx-quickstart on Thu Nov 19 19:09:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytwoway
========

pytwoway is a Python module that provides classes and functions for the estimation
of two way fixed effect models. It includes AKM, homoskedastic- and heteroskedastic-corrected AKM, and CRE estimators,
as well as simulation tools. Estimators are tested against simulations to ensure they are correct.
The online documentation is hosted at `github.com/tlamadon/pytwoway <https://github.com/tlamadon/pytwoway/>`__.

.. :ref:`statsmodels <about:About statsmodels>` is a Python module that provides classes and functions for the estimation
.. of many different statistical models, as well as for conducting statistical tests, and statistical
.. data exploration. An extensive list of result statistics are available for each estimator.
.. The results are tested against existing statistical packages to ensure that they are correct. The
.. package is released under the open source Modified BSD (3-clause) license.
.. The online documentation is hosted at `statsmodels.org <https://www.statsmodels.org/>`__.

The main pytwoway API is split into four modules:

* ``pytoway.cre``: CRE estimator. Canonically imported
  using ``import statsmodels.api as sm``.
* ``pytoway.fe_approximate_correction_full.FEsolver``: FE esimators. Canonically imported
  using ``import pytoway.fe_approximate_correction_full.FEsolver as fe``.
* ``pytoway.path_cov``: covariance estimators. Canonically imported
  using ``import pytoway.path_cov as pc``.
* ``pytwoway.twfe_network.twfe_network``: Class to format labor data. Canonically imported using
  ``import pytwoway.twfe_network.twfe_network as tn``.
* ``pytwoway.sim_twfe_network.sim_twfe_network``: Class to format labor data. Canonically imported using
  ``import pytwoway.sim_twfe_network.sim_twfe_network as sn``.

.. The API focuses on models and the most frequently used statistical test, and tools.
.. :ref:`api-structure:Import Paths and Structure` explains the design of the two API modules and how
.. importing from the API differs from directly importing from the module where the
.. model is defined. See the detailed topic pages in the :ref:`user-guide:User Guide` for a complete
.. list of available models, statistics, and tools.

``pytoway.cre``
---------------
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
----------------------------------------------------
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

.. ``pytwoway.path_cov``
.. --------------------
.. .. autosummary::

..    ~pytwoway.path_cov.compute_path_cov
..    ~pytwoway.path_cov.draw_worker_path
..    ~pytwoway.path_cov.valid_neighbors
..    ~pytwoway.path_cov.valid_neighbors_simple

``pytwoway.twfe_network``
-------------------------------------
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
---------------------------------------------
.. autosummary::

   ~pytwoway.sim_twfe_network.rand
   ~pytwoway.sim_twfe_network.randint
   ~pytwoway.sim_twfe_network.sim_twfe_network
   ~pytwoway.sim_twfe_network.sim_twfe_network.plot_monte_carlo
   ~pytwoway.sim_twfe_network.sim_twfe_network.sim_network
   ~pytwoway.sim_twfe_network.sim_twfe_network.sim_network_draw_fids
   ~pytwoway.sim_twfe_network.sim_twfe_network.sim_network_gen_fe
   ~pytwoway.sim_twfe_network.sim_twfe_network.twfe_monte_carlo
   ~pytwoway.sim_twfe_network.sim_twfe_network.twfe_monte_carlo_interior
   ~pytwoway.sim_twfe_network.sim_twfe_network.update_dict

========
Citation
========

Please use following citation to cite pytwoway in academic publications:


Lamadon, Thibaut. "`pytwoway: two way fixed effect estimation with
python. <https://github.com/tlamadon/pytwoway>`_" *No title.* 2020.

Bibtex entry::

  @inproceedings{lamadon2020pytwoway,
    title={pytwoway: two way fixed effect estimation with python},
    author={Lamadon, Thibaut},
    booktitle={No title},
    year={2020},
  }

.. toctree::
   :maxdepth: 0
   :hidden:

   cre
   fe_approximate_correction_full
   path_cov
   twfe_network
   sim_twfe_network
   notebooks/pytwoway_example
   notebooks/monte_carlo_example
