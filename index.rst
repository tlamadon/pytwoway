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

   ~cre.CREsolver
   ~cre.CREsolver.compute_posterior_var
   ~cre.CREsolver.estimate_between_cluster
   ~cre.CREsolver.estimate_within_cluster
   ~cre.CREsolver.estimate_within_parameters
   ~cre.expand_grid
   ~cre.main
   ~cre.pd_to_np
   ~cre.pipe_qcov

``pytwoway.fe_approximate_correction_full.FEsolver``
----------------------------------------------------
.. autosummary::

   ~fe_approximate_correction_full.FEsolver
   ~fe_approximate_correction_full.FEsolver.collect_res
   ~fe_approximate_correction_full.FEsolver.compute_early_stats
   ~fe_approximate_correction_full.FEsolver.compute_leverages_Pii
   ~fe_approximate_correction_full.FEsolver.compute_trace_approximation_fe
   ~fe_approximate_correction_full.FEsolver.compute_trace_approximation_he
   ~fe_approximate_correction_full.FEsolver.construct_Q
   ~fe_approximate_correction_full.FEsolver.create_fe_solver
   ~fe_approximate_correction_full.FEsolver.get_akm_estimates
   ~fe_approximate_correction_full.FEsolver.init_prepped_adata
   ~fe_approximate_correction_full.FEsolver.leverage_approx
   ~fe_approximate_correction_full.FEsolver.load
   ~fe_approximate_correction_full.FEsolver.mult_A
   ~fe_approximate_correction_full.FEsolver.mult_AAinv
   ~fe_approximate_correction_full.FEsolver.mult_Atranspose
   ~fe_approximate_correction_full.FEsolver.prep_data
   ~fe_approximate_correction_full.FEsolver.proj
   ~fe_approximate_correction_full.FEsolver.run_1
   ~fe_approximate_correction_full.FEsolver.run_2
   ~fe_approximate_correction_full.FEsolver.save
   ~fe_approximate_correction_full.FEsolver.save_early_stats
   ~fe_approximate_correction_full.FEsolver.save_res
   ~fe_approximate_correction_full.FEsolver.solve
   ~fe_approximate_correction_full.FEsolver.weighted_quantile
   ~fe_approximate_correction_full.FEsolver.weighted_var

``pytoway.path_cov``
--------------------
.. autosummary::

   ~path_cov.compute_path_cov
   ~path_cov.draw_worker_path
   ~path_cov.valid_neighbors
   ~path_cov.valid_neighbors_simple

``pytoway.twfe_network.twfe_network``
-------------------------------------
.. autosummary::

   ~twfe_network.twfe_network
   ~twfe_network.twfe_network.approx_cdfs
   ~twfe_network.twfe_network.clean_data
   ~twfe_network.twfe_network.cluster
   ~twfe_network.twfe_network.conset
   ~twfe_network.twfe_network.contiguous_fids
   ~twfe_network.twfe_network.data_validity
   ~twfe_network.twfe_network.n_firms
   ~twfe_network.twfe_network.n_workers
   ~twfe_network.twfe_network.refactor_es
   ~twfe_network.twfe_network.run_akm_corrected
   ~twfe_network.twfe_network.run_cre
   ~twfe_network.twfe_network.update_cols
   ~twfe_network.twfe_network.update_dict

``pytoway.sim_twfe_network.sim_twfe_network``
---------------------------------------------
.. autosummary::

   ~sim_twfe_network.rand
   ~sim_twfe_network.randint
   ~sim_twfe_network.sim_twfe_network
   ~sim_twfe_network.sim_twfe_network.plot_monte_carlo
   ~sim_twfe_network.sim_twfe_network.sim_network
   ~sim_twfe_network.sim_twfe_network.sim_network_draw_fids
   ~sim_twfe_network.sim_twfe_network.sim_network_gen_fe
   ~sim_twfe_network.sim_twfe_network.twfe_monte_carlo
   ~sim_twfe_network.sim_twfe_network.twfe_monte_carlo_interior
   ~sim_twfe_network.sim_twfe_network.update_dict

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
   :maxdepth: 5
   :hidden:

   ../site_sphinx/cre
   ../site_sphinx/fe_approximate_correction_full
   ../site_sphinx/path_cov
   ../site_sphinx/twfe_network
   ../site_sphinx/sim_twfe_network

.. .. toctree::
..    :maxdepth: 1

..    cre
..    fe_approximate_correction_full
..    twfe_network
..    sim_twfe_network
..    modules

.. Index
.. =====

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
