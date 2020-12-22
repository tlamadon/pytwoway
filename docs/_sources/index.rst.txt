.. pytwoway documentation master file, created by
   sphinx-quickstart on Thu Nov 19 19:09:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytwoway
========

pytwoway is a Python module that provides classes and functions for the estimation
of two way fixed effect models for labor. It includes AKM, homoskedastic- and heteroskedastic-corrected AKM, and CRE estimators,
as well as simulation tools. Estimators are tested against simulations to ensure they are correct.
The online documentation is hosted at `https://tlamadon.github.io/pytwoway/ <https://tlamadon.github.io/pytwoway/>`_.

.. :ref:`statsmodels <about:About statsmodels>` is a Python module that provides classes and functions for the estimation
.. of many different statistical models, as well as for conducting statistical tests, and statistical
.. data exploration. An extensive list of result statistics are available for each estimator.
.. The results are tested against existing statistical packages to ensure that they are correct. The
.. package is released under the open source Modified BSD (3-clause) license.
.. The online documentation is hosted at `statsmodels.org <https://www.statsmodels.org/>`__.

===========
Quick Start
===========

To install from pip, run::

  pip install pytwoway

To run using command line interface::

  pytw --my-config config.txt --akm --cre

Example config.txt::

    data = file.csv
    filetype = csv
    col_dict = "{'fid': 'your_firmid_col', 'wid': 'your_workerid_col', 'year': 'your_year_col', 'comp': 'your_compensation_col'}"

To run in Python:

- If you have data

.. code-block:: python

    from pytwoway import twfe_network
    tn = twfe_network.twfe_network
    # Create twfe object
    tw_net = tn.twfe_network(data, formatting, col_dict)
    # Convert long data into event study data (not necessary if the data is already in event study format):
    tw_net.refactor_es()
    # Run the bias-corrected AKM estimator:
    tw_net.run_akm_corrected(user_akm)
    # Cluster firms based on their wage CDFs (required for the CRE estimator)
    tw_net.cluster(user_cluster)
    # Run the CRE estimator
    tw_net.run_cre(user_cre)

- If you want to simulate data

.. code-block:: python

    from pytwoway import sim_twfe_network
    sn = sim_twfe_network.sim_twfe_network
    # Create simulated twfe object
    stw_net = sn(sim_params)
    # Generate data
    sim_data = stw_net.sim_network()

- If you want to run Monte Carlo on simulated data

.. code-block:: python

    from pytwoway import sim_twfe_network
    sn = sim_twfe_network.sim_twfe_network
    # Create simulated twfe object
    stw_net = sn(sim_params)
    # Run Monte Carlo
    stw_net.twfe_monte_carlo(N, ncore, akm_params, cre_params, cluster_params)
    # Plot results
    stw_net.plot_monte_carlo()


.. The API focuses on models and the most frequently used statistical test, and tools.
.. :ref:`api-structure:Import Paths and Structure` explains the design of the two API modules and how
.. importing from the API differs from directly importing from the module where the
.. model is defined. See the detailed topic pages in the :ref:`user-guide:User Guide` for a complete
.. list of available models, statistics, and tools.

========
Citation
========

Please use following citation to cite pytwoway in academic publications:

Bonhomme, Stéphane, Kerstin Holzheu, Thibaut Lamadon, Elena Manresa, Magne Mogstad, and Bradley Setzler. "`How Much Should we Trust Estimates of Firm Effects and Worker Sorting?. <https://www.nber.org/system/files/working_papers/w27368/w27368.pdf>`_" No. w27368. National Bureau of Economic Research, 2020.

Bibtex entry::

  @techreport{bonhomme2020much,
    title={How Much Should We Trust Estimates of Firm Effects and Worker Sorting?},
    author={Bonhomme, St{\'e}phane and Holzheu, Kerstin and Lamadon, Thibaut and Manresa, Elena and Mogstad, Magne and Setzler, Bradley},
    year={2020},
    institution={National Bureau of Economic Research}
  }

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   notebooks/pytwoway_example
   notebooks/monte_carlo_example
   api
   cre
   fe_approximate_correction_full
   path_cov
   twfe_network
   sim_twfe_network
