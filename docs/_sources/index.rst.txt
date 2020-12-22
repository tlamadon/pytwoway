.. pytwoway documentation master file, created by
   sphinx-quickstart on Thu Nov 19 19:09:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytwoway package
================

`pytwoway` is the Python package associated with the paper:

"`How Much Should we Trust Estimates of Firm Effects and Worker Sorting?. <https://www.nber.org/system/files/working_papers/w27368/w27368.pdf>`_" Stéphane Bonhomme, Kerstin Holzheu, Thibaut Lamadon, Elena Manresa, Magne Mogstad, and Bradley Setzler.  No. w27368. National Bureau of Economic Research, 2020.

The package provides implementations for a series of estimators for models with two sided heterogeneity:

 * two way fixed effect estimator as proposed by Abowd Kramarz and Margolis
 * homoskedastic bias correction as in Andrews et al
 * heteroskedastic correction as in KSS (TBD)
 * a group fixed estimator as in BLM
 * a group correlated random effect as presented in the main paper

The code is relatively efficient. Solving large sparse linear relies on using `https://github.com/pyamg/pyamg`. This is the code we used to estimate the different decompositions on the US data. 

The package provides a python interface as well as an intuitive command line interface. Installation is handled by `pip` or `conda` (TBD). The source of the package is available on github at `pytwoway <https://github.com/tlamadon/pytwoway>`_. The online documentation is hosted  `here <https://tlamadon.github.io/pytwoway/>`_.

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

========
Citation
========

Please use following citation to cite pytwoway in academic publications:

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

   api
   notebooks/pytwoway_example
   notebooks/monte_carlo_example
   cre
   fe_approximate_correction_full
   twfe_network
   sim_twfe_network
