PyTwoWay
--------

.. image:: https://badge.fury.io/py/pytwoway.svg
    :target: https://badge.fury.io/py/pytwoway

.. image:: https://anaconda.org/tlamadon/pytwoway/badges/version.svg
    :target: https://anaconda.org/tlamadon/pytwoway

.. image:: https://anaconda.org/tlamadon/pytwoway/badges/platforms.svg
    :target: https://anaconda.org/tlamadon/pytwoway

.. image:: https://circleci.com/gh/tlamadon/pytwoway/tree/master.svg?style=shield
    :target: https://circleci.com/gh/tlamadon/pytwoway/tree/master

.. image:: https://img.shields.io/badge/doc-latest-blue
    :target: https://tlamadon.github.io/pytwoway/

.. image:: https://badgen.net/badge//gh/pytwoway?icon=github
    :target: https://github.com/tlamadon/pytwoway

`PyTwoWay` is the Python package associated with the following paper:

"`How Much Should we Trust Estimates of Firm Effects and Worker Sorting? <https://www.nber.org/system/files/working_papers/w27368/w27368.pdf>`_"
by Stéphane Bonhomme, Kerstin Holzheu, Thibaut Lamadon, Elena Manresa, Magne Mogstad, and Bradley Setzler.
No. w27368. National Bureau of Economic Research, 2020.

The package provides implementations for a series of estimators for models with two sided heterogeneity:

1. two way fixed effect estimator as proposed by `Abowd, Kramarz, and Margolis <https://doi.org/10.1111/1468-0262.00020>`_
2. homoskedastic bias correction as in `Andrews, et al. <https://doi.org/10.1111/j.1467-985X.2007.00533.x>`_
3. heteroskedastic bias correction as in `Kline, Saggio, and Sølvsten <https://doi.org/10.3982/ECTA16410>`_
4. group fixed estimator as in `Bonhomme, Lamadon, and Manresa <https://doi.org/10.3982/ECTA15722>`_
5. group correlated random effect as presented in the main paper
6. fixed-point revealed preference estimator as in `Sorkin <https://doi.org/10.1093/qje/qjy001>`_
7. non-parametric sorting estimator as in `Borovičková and Shimer <https://drive.google.com/file/d/1KW0sZ4nV9bIdVhcs-UW8yW_dzUr782v5/view>`_

.. |binder_fe| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/tlamadon/pytwoway/HEAD?filepath=docs%2Fsource%2Fnotebooks%2Ffe_example.ipynb
.. |binder_cre| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/tlamadon/pytwoway/HEAD?filepath=docs%2Fsource%2Fnotebooks%2Fcre_example.ipynb
.. |binder_blm| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/tlamadon/pytwoway/HEAD?filepath=docs%2Fsource%2Fnotebooks%2Fblm_example.ipynb
.. |binder_sorkin| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/tlamadon/pytwoway/HEAD?filepath=docs%2Fsource%2Fnotebooks%2Fsorkin_example.ipynb
.. |binder_bs| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/tlamadon/pytwoway/HEAD?filepath=docs%2Fsource%2Fnotebooks%2Fborovickovashimer_example.ipynb

If you want to give it a try, you can start an example notebook for the FE estimator here: |binder_fe| for the CRE estimator here: |binder_cre| for the BLM estimator here: |binder_blm| for the Sorkin estimator here: |binder_sorkin| and for the Borovickova-Shimer estimator here: |binder_bs|. These start fully interactive notebooks with simple examples that simulate data and run the estimators.

The package provides a Python interface. Installation is handled by `pip` or `Conda` (TBD). The source of the package is available on GitHub at `PyTwoWay <https://github.com/tlamadon/pytwoway>`_. The online documentation is hosted `here <https://tlamadon.github.io/pytwoway/>`_.

The code is relatively efficient. A benchmark below compares `PyTwoWay`'s speed with that of `LeaveOutTwoWay <https://github.com/rsaggio87/LeaveOutTwoWay/>`_, a MATLAB package for estimating AKM and its bias corrections.

Quick Start
-----------

To install via pip, from the command line run::

    pip install pytwoway

To make sure you are running the most up-to-date version of `PyTwoWay`, from the command line run::

    pip install --upgrade pytwoway

Please DO NOT download the Conda version of the package, as it is outdated!

Help with Running the Package
-----------------------------

Please check out the `documentation <https://tlamadon.github.io/pytwoway/>`_ for detailed examples of how to use `PyTwoWay`. If you have a question that the documentation doesn't answer, please also check the `past Issues <https://github.com/tlamadon/pytwoway/issues?q=is%3Aissue+is%3Aclosed/>`_ to see if someone else has already asked this question and an answer has been provided. If you still can't find an answer, please open a `new Issue <https://github.com/tlamadon/pytwoway/issues/>`_ and we will try to answer as quickly as possible.

Benchmarking
------------

Data is simulated from `BipartitePandas <https://github.com/tlamadon/bipartitepandas/>`_ using the following code:

.. code-block:: python

    import numpy as np
    import bipartitepandas as bpd

    sim_params = bpd.sim_params({'n_workers': 500000, 'firm_size': 10, 'p_move': 0.05})
    rng = np.random.default_rng(1234)

    sim_data = bpd.SimBipartite(sim_params).simulate(rng)

This data is then estimated using the `PyTwoWay` class `FEEstimator` and using the MATLAB package `LeaveOutTwoWay <https://github.com/rsaggio87/LeaveOutTwoWay/>`_. For estimation using `PyTwoWay`, all estimators other than AMG use the incomplete Cholesky decomposition as a preconditioner.

Results are estimated on a 2021 MacBook Pro 14" with 16 GB Ram and an Apple M1 Pro processor with 8 cores.

Some summary statistics about the largest leave-one-match-out set:

+----------+-----------+--------+---------+
| Package  | #obs      | #firms | #movers |
+==========+===========+========+=========+
| KSS      | 2,255,370 | 44,510 | 88,542  |
+----------+-----------+--------+---------+
| PyTwoWay | 2,269,665 | 44,601 | 89,098  |
+----------+-----------+--------+---------+

Run time:

+---------------+----------+------------+--------+
| Solver        | Cleaning | Estimation | Total  |
+===============+==========+============+========+
| KSS           | N/A      | N/A        | 55.2s  |
+---------------+----------+------------+--------+
| PYTW-AMG      | 4.0s     | 3m2s       | 3m6s   |
+---------------+----------+------------+--------+
| PYTW-BICG     | 4.0s     | 20.4s      | 24.4s  |
+---------------+----------+------------+--------+
| PYTW-BICGSTAB | 4.0s     | 21.9s      | 25.9s  |
+---------------+----------+------------+--------+
| PYTW-CG       | 4.0s     | 19.6s      | 23.6s  |
+---------------+----------+------------+--------+
| PYTW-CGS      | 4.0s     | 20.6s      | 24.6s  |
+---------------+----------+------------+--------+
| PYTW-GMRES    | 4.0s     | 32.9s      | 36.9s  |
+---------------+----------+------------+--------+
| PYTW-MINRES   | 4.0s     | 10.7s      | 14.7s  |
+---------------+----------+------------+--------+
| PYTW-QMR      | 4.0s     | 3m53s      | 3m57s  |
+---------------+----------+------------+--------+

Contributing to the Package
----------------------------

If you want to contribute to the package, the easiest way is to test that it's working properly! If you notice a part of the package is giving incorrect results, please add a new post in `Issues <https://github.com/tlamadon/pytwoway/issues/>`_ and we will do our best to fix it as soon as possible.

We are also happy to consider any suggestions to improve the package and documentation, whether to add a new feature, make a feature more user-friendly, or make the documentation clearer. Please also post suggestions in `Issues <https://github.com/tlamadon/pytwoway/issues/>`_.

Finally, if you would like to help with developing the package, please make a fork of the repository and submit pull requests with any changes you make! These will be promptly reviewed, and hopefully accepted!

We are extremely grateful for all contributions made by the community!

Dependencies
------------

Solving large sparse linear models relies on a combination of `PyAMG <https://github.com/pyamg/pyamg/>`_ (this is the package we use to estimate the different decompositions on US data) and `SciPy`'s `iterative sparse linear solvers <https://scipy-lectures.org/advanced/scipy_sparse/solvers.html/>`_.

Many tools for handling sparse matrices come from `SciPy <https://scipy.org/>`_.

Additional preconditioners for linear solvers come from `PyMatting <https://github.com/pymatting/pymatting/>`_ (installing the package is not required, as the necessary files have been copied into the submodule `preconditioners`). The incomplete Cholesky preconditioner in turn relies on `Numba <http://numba.pydata.org/>`_.

Constrained optimization is handled by `QPSolvers <https://github.com/stephane-caron/qpsolvers/>`_.

Progress bars are generated with `tqdm <https://github.com/tqdm/tqdm/>`_.

Data cleaning is handled by `BipartitePandas <https://github.com/tlamadon/bipartitepandas/>`_.

We also rely on a number of standard libraries, such as `NumPy`, `Pandas`, `matplotlib`, etc.

Optionally, the code is compatible with `multiprocess <https://github.com/uqfoundation/multiprocess/>`_. Installing this may help if multiprocessing is raising errors related to pickling objects.

Citation
--------

Please use following citation to cite PyTwoWay in academic publications:

Bibtex entry::

  @techreport{bhlmms2020,
    title={How Much Should We Trust Estimates of Firm Effects and Worker Sorting?},
    author={Bonhomme, St{\'e}phane and Holzheu, Kerstin and Lamadon, Thibaut and Manresa, Elena and Mogstad, Magne and Setzler, Bradley},
    year={2020},
    institution={National Bureau of Economic Research}
  }

Authors
-------

Thibaut Lamadon,
Assistant Professor in Economics, University of Chicago,
lamadon@uchicago.edu


Adam A. Oppenheimer,
Research Professional, University of Chicago,
oppenheimer@uchicago.edu
