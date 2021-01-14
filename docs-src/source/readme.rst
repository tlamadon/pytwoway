.. image:: https://badge.fury.io/py/pytwoway.svg
    :target: https://badge.fury.io/py/pytwoway

.. image:: https://travis-ci.com/tlamadon/pytwoway.svg?branch=master
    :target: https://travis-ci.com/tlamadon/pytwoway

`pytwoway` is the Python package associated with the following paper:

"`How Much Should we Trust Estimates of Firm Effects and Worker Sorting?. <https://www.nber.org/system/files/working_papers/w27368/w27368.pdf>`_" 
by Stéphane Bonhomme, Kerstin Holzheu, Thibaut Lamadon, Elena Manresa, Magne Mogstad, and Bradley Setzler.  
No. w27368. National Bureau of Economic Research, 2020.

The package provides implementations for a series of estimators for models with two sided heterogeneity:

1. two way fixed effect estimator as proposed by Abowd Kramarz and Margolis
2. homoskedastic bias correction as in Andrews et al
3. heteroskedastic correction as in KSS (TBD)
4. a group fixed estimator as in BLM
5. a group correlated random effect as presented in the main paper

.. |binder| image:: https://mybinder.org/badge_logo.svg 
    :target: https://mybinder.org/v2/gh/tlamadon/pytwoway/HEAD?filepath=docs%2Fnotebooks%2Fpytwoway_example.ipynb

If you want to give it a try, you can start the example notebook here: |binder|. This starts a fully interactive notebook with a simple example that generates data and runs the estimators.

The code is relatively efficient. Solving large sparse linear models relies on using `pyamg <https://github.com/pyamg/pyamg>`_. This is the code we used to estimate the different decompositions on the US data. 

The package provides a python interface as well as an intuitive command line interface. Installation is handled by `pip` or `conda` (TBD). The source of the package is available on github at `pytwoway <https://github.com/tlamadon/pytwoway>`_. The online documentation is hosted  `here <https://tlamadon.github.io/pytwoway/>`_.

Quick Start
-----------

To install from pip, run::

    pip install pytwoway


To run using the command line interface::

    pytw --my-config config.txt --fe --cre


Example config.txt::

    data = file.csv
    filetype = csv
    col_dict = "{'fid': 'your_firmid_col', 'wid': 'your_workerid_col', 'year': 'your_year_col', 'comp': 'your_compensation_col'}"


Citation
--------

Please use following citation to cite pytwoway in academic publications:

Bibtex entry::

  @techreport{bhlmms2020,
    title={How Much Should We Trust Estimates of Firm Effects and Worker Sorting?},
    author={Bonhomme, St{\'e}phane and Holzheu, Kerstin and Lamadon, Thibaut and Manresa, Elena and Mogstad, Magne and Setzler, Bradley},
    year={2020},
    institution={National Bureau of Economic Research}
  }
