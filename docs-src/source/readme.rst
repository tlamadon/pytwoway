`pytwoway` is the Python package associated with the paper:

"`How Much Should we Trust Estimates of Firm Effects and Worker Sorting?. <https://www.nber.org/system/files/working_papers/w27368/w27368.pdf>`_" 
by Stéphane Bonhomme, Kerstin Holzheu, Thibaut Lamadon, Elena Manresa, Magne Mogstad, and Bradley Setzler.  
No. w27368. National Bureau of Economic Research, 2020.

The package provides implementations for a series of estimators for model with two sided heterogeneity:

1. two way fixed effect estimator as proposed by Abowd Kramarz and Margolis
2. homoskedastic bias correction as in Andrews et al
3. heteroskedastic correction as in KSS (TBD)
4. a group fixed estimator as in BLM
5. a group correlated random effect as presented in the main paper

The code is relatively efficient. Solving large sparse linear relies on using `pyamg <https://github.com/pyamg/pyamg>`_. This is the code we used to estimate the different decompositions on the US data. 

The package provides a python interface as well as an intuitive command line interface. Installation is handled by `pip` or `conda` (TBD). The source of the package is available on github at `pytwoway <https://github.com/tlamadon/pytwoway>`_. The online documentation is hosted  `here <https://tlamadon.github.io/pytwoway/>`_.

Quick Start
-----------

To install from pip, run::

  pip install pytwoway

To run using command line interface::

  pytw --my-config config.txt --akm --cre

Example config.txt::

    data = file.csv
    filetype = csv
    col_dict = "{'fid': 'your_firmid_col', 'wid': 'your_workerid_col', 'year': 'your_year_col', 'comp': 'your_compensation_col'}"


Citation
--------

Please use following citation to cite pytwoway in academic publications:

Bibtex entry::

  @techreport{bonhomme2020much,
    title={How Much Should We Trust Estimates of Firm Effects and Worker Sorting?},
    author={Bonhomme, St{\'e}phane and Holzheu, Kerstin and Lamadon, Thibaut and Manresa, Elena and Mogstad, Magne and Setzler, Bradley},
    year={2020},
    institution={National Bureau of Economic Research}
  }
