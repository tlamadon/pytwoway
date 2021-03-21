==========
Python API
==========

Overview
---------

The main pytwoway API is split into seven classes. pytwoway is canonically imported using

  .. code-block:: python

    import pytwoway as tw

* ``pytwoway.TwoWay``: Class for formatting data and running CRE and FE estimators.

* ``pytwoway.BLMEstimator``: Class for BLM estimator.

* ``pytwoway.CREEstimator``: Class for CRE estimator.

* ``pytwoway.FEEstimator``: Class for FE estimators.

* ``pytwoway.BipartiteData``: Class for formatting bipartite networks (usually of firms and workers).

* ``pytwoway.SimTwoWay``: Class to simulate labor data.

* ``pytwoway.TwoWayMonteCarlo``: Class to run Monte Carlo simulations.

Modules and Methods
-------------------

``pytwoway.TwoWay``
~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.TwoWay
   ~pytwoway.TwoWay.fit_cre
   ~pytwoway.TwoWay.fit_fe
   ~pytwoway.TwoWay.summary_cre
   ~pytwoway.TwoWay.summary_fe

``pytwoway.BLMEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. autosummary::
   
   ~pytwoway.BLMEstimator
   ~pytwoway.BLMEstimator.fit
   ~pytwoway.BLMEstimator.plot_A1
   ~pytwoway.BLMEstimator.plot_liks_connectedness

``pytwoway.CREEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.CREEstimator
   ~pytwoway.CREEstimator.fit
   ~pytwoway.cre.pd_to_np
   ~pytwoway.cre.pipe_qcov

``pytwoway.FEEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.FEEstimator
   ~pytwoway.FEEstimator.construct_Q
   ~pytwoway.FEEstimator.fit_1
   ~pytwoway.FEEstimator.fit_2
   ~pytwoway.FEEstimator.get_fe_estimates

``pytwoway.BipartiteData``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.BipartiteData
   ~pytwoway.BipartiteData.copy
   ~pytwoway.BipartiteData.clean_data
   ~pytwoway.BipartiteData.cluster
   ~pytwoway.BipartiteData.drop
   ~pytwoway.BipartiteData.rename
   ~pytwoway.BipartiteData.get_cs
   ~pytwoway.BipartiteData.long_to_collapsed_long
   ~pytwoway.BipartiteData.long_to_es
   ~pytwoway.BipartiteData.collapsed_long_to_es
   ~pytwoway.BipartiteData.es_to_long
   ~pytwoway.BipartiteData.es_to_collapsed_long
   ~pytwoway.BipartiteData.to_csv
   ~pytwoway.BipartiteData.to_feather
   ~pytwoway.BipartiteData.to_stata

``pytwoway.SimTwoWay``
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.SimTwoWay
   ~pytwoway.SimTwoWay.sim_network

``pytwoway.TwoWayMonteCarlo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.TwoWayMonteCarlo
   ~pytwoway.TwoWayMonteCarlo.plot_monte_carlo
   ~pytwoway.TwoWayMonteCarlo.twfe_monte_carlo
