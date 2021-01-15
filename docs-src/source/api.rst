API Reference
=============

The main pytwoway API is split into six classes. pytwoway is canonically imported using

  .. code-block:: python

    import pytwoway as tw

* ``pytwoway.CRESolver``: Class for CRE estimator.

* ``pytwoway.FESolver``: Class for FE estimators.

* ``pytwoway.BipartiteData``: Class for formatting bipartite networks (usually of firms and workers).

* ``pytwoway.TwoWay``: Class for formatting data and running CRE and FE estimators.

* ``pytwoway.SimTwoWay``: Class to simulate labor data.

* ``pytwoway.TwoWayMonteCarlo``: Class to run Monte Carlo simulations.

Modules and Methods
-------------------

``pytwoway.cre``
~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.CRESolver
   ~pytwoway.CRESolver.fit
   ~pytwoway.cre.pd_to_np
   ~pytwoway.cre.pipe_qcov

``pytwoway``
~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.FESolver
   ~pytwoway.FESolver.construct_Q
   ~pytwoway.FESolver.fit_1
   ~pytwoway.FESolver.fit_2
   ~pytwoway.FESolver.get_fe_estimates

``pytwoway.BipartiteData``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.BipartiteData
   ~pytwoway.BipartiteData.clean_data
   ~pytwoway.BipartiteData.cluster
   ~pytwoway.BipartiteData.es_to_cs
   ~pytwoway.BipartiteData.es_to_long
   ~pytwoway.BipartiteData.long_to_es

``pytwoway.TwoWay``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.TwoWay
   ~pytwoway.TwoWay.fit_cre
   ~pytwoway.TwoWay.fit_fe

``pytwoway.SimTwoWay``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.SimTwoWay
   ~pytwoway.SimTwoWay.sim_network

``pytwoway.TwoWayMonteCarlo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.TwoWayMonteCarlo
   ~pytwoway.TwoWayMonteCarlo.plot_monte_carlo
   ~pytwoway.TwoWayMonteCarlo.twfe_monte_carlo
