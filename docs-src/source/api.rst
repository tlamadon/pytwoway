API Reference
=============

The main pytwoway API is split into six classes:

* ``pytwoway.CRESolver``: Class for CRE estimator. Canonically imported
  using

  .. code-block:: python

    from pytwoway import CRESolver as cre

* ``pytwowaySolver``: Class for FE estimators. Canonically imported
  using

  .. code-block:: python

    from pytwoway import FESolver as fe

* ``pytwoway.BipartiteData``: Class for formatting bipartite networks (usually of firms and workers). Canonically imported using

  .. code-block:: python

    from pytwoway import BipartiteData as bd

* ``pytwoway.TwoWay``: Class for formatting data and run CRE and FE estimators. Canonically imported using

  .. code-block:: python

    from pytwoway import TwoWay as tw

* ``pytwoway.SimTwoWay``: Class to simulate labor data. Canonically imported using

  .. code-block:: python

    from pytwoway import SimTwoWay as stw

* ``pytwoway.TwoWayMonteCarlo``: Class to run Monte Carlo simulations. Canonically imported using

  .. code-block:: python

    from pytwoway import TwoWayMonteCarlo as twmc

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
