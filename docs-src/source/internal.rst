==========
Python API
==========

Overview
---------

The main pytwoway API is split into six classes. pytwoway is canonically imported using

  .. code-block:: python

    import pytwoway as tw

* ``pytwoway.TwoWay``: Class for formatting data and running estimators

* ``pytwoway.BLMEstimator``: Class for BLM estimator

* ``pytwoway.CREEstimator``: Class for CRE estimator

* ``pytwoway.FEEstimator``: Class for FE estimators

* ``pytwoway.SimTwoWay``: Class for simulating labor data

* ``pytwoway.TwoWayMonteCarlo``: Class for running Monte Carlo simulations

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
