==========
Python API
==========

Overview
---------

The main PyTwoWay API is split into eight classes, seven of which are for estimating models, and one of which is for simulating BLM data. It also has three modules: one for constructing the variance-covariance matrix for AKM and its bias corrections; one for generating constraints for BLM; and one for estimating attrition using increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers. PyTwoWay is canonically imported using

  .. code-block:: python

    import pytwoway as tw

Classes
~~~~~~~

* ``pytwoway.FEEstimator``: Class for estimating AKM and its bias corrections

* ``pytwoway.CREEstimator``: Class for estimating the CRE model

* ``pytwoway.BLMModel``: Class for estimating the BLM model once

* ``pytwoway.BLMEstimator``: Class for estimating the BLM model with multiple sets of starting values

* ``pytwoway.BLMBootstrap``: Class for estimating the BLM model with bootstrapped confidence intervals

* ``pytwoway.Attrition``: Class for generating attrition plots

* ``pytwoway.MonteCarlo``: Class for running Monte Carlo estimations

* ``pytwoway.SimBLM``: Class for simulating BLM data

Mdules
~~~~~~

* ``pytwoway.Q``: Module for constructing the variance-covariance matrix for AKM and its bias corrections

* ``pytwoway.constraints``: Module for generating constraints for BLM

* ``pytwoway.attrition_utils``: Module for estimating attrition using increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers

Classes and Methods
-------------------

``pytwoway.FEEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.FEEstimator
   ~pytwoway.FEEstimator.fit

``pytwoway.CREEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.CREEstimator
   ~pytwoway.CREEstimator.fit

``pytwoway.Attrition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.Attrition
   ~pytwoway.Attrition.attrition
   ~pytwoway.Attrition.boxplots
   ~pytwoway.Attrition.plots

``pytwoway.MonteCarlo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.MonteCarlo
   ~pytwoway.MonteCarlo.hist
   ~pytwoway.MonteCarlo.monte_carlo

``pytwoway.BLMModel``
~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. autosummary::
   
   ~pytwoway.BLMModel
   ~pytwoway.BLMModel.compute_connectedness_measure
   ~pytwoway.BLMModel.fit_movers
   ~pytwoway.BLMModel.fit_movers_cstr_uncstr
   ~pytwoway.BLMModel.fit_stayers
   ~pytwoway.BLMModel.plot_log_earnings
   ~pytwoway.BLMModel.plot_type_proportions

``pytwoway.BLMEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. autosummary::
   
   ~pytwoway.BLMEstimator
   ~pytwoway.BLMEstimator.fit
   ~pytwoway.BLMEstimator.plot_liks_connectedness
   ~pytwoway.BLMEstimator.plot_log_earnings
   ~pytwoway.BLMEstimator.plot_type_proportions

``pytwoway.BLMBootstrap``
~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. autosummary::
   
   ~pytwoway.BLMBootstrap
   ~pytwoway.BLMBootstrap.fit
   ~pytwoway.BLMBootstrap.plot_liks_connectedness
   ~pytwoway.BLMBootstrap.plot_log_earnings
   ~pytwoway.BLMBootstrap.plot_type_proportions

``pytwoway.SimBLM``
~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. autosummary::
   
   ~pytwoway.SimBLM
   ~pytwoway.SimBLM.simulate

Modules and Methods
-------------------

``pytwoway.Q``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.Q.CovPsiAlpha
   ~pytwoway.Q.CovPsiPrevPsiNext
   ~pytwoway.Q.VarAlpha
   ~pytwoway.Q.VarPsi

``pytwoway.constraints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.constraints.BoundedAbove
   ~pytwoway.constraints.BoundedBelow
   ~pytwoway.constraints.Linear
   ~pytwoway.constraints.Monotonic
   ~pytwoway.constraints.NoWorkerTypeInteraction
   ~pytwoway.constraints.QPConstrained
   ~pytwoway.constraints.Stationary
   ~pytwoway.constraints.StationaryFirmTypeVariation

``pytwoway.attrition_utils``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.attrition_utils.AttritionIncreasing
   ~pytwoway.attrition_utils.AttritionDecreasing
