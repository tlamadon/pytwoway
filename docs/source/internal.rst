==========
Python API
==========

Overview
---------

The main PyTwoWay API is split into multiple classes for estimating models and one for simulating BLM data. It also has three modules: one for constructing the variance-covariance matrix for AKM and its bias corrections; one for generating constraints for BLM; and one for generating attrition plots using increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers. PyTwoWay is canonically imported using

  .. code-block:: python

    import pytwoway as tw

Classes
~~~~~~~

* ``pytwoway.FEEstimator``: Class for estimating AKM and its bias corrections without controls

* ``pytwoway.FEControlEstimator``: Class for estimating AKM and its bias corrections with controls

* ``pytwoway.CREEstimator``: Class for estimating the CRE model

* ``pytwoway.BLMModel``: Class for estimating the BLM model with one set of starting values

* ``pytwoway.BLMEstimator``: Class for estimating the BLM model with multiple sets of starting values

* ``pytwoway.BLMBootstrap``: Class for estimating the BLM model with bootstrapped confidence intervals

* ``pytwoway.BLMVarianceDecomposition``: Class for estimating the variance decomposition of the BLM model with bootstrapped confidence intervals

* ``pytwoway.InteractedBLMModel``: Class for estimating the interacted BLM model

* ``pytwoway.SorkinEstimator``: Class for estimating the Sorkin fixed-point revealed preference model

* ``pytwoway.Attrition``: Class for generating attrition plots

* ``pytwoway.MonteCarlo``: Class for running Monte Carlo estimations

* ``pytwoway.SimBLM``: Class for simulating BLM data

Modules
~~~~~~~

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

``pytwoway.FEControlEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.FEControlEstimator
   ~pytwoway.FEControlEstimator.fit

``pytwoway.CREEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.CREEstimator
   ~pytwoway.CREEstimator.fit

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

``pytwoway.BLMVarianceDecomposition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.BLMVarianceDecomposition
   ~pytwoway.BLMVarianceDecomposition.fit
   ~pytwoway.BLMVarianceDecomposition.table

``pytwoway.InteractedBLMModel``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.InteractedBLMModel
   ~pytwoway.InteractedBLMModel.fit_b_fixed_point
   ~pytwoway.InteractedBLMModel.fit_b_linear

``pytwoway.SorkinEstimator``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~pytwoway.SorkinEstimator
   ~pytwoway.SorkinEstimator.fit

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

   ~pytwoway.Q.CovCovariate
   ~pytwoway.Q.CovPsiAlpha
   ~pytwoway.Q.CovPsiPrevPsiNext
   ~pytwoway.Q.VarAlpha
   ~pytwoway.Q.VarCovariate
   ~pytwoway.Q.VarPsi
   ~pytwoway.Q.VarPsiPlusAlpha

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
