Using from Python
=================

To install via pip, from the command line run::

   pip install pytwoway

To install via Conda, from the command line run::

   conda install -c tlamadon pytwoway

Sample data: :download:`download <twoway_sample_data.csv>`

To run in Python:

- If you want to estimate AKM and its bias corrections:

.. code-block:: python

   import pandas as pd
   import bipartitepandas as bpd
   import pytwoway as tw
   # Load data into Pandas DataFrame
   df = pd.read_csv(filepath)
   # Convert into BipartitePandas DataFrame
   bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['y'], t=df['t'])
   # Clean data
   bdf = bdf.clean()
   # Collapse data at the worker-firm spell level
   bdf = bdf.collapse()
   # Initialize FE estimator
   fe_estimator = tw.FEEstimator(bdf)
   # Fit FE estimator
   fe_estimator.fit()
   # Investigate results
   print(fe_estimator.summary)

- If you want to estimate CRE:

.. code-block:: python

   import pandas as pd
   import bipartitepandas as bpd
   import pytwoway as tw
   # Load data into Pandas DataFrame
   df = pd.read_csv(filepath)
   # Convert into BipartitePandas DataFrame
   bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['y'], t=df['t'])
   # Clean data
   bdf = bdf.clean()
   # Collapse data at the worker-firm spell level
   bdf = bdf.collapse()
   # Cluster
   bdf = bdf.cluster()
   # Convert to cross-section format
   bdf = bdf.to_eventstudy().get_cs()
   # Initialize CRE estimator
   cre_estimator = tw.CREEstimator(bdf)
   # Fit CRE estimator
   cre_estimator.fit()
   # Investigate results
   print(cre_estimator.summary)

- If you want to estimate BLM:

.. code-block:: python

   import pandas as pd
   import bipartitepandas as bpd
   import pytwoway as tw
   # Load data into Pandas DataFrame
   df = pd.read_csv(filepath)
   # Convert into BipartitePandas DataFrame
   bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['y'], t=df['t'])
   # Clean data
   bdf = bdf.clean()
   # Collapse data at the worker-firm spell level
   bdf = bdf.collapse()
   # Cluster
   bdf = bdf.cluster()
   # Convert to event study format
   bdf = bdf.to_eventstudy()
   # Separate movers and stayers
   movers = bdf.get_worker_m()
   jdata = sim_data_observed.loc[movers, :]
   sdata = sim_data_observed.loc[~movers, :]
   # Initialize BLM estimator
   blm_estimator = tw.BLMEstimator(bdf)
   # Fit BLM estimator
   blm_estimator.fit(jdata, sdata)
   # Plot results
   blm_estimator.plot_log_earnings()
   blm_estimator.plot_type_proportions()

Check out the notebooks for more detailed examples!
