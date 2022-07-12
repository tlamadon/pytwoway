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
   # Clean data and collapse it at the worker-firm spell level
   clean_params = bpd.clean_params({'connectedness': 'leave_out_spell', 'collapse_at_connectedness_measure': True, 'drop_single_stayers': True})
   bdf = bdf.clean(clean_params)
   # Initialize FE estimator
   fe_params = tw.fe_params({'he': True})
   fe_estimator = tw.FEEstimator(bdf, fe_params)
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
   clean_params = bpd.clean_params({'connectedness': 'connected'})
   bdf = bdf.clean(clean_params)
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

- If you want to estimate Sorkin:

.. code-block:: python

   import pandas as pd
   import bipartitepandas as bpd
   import pytwoway as tw
   # Load data into Pandas DataFrame
   df = pd.read_csv(filepath)
   # Convert into BipartitePandas DataFrame
   bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['y'], t=df['t'])
   # Clean data and collapse it at the worker-firm spell level
   clean_params = bpd.clean_params({'connectedness': 'strongly_connected'})
   bdf = bdf.clean(clean_params).collapse()
   # Convert to event study format
   bdf = bdf.to_eventstudy()
   # Initialize Sorkin estimator
   sorkin_estimator = tw.SorkinEstimator()
   # Fit Sorkin estimator
   sorkin_estimator.fit(bdf)
   # Investigate results
   print(sorkin_estimator.V_EE)

- If you want to estimate Borovickova-Shimer:

.. code-block:: python

   import pandas as pd
   import bipartitepandas as bpd
   import pytwoway as tw
   # Load data into Pandas DataFrame
   df = pd.read_csv(filepath)
   # Convert into BipartitePandas DataFrame
   bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['y'], t=df['t'])
   # Clean data and collapse it at the worker-firm spell level
   bdf = bdf.clean().collapse()
   ## Make sure all workers and firms have at least 2 observations ##
   prev_len = 0
   while prev_len != len(bdf):
      prev_len = len(bdf)

      # Drop stayers
      bdf = bdf.loc[bdf.get_worker_m(is_sorted=True), :].clean()

      # Drop firms with a single observation
      bdf = bdf.min_obs_frame(is_sorted=True, copy=False).clean()
   # Initialize Borovickova-Shimer estimator
   bs_estimator = tw.BSEstimator()
   # Fit Borovickova-Shimer estimator
   bs_estimator.fit(bdf)
   # Investigate results
   print(bs_estimator.res)

Check out the notebooks for more detailed examples!
