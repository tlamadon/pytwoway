Using from Stata
================

To install via Stata, from the Stata command line run::

   ssc install statatwoway

The most efficient way to run StataTwoWay is through Python. Please read :doc:`From Python <doc-python>` on how to install PyTwoWay for Python. A version that does not depend on Python is currently in development.

Sample data: :download:`download <twoway_sample_data.csv>`

To run in Stata:

.. code-block:: stata

    statatwoway fe cre, config("config.txt") env("stata-env")

To see all available results, type:

.. code-block:: stata

    return list

The estimators you can run are FE, CRE, or both. Additionally, while a config file is required, an environment is not. However, setting up an environment is recommended. Read about setting up environments in Anaconda `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Use the following example config file as a template. There is no need to include `data` or `filetype` options. StataTwoWay generates a temporary datafile and automatically inputs the name of the generated dataset and filetype `dta`. For help about all the options that can be included in your config file, from the command line run::

  pytw -h

Example config.txt::

    col_dict = "{'i': 'your_workerid_col', 'j': 'your_firmid_col', 'y': 'your_compensation_col', 't': 'your_year_col'}"

.. note::
   Your data must include the following columns:
    - ``i``: the worker identifier
    - ``j``: the firm identifier
    - ``y``: the compensation variable
    - ``t``: the time
   .. list-table:: Example data
      :widths: 25 25 25 25
      :header-rows: 1
      :align: center

      * - i
        - j
        - y
        - t

      * - 1
        - 1
        - 1000
        - 2019
      * - 1
        - 2
        - 1500
        - 2020
      * - 2
        - 3
        - 500
        - 2019
      * - 2
        - 3
        - 550
        - 2020
