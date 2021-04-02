Using from stata
================

To install from stata, from the stata command line run::

   ssc install statatwoway

The most efficient way to run statatwoway is through python. Please read :doc:`From python <doc-python>` on how to install pytwoway for python.

Sample data: :download:`download <twoway_sample_data.csv>`

To run in stata:

.. code-block:: stata

    statatwoway fe cre, config("config.txt") env("stata-env")

To see all available results, type:

.. code-block:: stata

    return list

You can choose to run fe, cre, or both. Additionally, while a config file is required, an environment is not. However, setting up an environment is recommended. Read about setting up environments in Anaconda `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Use the following example config file as a template. Including `stata=True` automatically sets the correct data and filetypes. statatwoway generates a temporary datafile, and the `stata` option automatically inputs those values. For help about all the options that can be included in your config file, run the following in Terminal::

  pytw -h

Example config.txt::

    stata = True
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
