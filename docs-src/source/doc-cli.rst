Command line interface
======================
 
To install from pip, from the command line run::

  pip install pytwoway

Sample data: :download:`download <twoway_sample_data.csv>`

To run pytwoway using command line interface::

  pytw --my-config config.txt --fe --cre

Example config.txt::

    data = file.csv
    filetype = csv
    col_dict = "{'i': 'your_workerid_col', 'j': 'your_firmid_col', 'y': 'your_compensation_col', 't': 'your_year_col'}"

For help about all the options that can be included in your config file, run the following in Terminal (all options can also be included directly in the command line)::

  pytw -h
