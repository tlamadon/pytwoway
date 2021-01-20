Command line interface
======================
 
To install from pip, from the command line run::

  pip install pytwoway

To run pytwoway using command line interface::

  pytw --my-config config.txt --fe --cre

Example config.txt::

    data = file.csv
    filetype = csv
    col_dict = "{'fid': 'your_firmid_col', 'wid': 'your_workerid_col', 'year': 'your_year_col', 'comp': 'your_compensation_col'}"

For help about all the options that can be included in your config file (all options can also be included directly in the command line)::

  pytw -h
