Command line interface 
======================
 
To install from pip, run::

  pip install pytwoway

To run using command line interface::

  pytw --my-config config.txt --fe --cre

Example config.txt::

    data = file.csv
    filetype = csv
    col_dict = "{'fid': 'your_firmid_col', 'wid': 'your_workerid_col', 'year': 'your_year_col', 'comp': 'your_compensation_col'}"