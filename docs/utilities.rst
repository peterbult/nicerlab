=========
Utilities
=========

Utils
-----
The utils module implements a set of convenience functions that are used
througout the library. 

- find_first_of: This function will find the index of the first element in an
  array that exceeds a given threshold value. The function is used a number 
  of times to dissect light curve or event data into discrete blocks.

- truncate: This function will 


GTI tools
---------

The GTI Tools subpackages offers a set of functions that manage good time
intervals. Available tools are

- durations -- computes the GTI exposures
- trunctate -- truncate a list of GTIs to a lower and/or upper boundary in time.
- good_to_bad -- convert between good time and bad time
- bad_to_good -- convert between good time and bad time
- merge -- merge two GTI lists using and/or logic



Ftools
------

The Ftools subpackage wraps around the heasoft FTOOLS and allows the user to
call a number of operations on a fits file. These ftool operations are performed
in a subproccess outside the python environment, but can be useful in preparing
or cleaning the data for analysis.

