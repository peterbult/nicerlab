=======
Scripts
=======

ni-lightcurve
-------------

The `ni-lightcurve` script is a simple nicerlab implementation for constructing
light curves from event files. Basic usage is

.. code-block:: bash

   ni-lightcurve evt.meta 

By default `ni-lightcurve` will output
count-rates as a function of MJD using a 16 second time resolution. All data is
written to a single ASCII file named `lc.dat`, with a continuous block for each
good time interval. Good time intervals are read from the fits file.

+---------------+-----------------------------------------------+
| ``--dt``      |  Set the time resolution                      |
+---------------+-----------------------------------------------+
| ``--clobber`` |  Overwrite the outputfile is it exists        |
+---------------+-----------------------------------------------+
| ``--met``     |  Format the time axis as mission-elapsed time |
+---------------+-----------------------------------------------+
| ``--output``  |  Set the output file name                     |
+---------------+-----------------------------------------------+
| ``--help``    |  Display usage instructions                   |
+---------------+-----------------------------------------------+



gti-select
----------

The `gti-select` tool gives an interactive method for shaping an event file's
good time interval table. Basic usage is as

.. code-block:: bash

   gti-select evt.meta --dt 1



+---------------+-----------------------------------------------+
| ``--dt``      |  Set the time resolution                      |
+---------------+-----------------------------------------------+
| ``--clobber`` |  Overwrite the eventfile internal GTI table   |
+---------------+-----------------------------------------------+
| ``--help``    |  Display usage instructions                   |
+---------------+-----------------------------------------------+

After initialization `gti-select` will iterate through the event files listed
in the metafile. For each event file it will construct a light curve at the
requested resolution and produce a plot with the current GTI's overlayed. One
can then use the mouse to shape the good time intervals, and the terminal to
move to the next event file. The available controls are

+---------------+-----------------------------------------------+
| left-mouse    |  Add bad time                                 |
+---------------+-----------------------------------------------+
| right-mouse   |  Add good time                                |
+---------------+-----------------------------------------------+
| type 'undo'   |  Reset the current event file                 |
+---------------+-----------------------------------------------+
| type 'quit'   |  Quit without saving current file             |
+---------------+-----------------------------------------------+
| type <enter>  |  Save and go to next file                     |
+---------------+-----------------------------------------------+


