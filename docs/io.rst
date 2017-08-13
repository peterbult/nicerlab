===
I/O
===

Read
====

Reading a generic fits file
---------------------------

Example::

    >>> import nicerlab as ni
    >>> filename = 'data_cl.evt'
    >>> table, keys = ni.io.read_from_fits(
    ...                 filename, ext='EVENTS', cols=['TIME', 'PI'],
    ...                 keys=['TSTART', 'TSTOP', 'OBJECT'],
    ...                 as_table=True)
    >>> keys
    {'TSTART': 523327317.6109142, 'TSTOP': 523345136.3998488, 'OBJECT': 'TOO'}
    >>> table
    array([[  5.23327318e+08,   6.11000000e+02],
           [  5.23327319e+08,   6.23000000e+02],
           [  5.23327320e+08,   3.02000000e+02],
           ...,
           [  5.23345135e+08,   5.99000000e+02],
           [  5.23345136e+08,   2.89000000e+02],
           [  5.23345136e+08,   2.83000000e+02]]) 

Alternatively we could have also just used astropy as::
    
    >>> from astropy.table import Table
    >>> tb = Table.read(filename, format='fits', hdu=1)['TIME', 'PI']
    >>> tb
    <Table length=5946>
         TIME       PI
       float64    int32
    ------------- -----
    523327317.829   611
    523327319.446   623
    523327320.199   302
              ...   ...
    523345135.417   599
    523345135.787   289
    523345136.275   283
    >>> keys = {k: tb.meta[k] for k in ['TSTART', 'TSTOP', 'OBJECT']}
    >>> keys
    {'TSTART': 523327317.6109142, 'TSTOP': 523345136.3998488, 'OBJECT': 'TOO'}

However, this will load all columns from the fits table into memory, which is
sometimes unwieldy. In any case the ``read_from_fits()`` function is just a
building block used to construct a set of higher level convience functions, and
should seldomly be used directly.


Reading event data
------------------

Events only
^^^^^^^^^^^
If only the event arrival times are of interest these can be obtained by
invoking the fits interface as::

    >>> table = ni.io.read_from_fits(filename, ext='EVENTS', cols=['TIME'])

However, the ``table`` will be a 2-dimensional numpy array with only one defined
column, hence::

    >>> events = table[:,0]

What's more, we will actually want to construct an :ref:`eventlist` object with the
proper attributes set. To that end a convience function has been implemented that does
all this for you::

    >>> evt = ni.io.read_events(filename)
    >>> evt.info()
    Eventlist:
    > counts......: 5946
    > exposure....: 17818.788934648037 seconds
    > start/stop..: 523327317.6109142 / 523345136.3998488
    > MJD.........: 57967.02988188558
    


Events and ...
^^^^^^^^^^^^^^
For convience a number of combination functions are implemented that read event
arrival times *and* some other datum from the fits table::

    >>> evt, gti = ni.io.read_events_and_gti(filename)
    >>> gti
    array([[  5.23327318e+08,   5.23329114e+08],
           [  5.23333245e+08,   5.23334856e+08],
           [  5.23338946e+08,   5.23340281e+08],
           [  5.23344887e+08,   5.23345136e+08]])
    >>>
    >>> tb = ni.io.read_events_and_pi(filename)
    >>> tb
    array([[  5.23327318e+08,   6.11000000e+02],
           [  5.23327319e+08,   6.23000000e+02],
           [  5.23327320e+08,   3.02000000e+02],
           ...,
           [  5.23345135e+08,   5.99000000e+02],
           [  5.23345136e+08,   2.89000000e+02],
           [  5.23345136e+08,   2.83000000e+02]])


Reading good time intervals
---------------------------

You can also read the GTI table only::

    >>> table = ni.io.read_from_fits(filename, ext='GTI', cols=['START', 'STOP'])

which has its own convience function::

    >>> gti = ni.io.read_gti(filename)
    >>> gti
    array([[  5.23327318e+08,   5.23329114e+08],
           [  5.23333245e+08,   5.23334856e+08],
           [  5.23338946e+08,   5.23340281e+08],
           [  5.23344887e+08,   5.23345136e+08]])


Write
=====

Write power spectrum
----------------------

Write a single :ref:`powerspectrum` object to a fits file

.. code-block:: python

    ni.io.write_pds(pds, "pds.fits")


If the pds has multiple rows, then this works too.


Write spectrum
--------------

Write a :ref:`pispectrum` object as an OGIP compatible fits file.

.. code-block:: python
    
    ni.io.write_spectrum(spec, "pi_spectrum.fits")


