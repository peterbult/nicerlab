.. nicerlab documentation master file, created by
   sphinx-quickstart on Wed Aug  9 10:54:14 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. :tocdepth: 3

####################################
Welcome to nicerlab's documentation!
####################################

The nicerlab package offers x-ray timing analysis tools for python. The goal of
nicerlab is not to present a comprehensive timing library, but rather to give a
framework of efficient processing methods and convenient data classes. These tools
are intended to be used as building blocks for a customized python-based pipeline.

To achieve flexibility, the data objects of nicerlab are derived from numpy's array
object. This allows, for instance, a :ref:`lightcurve` object to be reshaped and
binned as though it is simply a multi-dimensional ``ndarray``.

The I/O operations of nicerlab use astropy to interface directly with fits
formatted data. 

To give a taste of what a nicerlab implementation might look like, the following
code block reads a standard event list and computes a power spectrum for each
good time interval.

.. code-block:: python

    import nicerlab as ni

    filename = "my_file.fits"
    
    events, gti = ni.read_events_and_gti(filename)

    light_curves = [
        ni.make_light_curve(events, dt=1, tstart=t0, tstop=t1) for t0,t1 in gti
    ]

    power_spectra = [
        ni.make_power_spectrum(lc, tseg=64, collect='avg') for lc in light_curves
    ]

    for i,pds in enumerate(power_spectra):
        ni.io.write_pds(pds, "pds{}.fits".format(i))


.. warning::

   While most of the current implementation will work as expected, nicerlab is
   still in development. Some 



.. _documentation:

User Documentation
******************

.. toctree::
   :maxdepth: 2

   install
   data_structures
   io
   utilities


Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
