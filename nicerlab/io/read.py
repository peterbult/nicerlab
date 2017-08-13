__all__ = ['read_from_fits', 'read_events', 'read_gti', 'read_events_and_gti',
'read_pi', 'read_events_and_pi']

import numpy as np
from astropy.io import fits
from astropy.table import Table
from nicerlab.eventlist import Eventlist

def read_from_fits(filename, cols, ext=1, keywords=[], as_table=False):
    """
    Reads selected columns from a fits file.

    Parameters
    ----------
    filename: string
        Filename of the input fits file

    cols: iterable
        A list or array containing the column indices to be read.

    ext: object
        Fits extension identifier, accepts the HDU name or the integer index.

    Returns
    -------
    numpy.array()
        A multidimensional numpy array in row-major order. That is, 
            row = array[0]
        gives the first row of the fits table and 
            len(array[0]) == len(cols)

    """

    hdulist = fits.open(filename)
    keys = {key: hdulist[ext].header[key] for key in keywords}
    table = [hdulist[ext].data.field(i) for i in cols]
    hdulist.close()

    if as_table:
        if len(keywords) == 0:
            return Table(table, names=cols)
        else:
            return Table(table, names=cols), keys

    if len(keywords) == 0:
        return np.array(table).T
    else:
        return np.array(table).T, keys


def read_events(filename):
    """
    Reads the TIME column from the EVENTS table in a fits file.

    Parameters
    ----------
    filename: string
        Input filename
    """
    
    evt, keys = read_from_fits(filename, cols=['TIME'], ext='EVENTS',
            keywords=['MJDREFI', 'MJDREFF', 'TSTART', 'TSTOP'])

    mjd = keys['MJDREFI'] + keys['MJDREFF'] + keys['TSTART']/86400.0
    return Eventlist(evt[:,0], tstart=keys['TSTART'], tstop=keys['TSTOP'],
                    mjd=mjd)


def read_gti(filename):
    """
    Reads the GTI times from a fits file.

    Parameters
    ----------
    filename: string
        Input filename
    """

    return read_from_fits(filename, cols=['START', 'STOP'], ext='GTI')


def read_events_and_gti(filename):
    events = read_events(filename)
    gti = read_gti(filename)
    return events, gti


def read_pi(filename):
    counts = read_from_fits(filename, cols=['PI'], ext='EVENTS')
    return counts[:,0]


def read_events_and_pi(filename):
    return read_from_fits(filename, cols=['TIME', 'PI'], ext='EVENTS')


