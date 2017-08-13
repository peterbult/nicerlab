
import numpy as np
from nicerlab.eventlist import Eventlist

__all__ = ['find_first_of', 'truncate', 'truncate_view', 'fround']

def find_first_of(data, value, start=0):
    """
    Find the index of the first data entry that is greater-than value. This
    function assumes data is a value-sorted array.

    Parameters
    ----------
    data: iterable
        Value-sorted array of data.
    
    value: float
        Boundary value to search for

    start: int
        Left-boundary index. All entries smaller than this index will not be
        searched. This parameter can be used to speed up the search if prior
        knowledge is available.

    Returns
    -------
    index: int
        The index of the first data entry that is greater than `value`. If data elements are below `value`, the function returns `None`.
    """
    # Catch empty data container
    if len(data) == 0:
        return None

    # Catch improper value
    if value is None:
        raise ValueError("improper value passed")

    # Preempt overflow
    if (value > data[-1]):
        return None

    # Otherwise perform a right-to-left search
    return np.searchsorted(data[start:], value) + start

def truncate(data, lower=None, upper=None):
    mjd = getattr(data, 'mjd', 0)
    return Eventlist(truncate_view(data, lower, upper),
                     tstart=lower, tstop=upper, mjd=mjd)

def truncate_view(data, lower=None, upper=None):
    start = find_first_of(data,lower)
    stop = find_first_of(data,upper)
    return data[start:stop]


def fround(x, f=0.5):
    """
    Round a floating-point number to an interger using a fractional offset.

    By default fround will use normal arithmatic rounding, that is,
        [2.50... , 3.49... ] -> 3

    By setting a fractional offset the boundaries of the floating-point 
    domain can be shifted, for instance, setting f=0.7 gives
        [2.30... , 3.29... ] -> 3

    Parameter
    ---------
    x: value
        A floating point value to be rounded.

    f: value
        The fractional shift of the rounding domain. Defaults to 0.5.

    Returns
    -------
    integer
        The rounded integer number associated with x.
    """
    return int(x+f)


