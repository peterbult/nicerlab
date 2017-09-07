import numpy as np
import warnings

__all__ = ['durations', 'verify', 'truncate_below', 'truncate_above',
'truncate', 'good_to_bad', 'bad_to_good', 'merge_gti_or', 'merge_gti_and',
'merge', 'times_in_interval', 'apply_gti']


def _asarray(gti):
    """
    Cast a GTI table into a 2d ndarray.
    
    Parameters
    ----------
    gti: iterable
        Standard list, numpy array or astropy Table of good time intervals

    Returns
    -------
    ndarray
        The GTI table in ndarray format
    """
    arr = np.asarray(gti)
    if(len(arr) ==0 or len(arr.shape) > 1):
        return arr
    else:
        return arr.view(np.float64).reshape(arr.shape + (-1,))

def durations(gti):
    # Cast as ndarray
    gti = _asarray(gti)
    # Ensure content
    if (len(gti) == 0):
        return []
    else:
        start = np.array([row[0] for row in gti])
        stop  = np.array([row[1] for row in gti])
        return stop - start


def verify(gti):
    # Cast as ndarray
    gti = _asarray(gti)
    # Verify
    if np.any(durations(gti)<0):
        warnings.warn("Warning: negative duration interval found")

    if not np.array_equal(gti[:,0], np.sort(gti[:,0])):
        warnings.warn("Warning: GTIs are out of order")

    if np.any(gti[1:,0]-gti[:-1,1] < 0):
        warnings.warn("Warning: some GTIs overlap")


def truncate_below(gti, bound):
    # Treat gti as ndarray
    gti = _asarray(gti)
    # Ensure gti is not empty
    if len(gti) is 0:
        return np.zeros((0,2))
    # Check against last gti
    if (gti[-1,1] <= bound):
        return np.zeros((0,2))
    # Truncate
    trunc = gti[gti[:,1] > bound ]#np.copy(gti[idx:])
    # Alter the lower bound if needed
    if (trunc[0,0] < bound):
        trunc[0,0] = bound
    return trunc


def truncate_above(gti, bound):
    # Treat gti as ndarray
    gti = _asarray(gti)
    # Ensure gti is not empty
    if len(gti) is 0:
        return np.zeros((0,2))
    # Check against first gti
    if (gti[0,0] >= bound):
        return np.zeros((0,2))
    # Find the first stop time above the boundary
    trunc = gti[gti[:,0] < bound]
    # Alter the upper bound if needed
    if (trunc[-1,1] >= bound):
        trunc[-1,1] = bound
    return trunc


def truncate(gti, lower=None, upper=None):
    gti = _asarray(gti)
    if lower is not None:
        gti = truncate_below(gti, lower)

    if upper is not None:
        gti = truncate_above(gti, upper)

    return gti


def good_to_bad(gti, lower=float("-inf"), upper=float('inf')):
    # Treat gti as ndarray
    gti = _asarray(gti)
    # Limit gti to boundaries
    gti = truncate(gti, lower, upper)
    # Ensure gti is not empty
    if len(gti) == 0:
        return [[lower,upper]]
    # Compute bti
    bti = np.array([gti[:-1,1], gti[1:,0]]).T
    # Adjust boundaries
    if ( lower < gti[0,0] ):
        first = np.array([[lower, gti[0,0]]])
        bti = np.concatenate((first, bti), axis=0)

    if ( upper > gti[-1,1] ):
        last = np.array([[gti[-1,1], upper]])
        bti = np.concatenate((bti, last), axis=0)

    return bti


def bad_to_good(bti, lower=float("-inf"), upper=float('inf')):
    # Treat the input as a numpy array
    bti = _asarray(bti)
    # Truncate on the given boundaries
    bti = truncate(bti, lower, upper)
    # Special case on empty input
    if bti.size is 0:
        return np.array([[lower, upper]])
    gti = np.array([bti[:-1,1], bti[1:,0]]).T

    if ( lower < bti[0,0] ):
        first = np.array([[lower, bti[0,0]]])
        gti = np.concatenate((first, gti), axis=0)

    if ( upper > bti[-1,1] ):
        last = np.array([[bti[-1,1], upper]])
        gti = np.concatenate((gti, last), axis=0)

    return gti



def matrix_sort( m, column ):
    return m[m[:,column].argsort()]


def merge_gti_or( gti1, gti2 ):
    # Cast as ndarrays
    gti1 = _asarray(gti1)
    gti2 = _asarray(gti2)

    # Merge the (START,STOP) pairs to a single list
    gti3 = np.concatenate((gti1,gti2))

    # Sort the merged list on START times
    gti3 = matrix_sort( gti3, column=0 )

    # Construct an empty list
    gti4 = []
    # And an interval tracker 
    tracker = gti3[0]

    # Perform forward scan
    for i in np.arange(1,len(gti3)):
        if tracker[1] < gti3[i][0]:     # if this.STOP < next.START
            # No overlap
            gti4.append( tracker )
            tracker = gti3[i]
        elif tracker[1] < gti3[i][1]:   # if next.START < this.STOP < next.STOP
            # Partial overlap
            tracker[1] = gti3[i][1]
            next
        elif tracker[1] > gti3[i][0]:    # if next.STOP < this.STOP
            # Full overlap
            next

    # Add the trailing interval
    gti4.append(tracker)

    return np.array(gti4)


def merge_gti_and( gti1, gti2 ):
    # Cast as ndarrays
    gti1 = _asarray(gti1)
    gti2 = _asarray(gti2)

    # Merge the (START,STOP) pairs to a single list
    gti3 = np.concatenate((gti1,gti2))

    # Sort the merged list on START times
    gti3 = matrix_sort( gti3, 0 )

    # Construct an empty list
    gti4 = []
    # And an interval tracker 
    tracker = gti3[0]

    # Perform forward scan
    for i in np.arange(1,len(gti3)):
        if tracker[1] < gti3[i][0]:     # if this.STOP < next.START
            # No overlap
            tracker = gti3[i]
            next
        elif tracker[1] < gti3[i][1]:   # if next.START < this.STOP < next.STOP
            # Partial overlap
            tracker[0] = gti3[i][0]
            gti4.append(tracker)
            tracker = [tracker[1], gti3[i][1]]
            next
        elif tracker[1] > gti3[i][0]:    # if next.STOP < this.STOP
            # Full overlap
            gti4.append(gti3[i])
            tracker[0] = gti3[i][1]
            next

    return np.array(gti4)


def merge(gtis, method=None):
    # Cast all gtis as ndarray
    gtis = [_asarray(gti) for gti in gtis]
    # Call method
    if method == "and":
        gti = gtis[0]
        for i in range(1,len(gtis)):
            gti = merge_gti_and(gti, gtis[i])
        return gti

    elif method == "or":
        gti = gtis[0]
        for i in range(1,len(gtis)):
            gti = merge_gti_or(gti, gtis[i])
        return gti

    else:
        raise ValueError("Method not recognized: use 'and'/'or'")

def where(t, ti):
    ti = _asarray(ti)

    indices = []
    for i,row in enumerate(ti):
        if t > row[0] and t < row[1]:
            indices.append(i)

    return np.array(indices)

def times_in_interval(times, interval):
    # Cast times as ndarray
    times = _asarray(times)

    # Allocate output
    indices = []

    # Find where
    for i,t in enumerate(times):
        if t > interval[0] and t < interval[1]:
            indices.append(i)

    return np.array(indices)


def apply_gti(times, gti):
    """
    Applies the given GTI filter to a list of event times.
    """
    # Cast times as ndarray
    times = _asarray(times)

    # Allocate the indices
    indices = [times_in_interval(times, ti) for ti in gti]

    return np.array(indices)

