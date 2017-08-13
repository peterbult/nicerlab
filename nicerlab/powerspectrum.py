
import warnings
import itertools
import numpy as np

class Powerspectrum(np.ndarray):
    """
    Basic Powerspectrum ndarray
    """

    def __new__(cls, input_array, df=1, tstart=None, tstop=None, stack=None,
            mjd=0):
        obj = np.asarray(input_array).view(cls)
        obj.df = df
        obj.tstart = tstart
        obj.tstop = tstop
        obj.stack = stack
        obj.mjd = mjd
        return obj

    def __array_wrap__(self, array, context=None):
        return np.ndarray.__array_wrap__(self,array,context)

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        
        self.df = getattr(obj, 'df', None)
        self.tseg = getattr(obj, 'tseg', None)
        self.tstart = getattr(obj, 'tstart', None)
        self.tstop = getattr(obj, 'tstop', None)
        self.stack = getattr(obj, 'stack', None)
        self.mjd = getattr(obj, 'mjd', None)

    def freqspace(self, dc=True):
        if dc is True:
            return np.arange(len(self))*self.df
        else:
            return np.arange(1,len(self))*self.df


def make_power_spectrum(lc, tseg=None, collect=None, dt=None, mjd=None):
    """
    Construct a power spectrum for a given light curve.

    Parameters
    ----------
    lc: iterable
        Array of counting data

    dt: float
        time resolution of the input light curve

    tseg: object
        Time duration of a single Fourier Transform segment. If `tseg=None` the
        entire light curve will be transformed in one go.

    collect: object
        Method used to collect many Fourier Transform segments into one power
        spectrum. Options are    
            -  None: return an array of single-segment spectra
            - "avg": average the power
            - "sum": sum the powers

    Returns
    -------
    pds, stack: iterable, int
        The routine returns two objects; one container for the power spectrum
        data and a second integer indicating how many segments were collected
        into that spectrum.
    """

    # Try to resolve the segment length
    if tseg is None:
        tstart = getattr(lc, 'tstart', None)
        if tstart is None: tstart = 0

        tstop = getattr(lc, 'tstop', None) 
        if tstop is None: tstop = len(lc)*dt
        
        tseg = tstop - tstart

    # Try to resolve the MJD
    if mjd is None:
        mjd = getattr(lc, 'mjd', 0)

    # Try to resolve the dt
    if dt is None:
        dt = getattr(lc, 'dt', 1)

    # Calculate the number of bins per segment
    num = int(np.ceil(tseg/dt))
    # Compute the frequency resolution
    df = 1/(num*dt)
    # Check size
    if np.abs(num*dt-tseg) > 1e-8:
        warnings.warn("sampled segment differs from tseg", FutureWarning)

    # Calculate the number of segments in the light curve
    number_of_segments = int(lc.shape[0]/num)

    # Ensure there is enough data
    if number_of_segments < 1:
        tstart = getattr(lc, 'tstart', None)
        tstop = getattr(lc, 'tstop', None)
        return Powerspectrum([], df=df, tstart=tstart, tstop=tstop, 
                             stack=0, mjd=mjd)

    # Define indices
    istart = num*np.arange(0,number_of_segments,1)
    istop  = num*np.arange(1,number_of_segments+1,1)

    # Construct the power spectra
    pds = np.array([
        (2/np.sum(lc[i0:i1]))
        * np.abs(np.fft.rfft(lc[i0:i1]))**2 
        for i0,i1 in zip(istart,istop)
        ])

    if (len(pds) > 1):
        pds = np.vstack(pds)

    # Collect if needed
    if collect is None or collect is "none":
        pass
    elif collect is "avg":
        pds = np.mean(pds, axis=0)
    elif collect is "sum":
        pds = np.sum(pds, axis=0)
    else:
        raise ValueError("Method not recognized: use None/'avg'/'sum'")

    tstart = getattr(lc, 'tstart', None)
    tstop = getattr(lc, 'tstop', None)
    return Powerspectrum(pds, df=df, tstart=tstart, tstop=tstop, 
                         stack=number_of_segments, mjd=mjd)


def make_co_spectrum(lc, tseg=None, collect=None, dt=None, mjd=None):
    """
    Construct a co-spectrum for a given light curve.

    Parameters
    ----------
    lc: iterable
        Nested array of light curve data. Each array must be aligned in time 
        for results to make sense.

    dt: float
        time resolution of the input light curve

    tseg: object
        Time duration of a single Fourier Transform segment. If `tseg=None` the
        entire light curve will be transformed in one go.

    collect: object
        Method used to collect many Fourier Transform segments into one 
        co-spectrum. Options are
            -  None: return an array of single-segment spectra
            - "avg": average the power
            - "sum": sum the powers

    Returns
    -------
    pds: iterable
        The routine returns a Powerspectrum object; a (possibly nested) 
        ndarray of powers with several added attributes.
    """

    # Ensure lc symmetry
    all_sizes = [ts.size for ts in lc]
    if not np.allclose(all_sizes, all_sizes[0]):
        raise ValueError("light curves do not all have the same size")

    # Ensure coalignment
    all_tstart = [getattr(ts, 'tstart', None) for ts in lc]
    if not np.allclose(all_tstart, all_tstart[0]):
        warnings.warn("tstart does not match across light curves", FutureWarning)

    # Try to resolve the segment length
    if tseg is None:
        tstart = getattr(lc[0], 'tstart', None)
        if tstart is None: tstart = 0

        tstop = getattr(lc[0], 'tstop', None) 
        if tstop is None: tstop = len(lc[0])*dt
        
        tseg = tstop - tstart

    # Try to resolve the MJD
    if mjd is None:
        mjd = getattr(lc, 'mjd', 0)

    # Try to resolve the dt
    if dt is None:
        dt = getattr(lc, 'dt', 1)

    # Calculate the number of bins per segment
    num = int(np.ceil(tseg/dt))
    # Compute the frequency resolution
    df = 1/(num*dt)
    # Check size
    if np.abs(num*dt-tseg) > 1e-8:
        warnings.warn("sampled segment differs from tseg", FutureWarning)

    # Calculate the number of segments in the light curve
    number_of_segments = int(lc[0].shape[0]/num)
    print("num of segs:", number_of_segments)

    # Ensure there is enough data
    if number_of_segments < 1:
        tstart = getattr(lc[0], 'tstart', None)
        tstop = getattr(lc[0], 'tstop', None)
        return Powerspectrum([], df=df, tstart=tstart, tstop=tstop, 
                             stack=0, mjd=mjd)

    # Define indices
    istart = num*np.arange(0,number_of_segments,1)
    istop  = num*np.arange(1,number_of_segments+1,1)
    print('len(istart)', len(istart))


    # Construct the correlation pairs
    combos = np.array(list(itertools.combinations(np.arange(len(lc)),2)))
    print('combos:', len(combos))

    # Define a convenience function
    def calc_cds(tsvec, combos=combos):
        # Compute all transforms
        tras = [np.sqrt(2/np.sum(ts)) * np.fft.rfft(ts) for ts in tsvec]
        # Map out all unique cross correlations
        cds = [np.real( tras[a] * np.conj(tras[b]) ) for a,b in combos]
        # Sum the cross correlations
        return np.sum(cds,axis=0)
    
    # Compute the cospectra
    cds = np.array([
            calc_cds([ts[i0:i1] for ts in lc]) for i0,i1 in zip(istart,istop)
        ])

    # Reduce the superfluous dimensionality
    if (len(cds) > 1):
        cds = np.vstack(cds)

    # Collect if needed
    if collect is None or collect is "none":
        pass
    elif collect is "avg":
        cds = np.mean(cds, axis=0)/len(combos)
    elif collect is "sum":
        cds = np.sum(cds, axis=0)
    else:
        raise ValueError("Method not recognized: use None/'avg'/'sum'")

    tstart = getattr(lc[0], 'tstart', None)
    tstop = getattr(lc[0], 'tstop', None)
    return Powerspectrum(cds, df=df, tstart=tstart, tstop=tstop, 
                         stack=number_of_segments*len(combos), mjd=mjd)



