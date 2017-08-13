import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import nicerlab.utils as utils

__all__ = ["Lightcurve", "make_light_curve"]

class Lightcurve(np.ndarray):
    """
    Basic Lightcurve ndarray
    """

    def __new__(cls, input_array, dt=1, tstart=None, tstop=None, mjd=0):
        obj = np.asarray(input_array).view(cls)
        obj.dt = dt
        obj.tstart = tstart
        obj.tstop = tstop
        obj.mjd = mjd

        if obj.tstart is None:
            obj.tstart = 0

        if obj.tstop is None:
            obj.tstop = len(obj) * dt

        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        
        self.dt = getattr(obj, 'dt', 1)
        self.tstart = getattr(obj, 'tstart', None)
        self.tstop = getattr(obj, 'tstop', None)
        self.mjd = getattr(obj, 'mjd', None)

        if self.tstart is None:
            self.tstart = 0

        if self.tstop is None:
            self.tstop = (len(self)+1) * self.dt

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self,out_arr,context)

    def timespace(self):
        """
        Generate a time axis of mission-elapsed time.

        Returns
        -------
        iterable: array of len(self) containing start times of the
        light curve bins.
        """
        return self.tstart + np.arange(len(self))*self.dt

    def mjdspace(self):
        """
        Generate a time axis of mjd time.

        Returns
        -------
        iterable: array of len(self) containing start times of the
        light curve bins.
        """
        return self.mjd + (self.timespace()-self.tstart) / 86400.0

    def plot(self, xformat='sec', ax=None, **kwargs):
        """
        Convenience function for plotting the data using matplotlib
        """

        if ax is None:
            ax = plt.gca()

        # Label the axes
        ax.set_xlabel('Time')
        ax.set_ylabel('Counts')

        # Plot the light curves
        if xformat is 'sec':
            ax.plot(self.timespace(), self, color='C0')
        elif xformat is 'mjd':
            ax.plot(self.mjdspace(), self, color='C0')
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))    
        else:
            raise ValueError("Format not recognized: use 'sec'/'mjd'")

        return ax


# def make_light_curve(events, dt, tstart=None, tstop=None, tseg=None):
    # """
    # Construct a contiuous light curve from an array of event arrival times. The
    # events can be out of order.

    # Parameters 
    # ---------- 
    # events: iterable Array-like container of event arrival
        # times.

    # dt: float Time resolution (bin width) of the light curve

    # tstart: float Lower bound of the first time bin

    # tstop: float Upper bound of the final time bin

    # tseg: float Optionally construct the light curve as an array of segments,
        # each being `tseg` in length. 

    # """


    # # Resolve time boundaries if needed
    # # ~ First try to get the tstart attribute from the events array
    # # ~ If no such attribute exists, use the smallest/largest time
    # #   in the events array.
    # if tstart is None:
        # tstart = getattr(events, 'tstart', np.min(events))

    # if tstop is None:
        # tstop = getattr(events, 'tstop', np.max(events))

    # # Revolve the MJD, default to zero
    # mjd = getattr(events, 'mjd', 0)
    # # Adjust the MJD based on the tstart time
    # mjd += (tstart - np.min(events))/86400.0

    # # Apply the truncation
    # events = utils.truncate(events, tstart, tstop)

    # # Safeguard against empty data
    # if len(events) <= 1:
        # return Lightcurve([], dt=dt, tstart=tstart, tstop=tstop, mjd=mjd)

    # # Compute the input exposure
    # exposure = tstop - tstart

    # # Compute the output size
    # number_of_bins = int(np.ceil(exposure/dt))
    # # Note: the number of bins is rounded up, so the segment duration is
    # #       guaranteed to be equal-to or greater-than the input exposure
    # duration = number_of_bins * dt

    # # Allocate output
    # target = np.zeros(number_of_bins)

    # # Construct indices
    # idx = ((events-tstart)/dt).astype(dtype=np.intp)
    # # Collect duplicates
    # i,c = np.unique(idx, return_counts=True)
    # # Filter
    # istart = utils.find_first_of(i,0)
    # istop = utils.find_first_of(i,number_of_bins)
    # if istart is None:
        # pass
    # else:
        # # Broadcast
        # i,c = i[istart:istop], c[istart:istop]
        # # Update the light curve
        # target[i] += c

    # if tseg is not None:
        # # Calculate the number of bins per segment
        # number_of_bins = int(np.ceil(tseg/dt))
        # number_of_segs = int(target.size/number_of_bins)
        # # Drop the tail elements
        # target.resize(number_of_bins*number_of_segs)
        # # Reshape
        # target = target.reshape((number_of_segs, number_of_bins))
    
    # return Lightcurve(target, dt=dt, tstart=tstart, tstop=tstop, mjd=mjd)



def make_light_curve(events, dt, tstart=None, tstop=None, mjd=None):
    """
    Construct a contiuous light curve from an array of event arrival times. The
    events can be out of order.

    Parameters 
    ---------- 
    events: iterable Array-like container of event arrival
        times.

    dt: float Time resolution (bin width) of the light curve

    tstart: float Lower bound of the first time bin

    tstop: float Upper bound of the final time bin

    """


    # Resolve time boundaries if needed
    # ~ First try to get the tstart attribute from the events array
    # ~ If no such attribute exists, use the smallest/largest time
    #   in the events array.
    if tstart is None:
        tstart = getattr(events, 'tstart', np.min(events))

    if tstop is None:
        tstop = getattr(events, 'tstop', np.max(events))

    # Revolve the MJD, default to zero
    if mjd is None:
        mjd = getattr(events, 'mjd', 0)
        # Adjust the MJD based on the tstart time
        mjd += (tstart - np.min(events))/86400.0

    # Apply the truncation
    events = utils.truncate_view(events, tstart, tstop)

    # Safeguard against empty data
    if len(events) <= 1:
        return Lightcurve([], dt=dt, tstart=tstart, tstop=tstop, mjd=mjd)

    # Compute the number of bins
    # Note: Using 'fractional-round' ensures that the event list
    #       exposure covers at least 99% of the final time bin.
    number_of_bins = utils.fround((tstop-tstart)/dt, 0.01)

    # Compute the indices
    indices = np.array((events-tstart)/dt, dtype=np.intp)

    # Compute the counts
    counts = np.bincount(indices, minlength=number_of_bins)[:number_of_bins]

    # Correct the exposure
    tstop = tstart + number_of_bins*dt
    
    # Return the counts as a Lightcurve
    return Lightcurve(counts, dt=dt, tstart=tstart, tstop=tstop, mjd=mjd)

