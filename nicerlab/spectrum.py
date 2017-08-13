
import numpy as np
import nicerlab.utils as utils

class Spectrum(np.ndarray):
    """
    Basic energy Spectrum ndarray
    """

    def __new__(cls, input_array, cstart=None, cstop=None, name='PI'):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.cstart = cstart
        obj.cstop = cstop
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        
        self.name = getattr(obj, 'name', None)
        self.cstart = getattr(obj, 'cstart', None)
        self.cstop = getattr(obj, 'cstop', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self,out_arr,context)

    def channelspace(self):
        return np.arange(self.cstart,self.cstop)
        # return np.linspace(self.tstart,self.tstop, self.size)


def make_spectrum(events, cstart=None, cstop=None, name='PI'):
    # Cast events as ndarray
    events = np.asarray(events)

    # Process channel boundaries
    if cstart is None:
        cstart = np.min(events)
    if cstop is None:
        cstop = np.max(events)

    # Truncate on boundaries
    events = utils.truncate(events, cstart, cstop)

    # Construct the hist
    data = np.bincount(events, minlength=cstop)

    # return as Spectrum
    return Spectrum(data[cstart:cstop], cstart=cstart, cstop=cstop, name=name)

