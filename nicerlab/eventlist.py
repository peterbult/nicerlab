import numpy as np

class Eventlist(np.ndarray):
    """
    Basic Eventlist ndarray
    """

    def __new__(cls, input_array, tstart=None, tstop=None, mjd=0):
        obj = np.asarray(input_array).view(cls)
        obj.tstart = tstart
        obj.tstop = tstop
        obj.mjd = mjd
        return obj

    def __array_wrap__(self, array, context=None):
        return np.ndarray.__array_wrap__(self,array,context)

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        
        self.tstart = getattr(obj, 'tstart', None)
        self.tstop = getattr(obj, 'tstop', None)
        self.mjd = getattr(obj, 'mjd', None)

    def info(self):
        """
        Print summary information on the Eventlist object 
        """

        print("Eventlist:")
        print("> counts......:", self.shape[0])
        print("> exposure....:", self.tstop-self.tstart, 'seconds')
        print("> start/stop..:", self.tstart, "/", self.tstop)
        print("> MJD.........:", self.mjd)


