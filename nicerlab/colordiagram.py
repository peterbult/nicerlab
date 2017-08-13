
import numpy as np

def make_color_diagram(events, dt, tstart=None, tstop=None, 
                       soft=[[ 40,100], [100, 150]],
                       hard=[[150,400], [400,1200]]):
    """
    Construct a color diagram

    Parameters
    ----------
    events: astropy table
        Table containing event times and PI channels

    dt: float
        time resolution of color diagram

    tstart: float
        lower temporal boundary of the color diagram

    tstop: float
        upper temporal boundary of the color diagram

    soft:
        PI channel boundaries of the soft bands

    hard:
        PI channel boundaries of the hard bands
    """
    

    def pi_where(col, lower, upper):
        return np.where(np.logical_and(col>lower,col<upper))

    # Get the indices
    soft_1_idx = pi_where(events['PI'], *soft[0])
    soft_2_idx = pi_where(events['PI'], *soft[1])
    hard_1_idx = pi_where(events['PI'], *hard[0])
    hard_2_idx = pi_where(events['PI'], *hard[1])

    tmp = np.sort(np.array(events[hard_2_idx]['TIME']))

    # Create the light curves
    soft_1_lc = make_light_curve(
            np.array(events[soft_1_idx]['TIME']),
            dt=dt, tstart=tstart, tstop=tstop)
    soft_2_lc = make_light_curve(events[soft_2_idx]['TIME'], dt=dt,
            tstart=tstart, tstop=tstop)
    hard_1_lc = make_light_curve(events[hard_1_idx]['TIME'], dt=dt,
            tstart=tstart, tstop=tstop)
    hard_2_lc = make_light_curve(np.sort(tmp), dt=dt,tstart=tstart, tstop=tstop)

    soft_ratio = soft_2_lc / soft_1_lc
    hard_ratio = hard_2_lc / hard_1_lc
    intensity  = np.sum([soft_1_lc, soft_2_lc, hard_1_lc, hard_2_lc], axis=0)

    return soft_ratio, hard_ratio, intensity


