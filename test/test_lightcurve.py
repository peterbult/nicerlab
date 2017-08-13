import sys
sys.path.insert(1, '../')

import pytest
import numpy as np
from astropy.table import Table, Column
from nicerlab.eventlist import Eventlist
from nicerlab.lightcurve import Lightcurve, make_light_curve

def almost_equal(x, y, threshold=1e-8):
    return abs(x-y) < threshold

# Construct 100 events uniformly distributed between
# TSTART = 1 and TSTOP = 2.
@pytest.fixture()
def events():
    return np.array([1.01258462, 1.01975451, 1.06750013, 1.06882107, 1.07572868,
                     1.08728431, 1.09358261, 1.09818108, 1.1023316 , 1.10713287,
                     1.11194692, 1.11299286, 1.1289159 , 1.13725379, 1.13877137,
                     1.14470099, 1.14765553, 1.15294236, 1.15965663, 1.16153602,
                     1.17230595, 1.18953611, 1.1942027 , 1.20249874, 1.21323518,
                     1.21411096, 1.22823101, 1.24059445, 1.24193266, 1.24613101,
                     1.27231591, 1.2777865 , 1.28334258, 1.28366799, 1.29572758,
                     1.29600673, 1.298876  , 1.30317705, 1.30844397, 1.31972218,
                     1.32691716, 1.32886126, 1.35300679, 1.35863108, 1.36491073,
                     1.36958479, 1.37819245, 1.41162677, 1.41249513, 1.43109174,
                     1.43140554, 1.44149657, 1.48074941, 1.51572285, 1.52083116,
                     1.5417945 , 1.54438134, 1.5505166 , 1.55756693, 1.57236062,
                     1.57705913, 1.61488011, 1.61521356, 1.63692977, 1.68311352,
                     1.69121445, 1.69160973, 1.71071567, 1.71820649, 1.71891157,
                     1.72131909, 1.72987036, 1.73203469, 1.73328469, 1.75946729,
                     1.76122847, 1.76133876, 1.77467171, 1.78482694, 1.78627373,
                     1.80739794, 1.8413353 , 1.85073929, 1.86126142, 1.86433452,
                     1.869859  , 1.88297339, 1.89916245, 1.89919236, 1.90221299,
                     1.90423977, 1.94183295, 1.94571893, 1.96240905, 1.97028919,
                     1.9754354 , 1.97928873, 1.98582233, 1.99853552, 1.99941076])

@pytest.fixture()
def eventlist(events):
    return Eventlist(events, tstart=1.0, tstop=2.0, mjd=55750.0)

@pytest.fixture()
def eventtable(events):
    tb = Table()
    tb['TIME'] = Column(events)
    return tb

def test_make_lightcurve_1(events):
    lc = make_light_curve(events, dt=0.2, tstart=1, tstop=2)
    assert np.sum(lc) == 100
    assert len(lc) == 5
    assert lc.tstart == pytest.approx(1)
    assert lc.tstop == pytest.approx(2)
    assert lc.mjd == pytest.approx(0, abs=1e-6)
    assert lc.dt == pytest.approx(0.2)

def test_make_lightcurve_2(events):
    lc = make_light_curve(events, dt=0.2, tstart=0, tstop=5, mjd=55000)
    assert np.sum(lc) == 100
    assert len(lc) == 25
    assert lc.tstart == pytest.approx(0)
    assert lc.tstop == pytest.approx(5)
    assert lc.mjd == pytest.approx(55000, abs=1e-6)
    assert lc.dt == pytest.approx(0.2)
    assert np.allclose(lc[0:5], np.zeros(5))
    assert np.allclose(lc[10:25], np.zeros(15))
    assert np.allclose(lc[5:10], [23,24,14,19,20])

def test_make_lightcurve_3(eventlist):
    lc = make_light_curve(eventlist, dt=0.25)
    assert np.sum(lc) == 100
    assert len(lc) == 4
    assert lc.tstart == pytest.approx(1)
    assert lc.tstop == pytest.approx(2)
    assert lc.mjd == pytest.approx(55750)
    assert lc.dt == pytest.approx(0.25)
    assert np.allclose(lc, [30, 23, 21, 26])

def test_make_lightcurve_4(eventlist):
    lc = make_light_curve(eventlist, dt=0.251)
    assert np.sum(lc) == 74
    assert len(lc) == 3
    assert lc.tstart == pytest.approx(1.000)
    assert lc.tstop == pytest.approx(1.753)
    assert lc.mjd == pytest.approx(55750)
    assert lc.dt == pytest.approx(0.251)

def test_make_lightcurve_5(eventlist):
    lc = make_light_curve(eventlist, dt=0.275)
    assert np.sum(lc) == 81
    assert len(lc) == 3
    assert lc.tstart == pytest.approx(1.000)
    assert lc.tstop == pytest.approx(1.825)
    assert lc.mjd == pytest.approx(55750)
    assert lc.dt == pytest.approx(0.275)

def test_make_lightcurve_6(eventtable):
    lc = make_light_curve(eventtable['TIME'], dt=0.2, tstart=1, tstop=2)
    assert np.sum(lc) == 100
    assert len(lc) == 5
    assert lc.tstart == pytest.approx(1)
    assert lc.tstop == pytest.approx(2)
    assert lc.mjd == pytest.approx(0, abs=1e-6)
    assert lc.dt == pytest.approx(0.2)



def test_light_curve_rebin(events):
    lc1 = make_light_curve(events, dt=0.1, tstart=1, tstop=2)
    lc2 = np.sum(lc1.reshape(-1,2), axis=1)
    lc3 = make_light_curve(events, dt=0.2, tstart=1, tstop=2)
    assert lc2.tstart == pytest.approx(1)
    assert lc2.tstop == pytest.approx(2)
    assert np.allclose(lc2, lc3)



