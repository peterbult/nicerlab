import sys
sys.path.insert(1, '../')

import pytest
import numpy as np
import nicerlab.gtitools as gtitools

@pytest.fixture()
def gti():
    return [[0.075, 0.099],
            [0.166, 0.237],
            [0.244, 0.389]]


#
# durations()
#
def test_durations1(gti):
    d = gtitools.durations(gti)
    assert np.allclose(d,[ 0.024,  0.071,  0.145])

def test_durations2():
    d = gtitools.durations([])
    assert len(d) == 0



#
# truncate_below()
#
testdata1 = [
    ('gti', 0.238, [[0.244, 0.389]]), # trunc between gtis
    ('gti', 0.400,  np.zeros((0,2))),            # trunc above gtis 
    ('gti', 0.200, [[0.200, 0.237],
                    [0.244, 0.389]]), # trunc in gti
    ('gti', 0.001, [[0.075, 0.099],
                    [0.166, 0.237],
                    [0.244, 0.389]])  # trunc below gtis
]
@pytest.mark.parametrize('gti,bound,expected', testdata1, indirect=['gti'])
def test_truncate_below(gti,bound,expected):
    g = gtitools.truncate_below(gti, bound)
    assert np.allclose(g, expected)



#
# truncate_above()
#
testdata2 = [
    ('gti', 0.140, [[0.075, 0.099]]), # trunc between gtis
    ('gti', 0.001,  np.zeros((0,2))),            # trunc below gtis 
    ('gti', 0.200, [[0.075, 0.099],
                    [0.166, 0.200]]), # trunc in gti
    ('gti', 0.400, [[0.075, 0.099],
                    [0.166, 0.237],
                    [0.244, 0.389]])  # trunc above gtis
]
@pytest.mark.parametrize('gti,bound,expected', testdata2, indirect=['gti'])
def test_truncate_above(gti,bound,expected):
    g = gtitools.truncate_above(gti, bound)
    assert np.allclose(g, expected)



def test_truncate():
    a = []
    b = gtitools.truncate(a)
    assert np.allclose(a,b)


#
# good_to_bad()
#
minf = float("-inf")
pinf = float("+inf")
testdata3 = [
    ('gti', minf, pinf, [[ minf, 0.075],
                         [0.099, 0.166],
                         [0.237, 0.244],
                         [0.389,  pinf]]), # default call
    ('gti', 0.00, 0.40, [[0.000, 0.075],
                         [0.099, 0.166],
                         [0.237, 0.244],
                         [0.389, 0.400]]), # fixed boundaries
    ('gti', 0.00, 0.20, [[0.000, 0.075],
                         [0.099, 0.166]]), # with in-bti truncation
    ('gti', minf, 0.10, [[ minf, 0.075],
                         [0.099, 0.100]]), # with in-gti truncation
    ('gti', minf,-0.10, [[ minf,-0.100]]),            # filter all
    ('gti', 0.50, pinf, [[0.500,  pinf]])             # filter all
]
@pytest.mark.parametrize('gti,lower,upper,expected', testdata3, indirect=['gti'])
def test_good_to_bad(gti,lower,upper,expected):
    bti = gtitools.good_to_bad(gti, lower, upper)
    assert np.allclose(np.ravel(bti), np.ravel(expected))



#
# bad_to_good()
#
testdata4 = [
    ('gti', minf, pinf, [[ minf, 0.075],
                         [0.099, 0.166],
                         [0.237, 0.244],
                         [0.389,  pinf]]), # default call
    ('gti', 0.00, 0.40, [[0.000, 0.075],
                         [0.099, 0.166],
                         [0.237, 0.244],
                         [0.389, 0.400]]), # fixed boundaries
    ('gti', 0.00, 0.20, [[0.000, 0.075],
                         [0.099, 0.166]]), # with in-bti truncation
    ('gti', minf, 0.10, [[ minf, 0.075],
                         [0.099, 0.100]]), # with in-gti truncation
    ('gti', minf,-0.10, [[ minf,-0.100]]),            # filter all
    ('gti', 0.50, pinf, [[0.500,  pinf]])             # filter all
]
@pytest.mark.parametrize('gti,lower,upper,expected', testdata4, indirect=['gti'])
def test_bad_to_good(gti,lower,upper,expected):
    bti = gtitools.good_to_bad(gti, lower, upper)
    assert np.allclose(np.ravel(bti), np.ravel(expected))



#
# merge_gti_or()
# > Merges two GTI lists accepting any time allowed by either list.
#
testdata5 = [
    # Merge a disjoint interval into a GTI
    # > Add to front
    ('gti', [[0.010, 0.050]], [[0.010, 0.050],
                               [0.075, 0.099],
                               [0.166, 0.237],
                               [0.244, 0.389]]),
    # > Add to middle
    ('gti', [[0.100, 0.150]], [[0.075, 0.099],
                               [0.100, 0.150],
                               [0.166, 0.237],
                               [0.244, 0.389]]),
    # > Add to back
    ('gti', [[0.400, 0.500]], [[0.075, 0.099],
                               [0.166, 0.237],
                               [0.244, 0.389],
                               [0.400, 0.500]]),
    # Merge an interval with a 1-sided overlap
    # > Front-right overlap
    ('gti', [[0.000, 0.080]], [[0.000, 0.099],
                               [0.166, 0.237],
                               [0.244, 0.389]]),
    # > Middle-right overlap
    ('gti', [[0.120, 0.175]], [[0.075, 0.099],
                               [0.120, 0.237],
                               [0.244, 0.389]]),
    # > Middle-left overlap
    ('gti', [[0.175, 0.240]], [[0.075, 0.099],
                               [0.166, 0.240],
                               [0.244, 0.389]]),
    # > Back-left overlap
    ('gti', [[0.300, 0.500]], [[0.075, 0.099],
                               [0.166, 0.237],
                               [0.244, 0.500]]), 
    # Merge an interval with a 2-sided overlap
    # > Middle-right-left
    ('gti', [[0.100, 0.240]], [[0.075, 0.099],
                               [0.100, 0.240],
                               [0.244, 0.389]]),
    # Merge an interal that spans one-or-more existing intervals
    # > Span front
    ('gti', [[0.050, 0.100]], [[0.050, 0.100],
                               [0.166, 0.237],
                               [0.244, 0.389]]),
    # > Span front+1
    ('gti', [[0.050, 0.175]], [[0.050, 0.237],
                               [0.244, 0.389]]),
    # > Span middle
    ('gti', [[0.150, 0.240]], [[0.075, 0.099],
                               [0.150, 0.240],
                               [0.244, 0.389]]),
    # > Span middle+1
    ('gti', [[0.150, 0.250]], [[0.075, 0.099],
                               [0.150, 0.389]]),
    # > Span back
    ('gti', [[0.240, 0.400]], [[0.075, 0.099],
                               [0.166, 0.237],
                               [0.240, 0.400]]),
    # > Span back+1
    ('gti', [[0.175, 0.400]], [[0.075, 0.099],
                               [0.166, 0.400]]),
    # Merge multiple intervals into a GTI
    # > Disjoint intervals
    ('gti', [[0.000, 0.050], 
             [0.400, 0.500]], [[0.000, 0.050],
                               [0.075, 0.099],
                               [0.166, 0.237],
                               [0.244, 0.389],
                               [0.400, 0.500]]),
    # > Disjoint and overlapping
    ('gti', [[0.000, 0.050], 
             [0.200, 0.400]], [[0.000, 0.050],
                               [0.075, 0.099],
                               [0.166, 0.400]]),
    # > Overlapping intervals
    ('gti', [[0.000, 0.100], 
             [0.200, 0.400]], [[0.000, 0.100],
                               [0.166, 0.400]]),
]

@pytest.mark.parametrize('gti,other,expected', testdata5, indirect=['gti'])
def test_merge_gti_or(gti, other, expected):
    out = gtitools.merge_gti_or(gti, other)
    assert np.allclose(np.ravel(out), np.ravel(expected))



#
# merge_gti_and()
# > Merges two GTI lists accepting only times allowed by both list.
#
testdata6 = [
    # Merge a disjoint interval into a GTI
    # > Add to front
    ('gti', [[0.010, 0.050]], np.zeros((0,2))),
    # > Add to middle
    ('gti', [[0.100, 0.150]], np.zeros((0,2))),
    ('gti', [[0.010, 0.050]], np.zeros((0,2))),
    # > Add to back
    ('gti', [[0.400, 0.500]], np.zeros((0,2))),
    # Merge an interval with a 1-sided overlap
    # > Front-right overlap
    ('gti', [[0.000, 0.080]], [[0.075, 0.080]]),
    # > Middle-right overlap
    ('gti', [[0.120, 0.175]], [[0.166, 0.175]]),
    # > Middle-left overlap
    ('gti', [[0.175, 0.240]], [[0.175, 0.237]]),
    # > Back-left overlap
    ('gti', [[0.300, 0.500]], [[0.300, 0.389]]), 
    # Merge an interval with a 2-sided overlap
    # > Middle-right-left
    ('gti', [[0.100, 0.240]], [[0.166, 0.237]]),
    # Merge an interal that spans one-or-more existing intervals
    # > Span front
    ('gti', [[0.050, 0.100]], [[0.075, 0.099]]),
    # > Span front+1
    ('gti', [[0.050, 0.175]], [[0.075, 0.099],
                               [0.166, 0.175]]),
    # > Span middle
    ('gti', [[0.150, 0.240]], [[0.166, 0.237]]),
    # > Span middle+1
    ('gti', [[0.150, 0.250]], [[0.166, 0.237],
                               [0.244, 0.250]]),
    # > Span back
    ('gti', [[0.240, 0.400]], [[0.244, 0.389]]),
    # > Span back+1
    ('gti', [[0.175, 0.400]], [[0.175, 0.237],
                               [0.244, 0.389]]),
    # Merge multiple intervals into a GTI
    # > Disjoint intervals
    ('gti', [[0.000, 0.050], 
             [0.400, 0.500]], np.zeros((0,2))),
    # > Disjoint and overlapping
    ('gti', [[0.000, 0.050], 
             [0.200, 0.400]], [[0.200, 0.237],
                               [0.244, 0.389]]),
    # > Overlapping intervals
    ('gti', [[0.000, 0.100], 
             [0.200, 0.400]], [[0.075, 0.099],
                               [0.200, 0.237],
                               [0.244, 0.389]]),
]

@pytest.mark.parametrize('gti,other,expected', testdata6, indirect=['gti'])
def test_merge_gti_and(gti, other, expected):
    out = gtitools.merge_gti_and(gti, other)
    assert np.allclose(np.ravel(out), np.ravel(expected))

