import pytest
import numpy as np
import nicerlab.utils as utils

# find_first_of()
testdata1 = [
    (np.arange(10),       7,    7),
    (np.linspace(0,1,10), 0.33, 3),
    (np.linspace(0,1,10),-0.15, 0),
    (np.linspace(0,1,10),13,    None),
    ([],                  4,    None)
]

@pytest.mark.parametrize("data,value,expected", testdata1)
def test_find_first_of(data,value,expected):
    assert utils.find_first_of(data,value) == expected

def test_find_first_of_error():
    with pytest.raises(ValueError):
        utils.find_first_of([1,2,3,4], None)


# truncate()


# truncate_view()


