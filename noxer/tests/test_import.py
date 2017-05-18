import numbers
import numpy as np

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_raises_regex


def test_import():
    import noxer as nx
