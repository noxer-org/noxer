"""
Feature preprocessing of data, such as expanding categorical features to numerical ones.
"""

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column with index `key` from some matrix X"""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self  # do nothing during fitting procedure

    def transform(self, data_matrix):
        return data_matrix[:, [self.key]]  # return a matrix with single column


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Wrapper around LabelBinarizer. Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def fit(self, X, y=None):
        # create label encoder
        M = X[:, 0]
        self.encoder = LabelBinarizer()
        self.encoder.fit(M)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X[:,0])


class IntegerEncoder(BaseEstimator, TransformerMixin):
    """Wrapper around LabelBinarizer. Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def fit(self, X, y=None):
        # create label encoder
        M = X[:, 0]
        self.encoder = LabelEncoder()
        self.encoder.fit(M)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X[:,0])[:, np.newaxis]