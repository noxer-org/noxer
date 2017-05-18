"""
Implements the keras RNN classifier as a sklearn model
"""

import keras.models
from keras.layers import Input, Dense, GRU
from keras.optimizers import Adam

import tempfile
import types
import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

make_keras_picklable()

class RNNClassifier(BaseEstimator):
    """
    Recurrent neural networks using keras as a backend.

    Parameters
    ----------
    n_neurons : int
        Width of the neural network.

    """
    def __init__(self, n_neurons=128):
        self.n_neurons=n_neurons

        self.model_ = None # keras model

    def fit(self, X, y):
        """
        Fit RNN model.

        Parameters
        ----------
        X : array of array of sequences [n_samples, seq_length, n_features]

        y : numpy array of shape [n_samples]
            Target classes. Can be string, int etc.

        Returns
        -------
        self : returns an instance of self.
        """

        n_classes = len(np.unique(y))
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        y = self.encoder.transform(y)

        ip = Input(shape=X[0].shape)
        x = ip
        x = GRU(self.n_neurons, activation="relu")(x)
        x = Dense(n_classes)(x)

        self.model = keras.models.Model(input=ip, output=x)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=0.0001))

        self.model.fit(X,y, nb_epoch=32, batch_size=128)
        return self

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict_int(self, X):
        yp = self.predict_proba(X)
        return np.argmax(yp, axis=1)

    def predict(self, X):
        return self.encoder.inverse_transform(self.predict_int(X))

    def score(self, X, y):
        y_enc = self.encoder.transform(y)
        y_prd = self.predict_int(X)
        return accuracy_score(y_enc, y_prd)


def make_subsequences(x, y, step=1):
    """
    Creates views to all subsequences of the sequence x. For example if
    x = [1,2,3,4]
    y = [1,1,0,0]
    step = 1
    the result is a tuple a, b, where:
    a = [[1],
    [1,2],
    [1,2,3],
    [1,2,3,4]
    ]
    b = [1,1,0,0]

    Note that only a view into x is created, but not a copy of elements of x.

    Parameters
    ----------
    X : array [seq_length, n_features]

    y : numpy array of shape [n_samples]
        Target values. Can be string, float, int etc.

    Returns
    -------
    a, b : a is all subsequences of x taken with some step, and b is labels assigned to these sequences.
    """

    r = range(step-1, len(x), step)

    X = []
    Y = []

    for i in r:
        X.append(x[:i+1])
        Y.append(y[i])

    return X, Y


class PaddedSubsequence(BaseEstimator, TransformerMixin):
    """
    Takes subsequences of fixed length from input list of sequences.
    If sequence is not long enough, it is left padded with zeros.

    Parameters
    ----------
    length : float, length of the subsequence to take

    """
    def __init__(self, length=10):
        self.length = length

    def fit(self,X,y=None):
        # remeber the num. of features
        self.n_features = X[0].shape[-1]
        return self

    def transform(self, X, y=None):
        # X might be a list
        R = []
        for x in X:
            if len(x) >= self.length:
                R.append(x[-self.length:])
            else:
                z = np.zeros((self.length - len(x), x.shape[-1]))
                zx = np.row_stack((z,x))
                R.append(zx)
        R = np.array(R)
        return R


class FlattenShape(BaseEstimator, TransformerMixin):
    """
    Flattens the shape of samples to a single vector. this is useful in cases
    when "classic" models like SVM are used.

    Parameters
    ----------

    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        V = np.array([np.ravel(x) for x in X])
        return V

### Example pipelines

def rnn_pipe():
    pipe = make_pipeline(
        PaddedSubsequence(),
        RNNClassifier()
    )
    grid = [
        {
            "paddedsubsequence__length":[2,4],
            "rnnclassifier__n_neurons":[32]
        }
    ]
    return pipe, grid


def svm_pipe():
    pipe = make_pipeline(
        PaddedSubsequence(),
        FlattenShape(),
        StandardScaler(),
        LinearSVC(),
    )
    grid = [
        {
            "paddedsubsequence__length":[1,2,4,8,16],
            "linearsvc__C":10 ** np.linspace(-10, 10, 51)
        }
    ]
    return pipe, grid


if __name__ == "__main__":
    pass

