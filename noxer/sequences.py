"""
Contains a set of useful procedures and classes for sequence processing.
"""

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# A set of procedures for preprocessing of sequences


def make_subsequences(x, y, step=1, max_len=2 ** 31):
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

    step : int
        Step with which to subsample the sequence.

    max_len : int, default 2 ** 31
        Step with which to subsample the sequence.

    Returns
    -------
    a, b : a is all subsequences of x taken with some step, and b is labels assigned to these sequences.
    """

    r = range(step-1, len(x), step)

    X = []
    Y = []

    for i in r:
        start = max(0, i - max_len)
        stop = i+1
        X.append(x[start:stop])
        Y.append(y[i])

    return X, Y


class PadSubsequence(BaseEstimator, TransformerMixin):
    """
    Takes subsequences of fixed length from input list of sequences.
    If sequence is not long enough, it is left padded with zeros.

    Parameters
    ----------
    length : float, length of the subsequence to take

    """
    def __init__(self, length=10):
        self.length = length

    def _check_input(self, X):
        if len(X.shape) < 2:
            raise ValueError("The input sequence to the ")

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


class CalculateSpectrum(BaseEstimator, TransformerMixin):
    """Calculates spectrum of sequence.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Perform fft on sequence along every feature

        Parameters
        ----------
        X : array-like, shape [n_samples, seq_len, n_features]
            The data used to fft along the features axis.
        """
        from scipy import fftpack
        X = abs(fftpack.fft(X, axis=1))
        return X


class FlattenShape(BaseEstimator, TransformerMixin):
    """
    Flattens the shape of samples to a single vector. This is useful in cases
    when "classic" models like SVM are used.

    Parameters
    ----------

    """

    def fit(self, X, y=None):
        self.shape = X[0].shape
        return self

    def transform(self, X, y=None):
        V = np.array([np.ravel(x) for x in X])
        return V

    def inverse_transform(self, X, y=None):
        V = np.array([np.reshape(x, self.shape) for x in X])
        return V

# Wrapper for the standard classes of sklearn to work with sequence labeling

class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, mode='stack'):
        """
        Applies transformer to every element in input sequence.
        transformer: TransformerMixin
        mode: How to preprocess sequences for transformer fitting.
              default: stack all sequences into one huge sequence
              so that then it looks like a normal 2d training set
        """
        self.transformer = transformer
        self.mode = mode
        self.transformer_ = None

    def fit(self, X, y=None):
        """
        Fit base transformer to the set of sequences.

        X: iterable of shape [n_samples, ...]
        y: iterable of shape [n_samples, ...]

        """
        # stack all the elements into one huge dataset
        self.transformer_ = clone(self.transformer)

        if self.mode == 'stack':
            X_conc = np.row_stack(x for x in X)

            # might have bugs here in future :(
            if y is not None:
                y_conc = np.concatenate([[v] * len(x) for x, v in zip(X, y)])
        else:
            X_conc = X
            y_conc = y

        if y is None:
            self.transformer_.fit(X_conc)
        else:
            self.transformer_.fit(X_conc, y_conc)

        return self

    def transform(self, X, y=None):
        if y is None:
            result = [self.transformer_.transform(xx) for xx in X]
        else:
            result = [self.transformer_.transform(xx, [yy] * len(xx)) for xx, yy in zip(X, y)]

        result = np.array(result)
        return result

    def set_params(self, **params):
        self.base_transformer.set_params(**params)
        return self


class Subsequensor(BaseEstimator):
    """
    Creates views in all subsequences of a numpy sequence.

    Parameters
    ----------
    step: int, step with which the subsequences are taken.

    max_subsequence: int or None, maximum subsequence size that is used
        in order to predict a certain output value.

    """

    def __init__(self, step, max_subsequence=None):
        self.step = step
        self.max_subsequence = max_subsequence

    def fit(self, X, Y):
        """Fit the transformer according to the given training data.

        Parameters
        ----------
        X : list of numpy arrays
            List of sequences, where every sequence is a 2d numpy array,
            where the first dimension corresponds to time, and last for features.

        Y : list of object
            List of labels assigned to corresponding sequences in X.
        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X, Y=None):
        """Transform the input data.

        Parameters
        ----------
        X : list of numpy arrays
            List of sequences, where every sequence is a 2d numpy array,
            where the first dimension corresponds to time, and last for features.

        Y : list of object
            List of labels assigned to corresponding sequences in X.

        Returns
        -------
        X : list
            Returns list of views into the sequences.
        """
        test_time = Y is None
        if test_time:
            Y = [[None]*len(x) for x in X]

        if self.max_subsequence is None:
            args = (self.step, )
        else:
            args = (self.step, self.max_subsequence)

        XY = [make_subsequences(*((x, y, ) + args)) for x, y in zip(X, Y)]
        X = [z[0] for z in XY]
        if test_time:
            return X
        return X, [z[1] for z in XY]


class SequenceEstimator(BaseEstimator):
    """
    This generic estimator class can be used to label every element in a sequence using underlying subsequence estimator.
    One example would be labeling which parts of sensory data correspond to what kind of activity of the user.

    Consider the following example:

    X = [[1,2,3]]
    y = [[0,0,1]]

    fit() will train the estimator to classify properly the following data:

    X = [[1], [1,2], [1,2,3]]
    y = [[0, 0, 1]]

    predict() on X will return labels for every element in a sequence.

    Parameters
    ----------
    estimator: BaseEstimator, model which is used to do estimations on subsequences.

    step: int, step with which the subsequences are taken for training of internal sestimator.

    """
    def __init__(self, estimator, step=1, max_subsequence=None):
        self.estimator = estimator
        self.step = step
        self.max_subsequence = max_subsequence

        self.subsequencer = None # class instance that is responsible for getting views into the sequence

    def set_params(self, **params):
        step_name = self.__class__.__name__.lower() + "__step"
        if step_name in params:
            self.step = params[step_name]
            params = params.copy()
            del params[step_name]
        self.estimator.set_params(**params)
        return self

    def fit(self, X, y):
        X, y = Subsequensor(step=self.step, max_subsequence=self.max_subsequence).transform(X, y)
        X, y = sum(X, []), sum(y, []) # concat all data together
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        X = Subsequensor(step=1).transform(X)
        R = [self.estimator.predict(x) for x in X]
        return R

    def score(self, X, y):
        X, y = Subsequensor(step=self.step, max_subsequence=self.max_subsequence).transform(X, y)
        X, y = sum(X, []), sum(y, []) # concat all data together
        return self.estimator.score(X, y)

# Classes that work with sequences directly


# Readers

def read_wav(filename, mono=False):
    """
    Reads a wav file into a sequence of vectors, which represent
    the intensity of sound at some time.
    Every vector has a lenght of 1 if mono mode is used, else 2.

    Parameters
    ----------
    filename : string, file to read

    mono: bool, whether to read audio as mono or stereo. Mono files are always read as mono.

    Returns
    -------
    numpy array containing sequence of audio intensities.
    """
    import scipy.io.wavfile as scw
    framerate, data = scw.read(filename)

    if len(data.shape) < 2:
        data = data[:,np.newaxis]

    if mono:
        data = np.mean(data, axis=1)
        data = data[:,np.newaxis]

    return data


# Example pipelines


def rnn_pipe():
    pipe = make_pipeline(
        PadSubsequence(),
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
        PadSubsequence(),
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

