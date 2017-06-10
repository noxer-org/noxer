from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import tempfile
import numpy as np

# Keras preprocessing - making it picklable
# The function is run only when keras is necessary
def make_keras_picklable():
    import keras.models
    cls = keras.models.Model

    if hasattr(cls, "is_now_picklable"):
        return

    cls.is_now_picklable = True

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

    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


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
        self.model_ = None # future keras model

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

        import keras.models
        from keras.layers import Input, Dense, GRU
        from keras.optimizers import Adam

        make_keras_picklable()

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
