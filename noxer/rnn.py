from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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


class KerasNNBase(BaseEstimator):
    """
    Recurrent neural networks using keras as a backend.

    Parameters
    ----------
    n_neurons : [int, default=32]
        Width of the neural network.

    lr : [float, default=32]
        Learning rate used in the optimizer for the network.

    beta1 : [float, default=0.9]
        beta_1 parameter of the Adam optimization algorithm.

    beta2 : [float, default=0.99]
        beta_2 parameter of the Adam optimization algorithm.

    """
    def __init__(self, n_neurons=32, n_layers=1, lr=1e-4, beta1=0.9, beta2=0.99,
                 batch_size=128, max_iter=128, max_patience=32, val_fraction=0.2):
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size

        self.max_iter = max_iter
        self.val_fraction = val_fraction
        self.max_patience = max_patience

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

        from keras.optimizers import Adam
        from copy import deepcopy

        make_keras_picklable()

        optimizer = Adam(
            lr=self.lr,
            beta_1=self.beta1,
            beta_2=self.beta2
        )

        self._make_model(X, y, optimizer)

        y = self.encoder.transform(y)

        # split data into training and validation parts
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=(1.0 - self.val_fraction))

        best_loss_ = 100000000000.0
        patience = self.max_patience
        max_iter = self.max_iter


        while patience > 0 and max_iter > 0:
            max_iter -= 1

            if max_iter % 10 == 0:
                print(max_iter)

            self.model.fit(X_train,y_train, nb_epoch=1, batch_size=self.batch_size, verbose=0)
            val_loss = self.model.evaluate(X_val, y_val, verbose=0)

            if val_loss < best_loss_:
                best_loss_ = val_loss
                best_model_ = [np.copy(w) for w in self.model.get_weights()]
                patience = self.max_patience
            else:
                patience -= 1

        self.model.set_weights(best_model_)

        return self

    def _predict(self, X):
        raise NotImplementedError("Abstract method not implemented!")

    def predict(self, X):
        return self.encoder.inverse_transform(self._predict(X))


class RNNClassifier(KerasNNBase, ClassifierMixin):

    def _make_model(self, X, y, optimizer):
        import keras.models
        from keras.layers import Input, Dense, GRU
        from keras.optimizers import Adam

        n_classes = len(np.unique(y))
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        y = self.encoder.transform(y)

        ip = Input(shape=X[0].shape)
        x = ip
        x = GRU(self.n_neurons, activation="relu")(x)
        x = Dense(n_classes)(x)

        model = keras.models.Model(input=ip, output=x)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy'
        )

        self.model = model

    def predict_proba(self, X):
        make_keras_picklable()
        return self.model.predict(X)

    def _predict(self, X):
        yp = self.predict_proba(X)
        return np.argmax(yp, axis=1)


class RNNRegressor(KerasNNBase, RegressorMixin):

    def _make_model(self, X, y, optimizer):
        import keras.models
        from keras.layers import Input, Dense, GRU
        from keras.optimizers import Adam

        self.encoder = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
        ip = Input(shape=X[0].shape)
        x = ip
        x = GRU(self.n_neurons, activation="relu")(x)
        x = Dense(1)(x)

        model = keras.models.Model(input=ip, output=x)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy'
        )



        self.model = model

    def _predict(self, X):
        return self.model.predict(X)