from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from abc import abstractmethod

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
                 batch_size=128, max_iter=128, max_patience=1e10, val_fraction=0.2):
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
        #X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=(1.0 - self.val_fraction))

        X_train = X
        y_train = y

        best_loss_ = 100000000000.0
        patience = self.max_patience
        max_iter = self.max_iter

        best_model_ = [np.copy(w) for w in self.model.get_weights()]

        while patience > 0 and max_iter > 0:
            max_iter -= 1

            val_loss = self.model.fit(X_train,y_train, epochs=1, batch_size=self.batch_size, verbose=0)
            val_loss = val_loss.history['loss'][-1]
            #val_loss = self.model.evaluate(X_val, y_val, verbose=0)

            if np.isnan(val_loss) or np.isinf(val_loss):
                break

            best_model_ = [np.copy(w) for w in self.model.get_weights()]
            max_iter -= 1

            """
            if val_loss < best_loss_:
                best_loss_ = val_loss
                patience = self.max_patience
            else:
                patience -= 1
            """

        self.model.set_weights(best_model_)

        return self

    def _predict(self, X):
        raise NotImplementedError("Abstract method not implemented!")

    def predict(self, X):
        return self.encoder.inverse_transform(self._predict(X))

def bad_activation(x):
    return x - 3.0

class KerasClassifierBase(KerasNNBase, ClassifierMixin):
    @abstractmethod
    def create_architecture(self, X, n_classes):
        """
        Generates the architecture of nn to be trained.
        """

    def _make_model(self, X, y, optimizer):
        import keras.models
        from keras.layers import Input, Dense, Conv1D, Flatten
        from keras.layers import Activation
        from keras.optimizers import Adam

        n_classes = len(np.unique(y))
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        y = self.encoder.transform(y)

        try:
            model = self.create_architecture(X, n_classes)
        except BaseException as ex:
            ip = Input(shape=X[0].shape)
            x = ip
            x = Flatten()(x)
            x = Dense(n_classes, activation='tanh')(x)
            x = Activation(bad_activation)(x)
            print('Infeasible!')
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


class RNNClassifier(KerasClassifierBase):
    def create_architecture(self, X, n_classes):
        import keras.models
        from keras.layers import Input, Dense, GRU
        from keras.layers.advanced_activations import LeakyReLU
        ip = Input(shape=X[0].shape)
        x = ip
        for i in range(self.n_layers):
            x = GRU(self.n_neurons)(x)
        x = Dense(n_classes, activation='softmax')(x)
        return keras.models.Model(input=ip, output=x)


class CNN1DClassifier(KerasClassifierBase):
    def __init__(self, conv_sz=3, stride=1, n_neurons=32, n_layers=1, lr=1e-4, beta1=0.9, beta2=0.99,
                 batch_size=128, max_iter=128, max_patience=32, val_fraction=0.2):
        super(CNN1DClassifier, self).__init__(
            n_neurons=n_neurons, n_layers=n_layers, lr=lr, beta1=beta1, beta2=beta2,
            batch_size=batch_size, max_iter=max_iter, max_patience=max_patience, val_fraction=val_fraction
        )
        self.conv_sz = conv_sz
        self.stride = stride

    def create_architecture(self, X, n_classes):
        import keras.models
        from keras.layers import Input, Dense, Conv1D, Flatten
        from keras.layers.advanced_activations import LeakyReLU
        ip = Input(shape=X[0].shape)
        x = ip
        for i in range(self.n_layers):
            x = Conv1D(filters=self.n_neurons, kernel_size=self.conv_sz,
                       strides=self.stride, padding='same')(x)
            x = LeakyReLU(0.05)(x)
        x = Flatten()(x)
        x = Dense(n_classes, activation='softmax')(x)
        return keras.models.Model(input=ip, output=x)


class DNNClassifier(KerasClassifierBase):
    def create_architecture(self, X, n_classes):
        import keras.models
        from keras.layers import Input, Dense, Flatten
        from keras.layers.advanced_activations import LeakyReLU
        ip = Input(shape=X[0].shape)
        x = ip
        x = Flatten()(x)
        for i in range(self.n_layers):
            x = Dense(self.n_neurons)(x)
            x = LeakyReLU(0.05)(x)
        x = Dense(n_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=ip, outputs=x)
        return model

class KerasRegressorBase(KerasNNBase, RegressorMixin):
    @abstractmethod
    def create_architecture(self, X):
        """
        Creates architecture of regressor.
        """

    def _make_model(self, X, y, optimizer):
        import keras.models
        from keras.layers import Input, Dense, GRU
        from keras.optimizers import Adam

        self.encoder = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

        try:
            model = self.create_architecture
        except BaseException as ex:
            ip = Input(shape=X[0].shape)
            x = ip
            x = Dense(1)(x)
            model = keras.models.Model(input=ip, output=x)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy'
        )

        self.model = model

    def _predict(self, X):
        return self.model.predict(X)

class RNNRegressor(KerasRegressorBase):
    def create_architecture(self, X):
        import keras.models
        from keras.layers import Input, Dense, GRU
        from keras.layers.advanced_activations import LeakyReLU
        ip = Input(shape=X[0].shape)
        x = ip
        for i in range(self.n_layers):
            x = GRU(self.n_neurons)(x)
            x = LeakyReLU(0.05)(x)
        x = Dense(1)(x)
        model = keras.models.Model(input=ip, output=x)
        return model

class CNN1DRegressor(KerasRegressorBase):
    def __init__(self, conv_sz, stride, *args, **kwargs):
        super(CNN1DRegressor, self).__init__(
            *args, **kwargs
        )
        self.conv_sz = conv_sz
        self.stride = stride

    def create_architecture(self, X):
        import keras.models
        from keras.layers import Input, Dense, Conv1D, Flatten
        from keras.layers.advanced_activations import LeakyReLU
        ip = Input(shape=X[0].shape)
        x = ip
        for i in range(self.n_layers):
            x = Conv1D(self.n_neurons, self.conv_sz, self.stride, padding='same')(x)
            x = LeakyReLU(0.05)(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        model = keras.models.Model(input=ip, output=x)
        return model

class DNNRegressor(KerasRegressorBase):
    def create_architecture(self, X):
        import keras.models
        from keras.layers import Input, Dense
        from keras.layers.advanced_activations import LeakyReLU
        ip = Input(shape=X[0].shape)
        x = ip
        for i in range(self.n_layers):
            x = Dense(self.n_neurons)(x)
            x = LeakyReLU(0.05)(x)
        x = Dense(1)(x)
        model = keras.models.Model(input=ip, output=x)
        return model