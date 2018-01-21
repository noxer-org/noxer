"""
This script contains a number of interfaces and
implementations for Supervised learning with PyTorch.
"""

import math
from abc import abstractmethod

import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


class PTLBase(BaseEstimator):
    """
    A base class for learning algorithms with pytorch.

    Parameters
    ----------

    epochs: int > 0
        Number of epochs to train neural network for.

    batch_size: int > 0
        Size of subsample of dataset to use to approximate the gradient in
        stochatic gradient descent procedure.

    alpha: float > 0
        Learning rate. Tunes the amount of update done after processing of
        single batch size.

    beta1: float 0.0 < x < 1.0
        Beta 1 parameter of Adam stochastic gradient descent algorithm.

    beta2: float 0.0 < x < 1.0
        Beta 2 parameter of Adam stochastic gradient descent algorithm.
    """
    def __init__(self, epochs=10, batch_size=256, alpha=0.001, beta1=0.9, beta2=0.999):

        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

        self.net = None

    @abstractmethod
    def make_architecture(self, X, y):
        """
        Should return nn.Module instance, which represents architecture
        of the neural network.

        Parameters
        ----------

        X: iterable of size n_samples
            Representation of dataset.

        y: iterable of size n_samples
            Representation of output

        Return
        ------
        net: an instance of a neural network to be trained.
        """
        pass

    def fit(self, X, y, criterion):
        """
        Trains a neural network on provided data.

        Parameters
        ----------

        X: iterable of size n_samples
            Representation of dataset.

        y: iterable of size n_samples
            Representation of output

        criterion: callable with 2 arguments, possibly a nn._Loss instance.
            Cost function to minimize.

        Return
        ------
        self
        """
        check_X_y(X, y, allow_nd=True, dtype=None)
        self.net = self.make_architecture(X, y)
        optimizer = optim.Adam(self.net.parameters(), lr=self.alpha, betas=(self.beta1, self.beta2))

        data = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y)
        )

        # this creates mixed batches
        trainloader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """
        Make estimation with trained neural network.

        Parameters
        ----------

        X: iterable of size n_samples
            Representation of inputs. Should be consistent with
            inputs in the training dataset.

        Return
        ------
        X: iterable of size n_samples
            Representation of estimated outputs.
        """
        if self.net is None:
            raise RuntimeError("The model is not fit. Did you forget to call the fit method on a dataset?")

        X = Variable(torch.FloatTensor(X), volatile=True)
        yp = self.net(X).data.numpy()
        return yp


class PTLClassifierBase(PTLBase, ClassifierMixin):
    """
    A base class for learning classifiers with pytorch.

    Parameters
    ----------
    See parent classes for corresponding parameters.
    """
    def __init__(self, epochs=10, batch_size=256, alpha=0.001,
                 beta1=0.9, beta2=0.999):
        super(PTLClassifierBase, self).__init__(
            epochs, batch_size, alpha, beta1, beta2
        )
        self.label_encoder = None

    def fit(self, X, y):
        """
        Trains a classifier on provided data.

        Parameters
        ----------

        X: iterable of size n_samples
            Representation of dataset.

        y: iterable of size n_samples
            Representation of classes

        Return
        ------
        self
        """
        # encode outputs
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        criterion = nn.CrossEntropyLoss()
        super(PTLClassifierBase, self).fit(X, y, criterion)
        return self

    def predict(self, X):
        """
        Estimate output classes.

        Parameters
        ----------
        X: iterable of size n_samples
            Representation of inputs to classify.

        Return
        ------
        y: iterable of size n_samples
            Representation of classes
        """
        yp = super(PTLClassifierBase, self).predict(X)
        yp = np.argmax(yp, axis=1)
        yp = self.label_encoder.inverse_transform(yp)
        return yp


class FFNNClassificationNN(nn.Module):
    """
    Simple fully connected feed forward NN.

    Parameters
    ----------
    xsz: int > 0
        Size of input vector

    ysz: int > 0
        Size of output vector

    n_layers: int > 0
        Number of layers in the neural network

    n_neurons: int > 0
        Number of neurons in every layer
    """
    def __init__(self, xsz, ysz, n_neurons, n_layers, dropout=None):
        super(FFNNClassificationNN, self).__init__()
        hsz = int(xsz)
        ysz = int(ysz)
        n_neurons = int(n_neurons)
        n_layers = int(n_layers)
        if dropout is not None:
            dropout = float(dropout)

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hsz, n_neurons))
            layers.append(nn.LeakyReLU())
            if dropout is not None:
                if dropout > 0.03:
                    layers.append(nn.Dropout(p=dropout))
            hsz = n_neurons

        layers.append(nn.Linear(hsz, ysz))
        layers.append(nn.Softmax(dim=-1))

        self.fc = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.fc:
            x = l(x)

        return x


class MLPClassifier(PTLClassifierBase):
    """
    Estimator with Feed Forward Neural Network.

    Parameters
    ----------
    For any parameters not listed, see PTLClassifierBase.

    n_layers: int > 0
        Number of layers in the NN

    n_neurons: int > 0
        Number of neurons in every layer
    """
    def __init__(self, dropout=None, n_layers=1, n_neurons=32, epochs=10, batch_size=256, alpha=0.001,
                 beta1=0.9, beta2=0.999):
        super(MLPClassifier, self).__init__(
            epochs, batch_size, alpha, beta1, beta2
        )
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.dropout = dropout

    def make_architecture(self, X, y):
        """
        See PTLBase.make_architecture for explanations.
        """
        net = FFNNClassificationNN(
            X.shape[-1], len(set(y)), self.n_neurons, self.n_layers,
            dropout=self.dropout
        )
        return net


class CNN1DClassificationNN(nn.Module):
    """
    Simple fully connected feed forward NN.

    Parameters
    ----------
    xsz: int > 0
        Size of input vector

    ysz: int > 0
        Size of output vector

    n_layers: int > 0
        Number of layers in the neural network

    n_neurons: int > 0
        Number of neurons in every layer
    """
    def __init__(self, xsz, ysz, n_neurons=64, n_layers=1, kernel_size=3, dropout=None):
        super(CNN1DClassificationNN, self).__init__()
        ssz = int(xsz[0])
        hsz = int(xsz[1])
        ysz = int(ysz)
        n_neurons = int(n_neurons)
        n_layers = int(n_layers)
        kernel_size = int(kernel_size)
        if dropout is not None:
            dropout = float(dropout)

        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv1d(hsz, n_neurons, kernel_size=kernel_size, padding=1))
            layers.append(nn.LeakyReLU())
            if dropout is not None:
                if dropout > 0.03:
                    layers.append(nn.Dropout(p=dropout))
            hsz = n_neurons
            # here avoid empty sequence
            essz = ssz / 2.0
            if essz < 1.0:
                break
            layers.append(nn.MaxPool1d(2, ceil_mode=True))
            ssz = math.ceil(essz)

        self.seq = nn.ModuleList(layers)

        # calculate flatten
        hsz = hsz * ssz

        self.ffnn = FFNNClassificationNN(hsz, ysz, n_neurons=n_neurons, n_layers=1, dropout=dropout)

    def forward(self, x):
        # reshape input to (batch size, channels, seq length)
        x = x.transpose(1, 2)
        for l in self.seq:
            x = l(x)
        # flatten the data
        x = x.view(x.size(0), -1)
        x = self.ffnn(x)
        return x


class CNN1DClassifier(PTLClassifierBase):
    """
    Estimator with one dimensional convolutional neural network.

    Parameters
    ----------
    For any parameters not listed, see PTLClassifierBase.

    n_layers: int > 0
        Number of layers in the NN

    n_neurons: int > 0
        Number of neurons in every layer
    """
    def __init__(self, kernel_size=3, dropout=None, n_layers=1, n_neurons=32, epochs=10, batch_size=256, alpha=0.001,
                 beta1=0.9, beta2=0.999):
        super(CNN1DClassifier, self).__init__(
            epochs, batch_size, alpha, beta1, beta2
        )
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

    def make_architecture(self, X, y):
        """
        See PTLBase.make_architecture for explanations.
        """
        net = CNN1DClassificationNN(
            X.shape[1:], len(set(y)),
            self.n_neurons, self.n_layers, self.kernel_size,
            dropout=self.dropout
        )
        return net


class GRUClassification(nn.Module):
    """
    Recurent neural network module. Maps sequence to vector output.

    Parameters
    ----------
    xsz: int > 0
        Size of input vector

    ysz: int > 0
        Size of output vector

    n_layers: int > 0
        Number of layers in the neural network

    n_neurons: int > 0
        Number of neurons in every layer
    """
    def __init__(self, xsz, ysz, n_neurons=64, n_layers=1, dropout=None):
        super(GRUClassification, self).__init__()
        ssz = int(xsz[0])
        hsz = int(xsz[1])
        ysz = int(ysz)
        n_neurons = int(n_neurons)
        n_layers = int(n_layers)
        if dropout is not None:
            dropout = float(dropout)
        else:
            dropout = 0.0

        self.rnn = nn.GRU(hsz, n_neurons, n_layers, dropout=dropout)
        # calculate flatten
        hsz = n_neurons

        self.ffnn = FFNNClassificationNN(hsz, ysz, n_neurons=n_neurons, n_layers=1)

    def forward(self, x):
        # swap to (seq_len, batch, input_size)
        x = x.transpose(0, 1)
        _, x = self.rnn(x)
        # flatten the data
        x = x[0, :, :]
        x = self.ffnn(x)
        return x


class GRUClassifier(PTLClassifierBase):
    """
    Estimator with one dimensional convolutional neural network.

    Parameters
    ----------
    For any parameters not listed, see PTLClassifierBase.

    n_layers: int > 0
        Number of layers in the NN

    n_neurons: int > 0
        Number of neurons in every layer
    """
    def __init__(self, n_layers=1, n_neurons=32, dropout=None, epochs=10, batch_size=256, alpha=0.001,
                 beta1=0.9, beta2=0.999):
        super(GRUClassifier, self).__init__(
            epochs, batch_size, alpha, beta1, beta2
        )
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.dropout = dropout

    def make_architecture(self, X, y):
        """
        See PTLBase.make_architecture for explanations.
        """
        net = GRUClassification(
            X.shape[1:], len(set(y)), self.n_neurons, self.n_layers,
            dropout=self.dropout
        )
        return net


def test_dnn_v_dnn(datafnc):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from searchgrid import set_grid, build_param_grid

    X, y = datafnc()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    estimator = make_pipeline(
            StandardScaler(),
            set_grid(
                PMLPClassifier(),
                epochs=[2 ** i for i in range(1, 8)],
                n_layers=list(range(1, 4)),
                n_neurons=[2 ** i for i in range(1, 8)],
                alpha=[1e-4, 1e-3, 1e-2]
            )
        )

    model = GridSearchCV(
        estimator=estimator,
        param_grid=build_param_grid(estimator),
        verbose=1000,
        cv=3,
        n_jobs=2
    )

    mlp = GridSearchCV(
        estimator=make_pipeline(
            StandardScaler(),
            MLPClassifier(),
        ),
        param_grid={
            'mlpclassifier__max_iter': [2 ** i for i in range(1, 8)]
        },
        verbose=1000,
        cv=3
    )

    model.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    print(datafnc.__name__)
    print(model.score(X_test, y_test))
    print(mlp.score(X_test, y_test))



def test_rnn(datafnc):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from searchgrid import set_grid, build_param_grid
    from noxer.sequences import PadSubsequence

    X, y = datafnc()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    estimator = make_pipeline(
            set_grid(
                GRUClassifier(n_neurons=64, n_layers=1, epochs=100),
                alpha=[1e-4, 1e-3, 1e-2]
            )
        )


    estimator = make_pipeline(
            set_grid(
                CNN1DClassifier(epochs=64),
                alpha=[0.01],
                n_layers=[1],
                n_neurons=[32],
                dropout=[0.2, 0.3, 0.4]
            )
        )

    model = GridSearchCV(
        estimator=estimator,
        param_grid=build_param_grid(estimator),
        verbose=1000,
        cv=3,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    print(datafnc.__name__)
    print(model.score(X_test, y_test))


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_digits

    def rnd_data():
        X = np.random.randn(2048, 30 * 20)
        y = X[:, 0] > 0.0
        return X, y

    def rnn_data():
        X = np.random.randn(2500, 30, 60)
        y = X[:, 0, 0] > 0.0
        return X, y

    #test_dnn_v_dnn(rnd_data)
    #test_dnn_v_dnn(lambda : load_digits(return_X_y=True))
    test_rnn(rnn_data)



