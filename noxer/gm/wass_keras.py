"""
Implementation of Wasserstein discriminator
as in https://arxiv.org/pdf/1701.07875.pdf and
related publications.

This implementation is Keras based.
"""

import numpy as np

from keras.layers import Input, Flatten, Dense, LeakyReLU, Reshape, \
    Activation, Convolution2D, Deconvolution2D, MaxPool2D, UpSampling2D, \
    BatchNormalization, Concatenate

from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error, mean_absolute_error

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from keras.constraints import Constraint
from keras import backend as K


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be
    inside a range.
    '''
    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


def wass_train_loss(y_true, y_pred):
    ''' maximizes score for positive true labels
     and minimizes for negative true'''
    return K.sum(-y_true * y_pred)


def _check_wass_weights(model):
    # sanity check on weights of the discriminator
    for layer in model.layers:
        weights = layer.get_weights()  # list of numpy arrays
        for w in weights:
            m = np.max(np.abs(w))
            print(m)


class WassersteinDiscriminator(BaseEstimator, RegressorMixin):
    """This class learns the objective function that
    discriminates between two distributions of samples.
    For a particular sample, such objective estimates
    how fake the sample looks. The less the value of
    objective - the more realistic the sample looks."""
    def __init__(self):
        self.model = None

    def make_architecture(self, X, Y):
        """
        Make the discriminator network.

        X: n-d array of samples
            Data to train on.
        Y: n-d array of binary values
            Label for every sample: +1 (real sample), -1 (fake sample)
        """

        sh = X[0].shape
        ip = Input(shape=sh)
        h = ip
        h = Flatten()(h) if len(sh) > 1 else h

        for i in range(2):
            h = Dense(128, W_constraint=WeightClip(), bias_constraint=WeightClip())(h)
            h = LeakyReLU()(h)

        # final output - single score value
        h = Dense(1, W_constraint=WeightClip(), bias_constraint=WeightClip())(h)

        self.model = Model(inputs=ip, outputs=h)
        return self

    def compile_architecture(self, X, Y):
        self.model.compile(
            optimizer=RMSprop(),
            loss=wass_train_loss
        )
        return self

    def train(self, X, Y, monitor=None):
        """
        Run the training on compiled model

        X: n-d array of samples
            Data to train on.

        Y: n-d array of binary values
            Label for every sample: +1 (real sample), -1 (fake sample)
        """
        for i in range(10):
            self.model.fit(X, Y, epochs=1)

            if monitor is not None:
                monitor()

    def fit(self, X, Y, monitor=None):
        """
        Fit the Wasserstein GAN discriminator to the data

        X: n-d array of samples
            Data to train on.

        Y: n-d array of binary values
            Label for every sample: +1 (real sample), -1 (fake sample)
        """
        self.make_architecture(X, Y)
        self.compile_architecture(X, Y)
        self.train(X, Y, monitor)

    def predict(self, X):
        """
        Makes estimations with Wasserstein GAN discriminator

        X: n-d array of samples
            Data to train on.

        Returns
        -------

        Y: n-d array of scores
            Score which indicates how far the sample
            is from being in "real" category
        """

        return self.model.predict(X)
