'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from .base import GeneratorBase
from sklearn.base import BaseEstimator, TransformerMixin

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from noxer.rnn import make_keras_picklable

make_keras_picklable()

class VaeGenerator(GeneratorBase):
    def __init__(self, latent_dim=2, intermediate_dim=256,
                 batch_size=100, epochs=50, D=1.0, epsilon_std=1.0,
                 lr=0.001, beta_1=0.9, beta_2=0.999):

        super(VaeGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.intermediate_dim = intermediate_dim
        self.D = D
        self.epsilon_std = epsilon_std

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def fit(self, X, Y, **kwargs):

        self.intermediate_dim = int(self.intermediate_dim)
        condition_dim = X.shape[-1]
        original_dim = Y.shape[-1]
        self.condition_dim = condition_dim

        Y_smpl = Input(shape=(original_dim,))
        X_cond = Input(shape=(condition_dim,))
        R_norm = Input(shape=(self.latent_dim,))

        YX_conc = Concatenate()([Y_smpl, X_cond])
        h = Dense(self.intermediate_dim, activation='relu')(YX_conc)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var, epsilon = args
            return z_mean + K.exp(z_log_var / 2) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        latent_g = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var, R_norm])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='linear')

        zx = Concatenate()([latent_g, X_cond])

        h_decoded = decoder_h(zx)
        y_decoded_mean = decoder_mean(h_decoded)

        vae_determinizm = self.D

        # Custom loss layer
        class VariationalLossLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLossLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean):
                xent_loss = metrics.mean_squared_error(x, x_decoded_mean)
                kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss * vae_determinizm + kl_loss)

            def call(self, inputs):
                x, x_decoded_mean, condition = inputs
                loss = self.vae_loss(x, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        y = VariationalLossLayer()([Y_smpl, y_decoded_mean, X_cond])
        vae = Model([R_norm, X_cond, Y_smpl], y)
        vae.compile(
            optimizer=Adam(
                lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2
            ),
            loss=None
        )

        for i in range(self.epochs):
            R = self._generate_noise(len(X))
            vae.fit([R, X, Y],
                    shuffle=True,
                    epochs=1,
                    batch_size=self.batch_size,
                    verbose=0)

        # build a model to project inputs on the latent space
        self.encoder = Model([Y_smpl, X_cond], z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_latent = Input(shape=(self.latent_dim,))
        decoder_condit = Input(shape=(self.condition_dim,))

        zx = Concatenate()([decoder_condit, decoder_latent])

        _h_decoded = decoder_h(zx)
        _x_decoded_mean = decoder_mean(_h_decoded)

        self.generator = Model([decoder_condit, decoder_latent], _x_decoded_mean)

    def _generate_noise(self, N):
        return np.random.randn(N, self.latent_dim)

    def predict(self, X, *args, **kwargs):
        lat = self._generate_noise(len(X))
        Yp = self.generator.predict([X, lat], verbose=0)
        return Yp
