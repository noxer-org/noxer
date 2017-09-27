"""
Simplistic example usage of Variational Auto Encoder
for generation of artificial data.
"""

import numpy as np


X = np.zeros((1024, 1))
Y = np.random.randn(len(X), 3)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from noxer.gm.vae import VaeGenerator


model = VaeGenerator(Y_prep=StandardScaler())
model.fit(X, Y)
model.set_params(y_prep__with_mean=False)

print(model.score(X, Y))