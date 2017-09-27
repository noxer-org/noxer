"""
Simplistic example usage of Variational Auto Encoder
for generation of artificial data.
"""

import numpy as np


X = np.zeros((1024, 1))
Y = np.random.randn(len(X), 3)

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from noxer.gm.base import IOTransform
from noxer.gm.metrics import distribution_similarity
from noxer.gm.vae import VaeGenerator
from skopt import BayesSearchCV

# it is possible to specify pipelines to work inside the generator model
pipe = IOTransform(
    #X_prep=make_pipeline(StandardScaler()), # preprocessing of inputs
    #Y_prep=make_pipeline(StandardScaler()), # preprocessing of outputs
    model=VaeGenerator(),
    metric=distribution_similarity,
)

cv_folds = [train_test_split(range(len(X)), train_size=0.666)]

model = BayesSearchCV(
    estimator=pipe,
    search_spaces={
        'model__latent_dim': (2, 20),
        'model__intermediate_dim': (8, 128),
        'model__epochs': (8, 128),
        'model__D': (1e-3, 1e+3, 'log-uniform'),
        'model__lr': (1e-4, 1e-2, 'log-uniform'),
    },
    n_iter=32,
    cv=cv_folds,
    refit=False,
    error_score=-1.0
)

model.on_step = lambda x: print((x, model.total_iterations(), model.best_score_))
model.fit(X, Y)
model.refit = True
model._fit_best_model(X, Y)
print(model.best_params_)
print(model.score(X, Y))
"""

model = pipe
model.set_params(**{'model__D': 5.1964624423233898, 'model__lr': 0.00010138257365940301,
                    'model__epochs': 26, 'model__intermediate_dim': 125, 'model__latent_dim': 2})
model.fit(X, Y)

print(model.predict(X, Y))
"""