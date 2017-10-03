"""
Example of 2d distribution of 4 gaussians.
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from noxer.gm.sgm import SGM, ScalarGenerator

# training data
N = 2048
X = np.zeros((N, 2))
Y = np.random.randn(N, 2) + np.sign(np.random.randn(N, 2))*2.0

# training testing split
Xt, Xe, Yt, Ye = train_test_split(X, Y, train_size=0.75)

# define a model
# this model generates outputs sequentially entry by entry
model = SGM(
    ScalarGenerator(
        GradientBoostingRegressor(
            n_estimators=32,
            learning_rate=0.1
        )
    )
)

# fit the generative model
model.fit(Xt, Yt)

# example generation with the fitted model
Yp = model.predict(X)

# visualize generated outputs
plt.subplot(2,1,1)
plt.hist2d(Yp[:, 0], Yp[:, 1])
plt.title("Generated data")
plt.subplot(2,1,2)
plt.title("Actual data")
plt.hist2d(Y[:, 0], Y[:, 1])

plt.show()

print('Performance estimate: 0.0 - very bad, 1.0 - very good')
print(model.score(Xe, Ye))

