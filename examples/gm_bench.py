"""
Tests how well the model performs with simplistic artificial distributions.
Thresholds are set for different kinds of situations that check whether a
model became worse or not.
"""
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from noxer.gm.sgm import SGM

def test_gauss_1d(visualize=False):
    """
    Test a generative model with single dimensional
    data of gaussian distributed samples.

    Parameters
    ----------
    """
    N = 2000
    X = np.zeros((N, 1))
    Y = np.random.randn(N, 1)

    Xt, Xe, Yt, Ye = train_test_split(X, Y, train_size=0.75)

    model = SGM(GradientBoostingRegressor(n_estimators=32, learning_rate=0.1))
    model.fit(Xt, Yt)

    if visualize:
        Yp = model.predict(X)
        import matplotlib.pyplot as plt
        plt.hist(Yp[:, 0])
        plt.grid()
        plt.show()

    print(model.score(Xe, Ye))


def test_noisy_onehot(visualize=False):
    """
    Test a generative model with a generation of data
    vector where only one value is 1.0 and rest is 0.0.

    Parameters
    ----------
    """
    N = 2000
    X = np.ones((N, 1))
    Y = np.random.randn(N, 3)*0.1

    for i in range(len(Y)):
        v = np.zeros(Y.shape[-1])
        j = np.random.randint(Y.shape[-1])
        v[j] = 1.0
        Y[i] += v

    Xt, Xe, Yt, Ye = train_test_split(X, Y, train_size=0.75)

    cv_folds = [train_test_split(range(len(Xt)), train_size=0.75)]

    model = GridSearchCV(
        estimator=SGM(GradientBoostingRegressor()),
        param_grid={
            'n_estimators': [2 ** i for i in range(1,11)],
            'learning_rate': [1.0, 0.1, 0.01]
        },
        n_jobs=-1,
        cv=cv_folds,
        verbose=1000
    )
    model.fit(Xt, Yt)

    if visualize:
        Yp = model.predict(X)
        for y in Yp:
            print(y)

    print(model.best_params_)
    print(model.best_score_)
    print(model.score(Xe, Ye))


if __name__ == "__main__":
    test_onehot(True)

