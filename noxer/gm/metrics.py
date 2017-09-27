"""
Evaluation metrics for quality of outputs of generative models.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score

import numpy as np


def distribution_similarity(X_true, X_pred, cross_testing=False): #
    """Compares the similarity of two distributions using a set
    of samples from "truth" distribution X_true and "predicted"
    distribution X_pred. Is useful for estimation of quality
    of GAN's and VAE's and the like.

    Parameters
    ----------
    X_true : array-like of shape = (n_samples, n_outputs)
        Samples from "ground truth" distribution.

    X_pred : array-like of shape = (n_samples, n_outputs)
        Samples from "ground truth" distribution.

    cross_testing : bool, optional
        Whether to use cross-validation like approach for testing.

    Returns
    -------
    z : float
        The similarity score for two distributions, calculated
        from the generalization estimate of the model that is
        trained to distinguish between two sets of samples.

    """

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyClassifier())
    ])

    dummy_search = {
        'model__strategy': ["stratified", "most_frequent", "uniform"]
    }

    lin_search = {
        'model': [LinearSVC()],
        'model__penalty': ['l1', 'l2'],
        'model__dual': [False],
        'model__C': 10 ** np.linspace(-10, 10),
        'model__max_iter': [10000],
    }

    gb_search = {
        'model': [GradientBoostingClassifier()],
        'model__learning_rate': [1.0, 0.1, 0.01, 0.001],
        'model__n_estimators': [2 ** i for i in range(11)],
    }

    model = GridSearchCV(
        pipe,
        [dummy_search, lin_search, gb_search],  # svc_search
        n_jobs=-1,
        verbose=0
    )


    X = np.row_stack([X_true, X_pred])
    X = X.reshape((len(X), -1))

    y = np.concatenate([
        np.ones(len(X_true)),
        np.zeros(len(X_pred))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, stratify=y)

    score = model.fit(X_train, y_train).score(X_test, y_test)

    # scale the error to be in range from 0.0 to 1.0
    U, C = np.unique(y_test, return_counts=True)
    scale = max(C * 1.0) / sum(C * 1.0)
    score = (1.0 - score)/scale
    score = min(1.0, score)

    return score

if __name__ == "__main__":
    # example usage
    X1 = np.random.randn(512,2)

    for offset in [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]:
        X2 = np.random.randn(512,2) + offset
        sim = distribution_similarity(X1, X2)
        print(sim)


