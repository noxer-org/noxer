"""
A set of helper classes for better pipelining of data preprocessing
for machine learning and beyond.
"""

from sklearn.base import BaseEstimator


class IOTransform(BaseEstimator):
    """
    A base class for training.
    Implements a set of useful methods and variables, such
    that preprocessing of the data can be done using scikit-learn
    like class instances.

    Parameters
    ----------
    X_prep : BaseEstimator
        Class instance that will be fitted to the input X
        for the model. This transformer is applied to the
        input X before it is fed into the model.

    Y_prep : BaseEstimator
        Class instance that will be fitted to the output values Y
        for the model. This transformer is applied to the values of
        Y when it is used for training.

    Y_post : BaseEstimator
        Class instance that will be fitted to the output values Y
        for the model. This transformer is applied after the values
        are generated.

    model : BaseEstimator
        Instance of a class that is used for mapping from inputs to
        outputs.

    metric : callable with two arguments
        Scorer that is used to evaluate predictions of the model. If
        None, the score function of the model will be used.

    """

    _estimator_type = "generator"

    def __init__(self, model, metric=None, augm=None, X_prep=None, Y_prep=None, Y_post=None):
        self.X_prep = X_prep
        self.Y_prep = Y_prep
        self.Y_post = Y_post

        if not isinstance(model, BaseEstimator):
            raise TypeError('Model should be an instance of BaseEstimator, got %s' % model)

        self.model = model
        self.metric = metric
        self.augm = augm

    def set_params(self, **params):
        """
        Custom setting of parameters for generative models.
        All parameters that start with 'x_prep', 'y_prep', 'y_post' are
        delegated to respective preprocessors.
        """

        elements = {'augm', 'X_prep', 'Y_prep', 'Y_post', 'model'}

        params = {
            k:v for k, v in params.items()
            if not any(
                k.startswith(p.lower()) for p in elements
            )
        }

        BaseEstimator.set_params(self, **params)

        # set attributes of elements
        for e in elements:
            element = getattr(self, e)

            if isinstance(element, BaseEstimator):
                element.set_params(
                    **{
                        k[len(e)+2:]: v for k, v in params.items()
                        if k.startswith(e.lower())
                    }
                )

        return self

    def _fit_preprocessors(self, X, Y):
        """Fits all preprocessors to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.

        Y : {array-like, sparse matrix}, shape [n_samples, ...]
            The target values estimated by the model.
        """

        if self.augm is not None:
            X, Y = self.augm.fit_transform(X, Y)

        if self.X_prep is not None:
            X = self.X_prep.fit_transform(X, Y)

        if self.Y_post is not None:
            self.Y_post.fit(Y, X)

        if self.Y_prep is not None:
            Y = self.Y_prep.fit_transform(Y, X)

        return X, Y

    def _transform_inputs(self, X, Y=None):
        """Transforms inputs so that they can be used for estimations
        with generative model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.

        Y : {array-like, sparse matrix}, shape [n_samples, ...]
            The target values estimated by the model.
        """
        if self.X_prep is not None:
            # account for some transformers taking only single argument
            if 'Y' in self.X_prep.transform.__code__.co_varnames:
                X = self.X_prep.transform(X, Y)
            else:
                X = self.X_prep.transform(X)

        if Y is None:
            return X

        if self.Y_prep is not None:
            # account for some transformers taking only single argument
            if 'Y' in self.Y_prep.transform.__code__.co_varnames:
                Y = self.Y_prep.transform(Y, X)
            else:
                Y = self.Y_prep.transform(Y)

        return X, Y

    def _transform_generated_outputs(self, Y, X=None):
        """Apply output transformers to the generated values

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.

        Y : {array-like, sparse matrix}, shape [n_samples, ...]
            The target values estimated by the model.
        """
        if self.Y_prep is not None:
            if 'Y' in self.Y_prep.inverse_transform.__code__.co_varnames:
                Y = self.Y_prep.inverse_transform(Y, X)
            else:
                Y = self.Y_prep.inverse_transform(Y)

        if self.Y_post is not None:
            if 'Y' in self.Y_post.transform.__code__.co_varnames:
                Y = self.Y_post.transform(Y, X)
            else:
                Y = self.Y_post.transform(Y)

        return Y

    def fit(self, X, Y, *args, **kwargs):
        """
        Complete fitting pipeline with data preprocessing for generative
        models.

        Includes data augmentation.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.

        Y : {array-like, sparse matrix}, shape [n_samples, ...]
            The target values estimated by the model.
        """
        X, Y = self._fit_preprocessors(X, Y)
        self.model.fit(X, Y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        """
        Full generation pipeline with all necessary steps such as data
        preprocessing.

        IMPORTANT: this function does not do augmentation of input
        values! Hence a particular form of X should be the one
        that self.augm returns.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.
        """
        X = self._transform_inputs(X)
        Y = self.model.predict(X, *args, **kwargs)
        Y = self._transform_generated_outputs(Y, X)
        return Y

    def score_no_augmentation(self, X, Y, *args, **kwargs):
        """
        Evaluates the quality of the model using comparison
        to real data.

        DOES NOT include the data augmentation.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.

        Y : {array-like, sparse matrix}, shape [n_samples, ...]
            The target values estimated by the model.

        Returns
        -------
        score : float
            Score from 0.0 to 1.0 that indicates quality of estimations.
        """

        if self.metric:
            Yp = self.predict(X, *args, **kwargs)
            score = self.metric(Y, Yp)
        else:
            score = self.model.score(X, Y)

        return score

    def score(self, X, Y, *args, **kwargs):
        """
        Evaluates the quality of the model using comparison
        to real data.

        Includes data augmentation.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, ...]
            The data used as inputs to generatie model's outputs.

        Y : {array-like, sparse matrix}, shape [n_samples, ...]
            The target values estimated by the model.

        Returns
        -------
        score : float
            Score from 0.0 to 1.0 that indicates quality of estimations.
        """

        if self.augm is not None:
            X, Y = self.augm.transform(X, Y)

        return self.score_no_augmentation(X, Y, *args, **kwargs)

