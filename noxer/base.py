"""
A set of abstract classes defining interfaces.
"""

class AugmentMixin(object):
    """Mixin class for augmentation of data in noxer."""

    def fit_transform(self, X, Y, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and Y with optional parameters fit_params
        and returns a transformed version of X and Y.

        Parameters
        ----------
        X : array of shape [n_samples, ...]
            Training set.

        Y : array of shape [n_samples, ...]
            Target values.

        Returns
        -------
        X_new : array of shape [n_samples, ...]
            Transformed array.

        Y_new : array of shape [n_samples, ...]
            Transformed outputs.

        """
        return self.fit(X, Y, **fit_params).transform(X, Y)