import numpy as np
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
from sksurv.ensemble import RandomSurvivalForest


class CustomRandomForest(BaseEstimator, SurvivalAnalysisMixin):
    """
    Custom RandomSurvival forest that deals with NaN values by using a double-coding strategy (1).

    Parameters
    ----------
    max_features : int, float, string or None
        The number of features to consider when looking for the best split:
            - If int, then consider max_features features at each split.
            - If float, then max_features is a fraction and int(max_features * n_features) features are considered at
             each split.
            - If “sqrt”, then max_features=sqrt(n_features).
            - If “log2”, then max_features=log2(n_features).
            - If None, then max_features=n_features.

    max_depth : int
        The maximum depth of the tree.

    Attributes
    ----------
    rf_ : RandomSurvivalForest
        Fitted Random Survival Forest model.

    nan_features_ : array of shape (n_nan_features,)
        Boolean mask that indicates features containing NaN values.

    References
    ----------
    1. Engemann DA, Kozynets O, Sabbagh D, Lemaître G, Varoquaux G, Liem F, et al. Combining magnetoencephalography with
     magnetic resonance imaging enhances learning of surrogate-biomarkers. Elife. 2020 May 19;9:e54055.

    """

    def __init__(self, max_features, max_depth):
        self.max_features = max_features
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Fit the CustomRandomForest model to the provided data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : strucutred array of shape (n_samples,)
            Target representing time to event and censoring.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.rf_ = RandomSurvivalForest(
            max_depth=self.max_depth, max_features=self.max_features
        )
        self.nan_features_ = np.argwhere(np.isnan(X).sum(axis=0) > 0).reshape(-1)
        if len(self.nan_features_) > 0:
            X_notnan = np.delete(X, self.nan_features_, axis=1)
            X_nan = np.copy(X)[:, self.nan_features_]
            X_new = np.hstack(
                (
                    X_notnan,
                    np.where(np.isnan(X_nan), -1000, X_nan),
                    np.where(np.isnan(X_nan), 1000, X_nan),
                )
            )
        else:
            X_new = np.copy(X)
        self.rf_.fit(X_new, y)
        return self

    def predict(self, X):
        """
        Predict risk score.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to predict risk scores.

        Returns
        -------
        array of shape (n_samples,)
            Predicted risk scores.
        """

        if len(self.nan_features_) > 0:
            X_notnan = np.delete(X, self.nan_features_, axis=1)
            X_nan = np.copy(X)[:, self.nan_features_]
            X_new = np.hstack(
                (
                    X_notnan,
                    np.where(np.isnan(X_nan), -1000, X_nan),
                    np.where(np.isnan(X_nan), 1000, X_nan),
                )
            )
        else:
            X_new = np.copy(X)
        # Deal with unseen missing values during training
        unseen_nan_features = np.argwhere(np.isnan(X_new).sum(axis=0) > 0).reshape(-1)
        if len(unseen_nan_features) > 0:
            X_new = np.where(np.isnan(X_new), -1000, X_new)
        return self.rf_.predict(X_new)
