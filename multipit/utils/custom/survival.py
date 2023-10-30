import numpy as np
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
from sksurv.ensemble import RandomSurvivalForest


class CustomRandomForest(BaseEstimator, SurvivalAnalysisMixin):
    """
    Parameters
    ----------
    max_features:

    max_depth:

    Attributes
    ----------
    rf_:

    nan_features_:
    """
    def __init__(self, max_features, max_depth):
        self.max_features = max_features
        self.max_depth = max_depth

    def fit(self, X, y):
        self.rf_ = RandomSurvivalForest(max_depth=self.max_depth, max_features=self.max_features)
        self.nan_features_ = np.argwhere(np.isnan(X).sum(axis=0) > 0).reshape(-1)
        if len(self.nan_features_) > 0:
            X_notnan = np.delete(X, self.nan_features_, axis=1)
            X_nan = np.copy(X)[:, self.nan_features_]
            X_new = np.hstack((X_notnan,
                               np.where(np.isnan(X_nan), -1000, X_nan),
                               np.where(np.isnan(X_nan), 1000, X_nan)))
        else:
            X_new = np.copy(X)
        self.rf_.fit(X_new, y)
        return self

    def predict(self, X):
        if len(self.nan_features_) > 0:
            X_notnan = np.delete(X, self.nan_features_, axis=1)
            X_nan = np.copy(X)[:, self.nan_features_]
            X_new = np.hstack((X_notnan,
                               np.where(np.isnan(X_nan), -1000, X_nan),
                               np.where(np.isnan(X_nan), 1000, X_nan)))
        else:
            X_new = np.copy(X)
        # Deal with unseen missing values during training
        unseen_nan_features = np.argwhere(np.isnan(X_new).sum(axis=0) > 0).reshape(-1)
        if len(unseen_nan_features) > 0:
            X_new = np.where(np.isnan(X_new), -1000, X_new)
        return self.rf_.predict(X_new)
