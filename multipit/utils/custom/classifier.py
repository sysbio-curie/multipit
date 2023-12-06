import xgboost as xgb


class CustomXGBoostClassifier(xgb.sklearn.XGBClassifier):
    """
    Custom XGBoost classifier to deal with class imbalance.

    """

    def __init__(self, **kwargs):
        super(CustomXGBoostClassifier, self).__init__(**kwargs)

    def fit(self, X, y, feature_weights=None):
        """
        Fit the classifier to the provided training data.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            Training data.

        y: array of shape (n_samples,)
            Target vector.

        feature_weights: array of shape (n_features,)
            Weights assigned to features. The default is None.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super(CustomXGBoostClassifier, self).set_params(
            scale_pos_weight=(y == 0).sum() / (y == 1).sum()
        )
        super(CustomXGBoostClassifier, self).fit(
            X=X, y=y, feature_weights=feature_weights
        )
        return self
