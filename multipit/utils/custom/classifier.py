import xgboost as xgb


class CustomXGBoostClassifier(xgb.sklearn.XGBClassifier):
    """

    """

    def __init__(self, **kwargs):
        super(CustomXGBoostClassifier, self).__init__(**kwargs)

    def fit(self, X, y, feature_weights=None):
        """
        Parameters
        ----------
        X:

        y:

        feature_weights:

        Returns
        -------
        self:
        """
        super(CustomXGBoostClassifier, self).set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        super(CustomXGBoostClassifier, self).fit(X=X, y=y, feature_weights=feature_weights)
        return self
