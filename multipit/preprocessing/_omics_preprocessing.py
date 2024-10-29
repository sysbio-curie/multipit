import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


class CustomOmicsImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for imputing missing values and encoding categorical features in omics data.

    Parameters
    ----------
    site_feature : int
        Index of the site feature to be imputed and encoded.

    min_frequency : float, default=0.1
        Minimum frequency threshold for encoding infrequent categories.

    Attributes
    ----------
    imputer_ : KNNImputer
        Fitted KNNImputer for imputing missing values.

    encoder_ : OneHotEncoder
        Fitted OneHotEncoder for categorical encoding.

    len_encoding_ : int
        Length of the encoding after transformation.
    """

    def __init__(self, site_feature, min_frequency=0.1):
        self.site_feature = site_feature
        self.min_frequency = min_frequency
        self.imputer_ = None
        self.encoder_ = None
        self.len_encoding_ = None

    def fit(self, X, y=None):
        """
        Fit the CustomOmicsImputer to the provided data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.imputer_ = KNNImputer(n_neighbors=1)
        X[:, self.site_feature] = self.imputer_.fit_transform(X)[:, self.site_feature]
        self.encoder_ = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=self.min_frequency,
            sparse_output=False,
        ).fit(X[:, self.site_feature].reshape(-1, 1))
        return self

    def transform(self, X):
        """
        Transform the input data by imputing missing values and encoding categorical features.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        2D array of shape (n_samples, (n_features-1) + len_encoding_)
            Transformed data after imputation and encoding.
        """
        X[:, self.site_feature] = self.imputer_.transform(X)[:, self.site_feature]
        b = self.encoder_.transform(X[:, self.site_feature].reshape(-1, 1))
        self.len_encoding_ = b.shape[1]
        a = np.delete(X, self.site_feature, 1)
        return np.hstack((a, b))
