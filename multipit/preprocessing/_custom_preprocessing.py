import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sksurv.metrics import concordance_index_ipcw


class CustomImputer(_BaseImputer):
    """
    Parameters
    ----------
    categoricals:

    numericals:

    Attributes
    ----------
    mask_cat_:

    mask_num_:
    """

    def __init__(self, categoricals=None, numericals=None):
        super(CustomImputer, self).__init__()
        self.categoricals = categoricals
        self.numericals = numericals
        if self.categoricals is not None:
            self.imputer_cat = SimpleImputer(strategy="most_frequent")
        if self.numericals is not None:
            self.imputer_num = SimpleImputer(strategy="median")
        # assert (self.categoricals is not None) | (self.numericals is not None), ""

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X:

        y:
        Returns
        -------

        """
        if self.categoricals is not None:
            self.mask_cat_ = np.zeros(X.shape[1], bool)
            self.mask_cat_[self.categoricals] = True
            self.imputer_cat.fit(X[:, self.mask_cat_])
        if self.numericals is not None:
            self.mask_num_ = np.zeros(X.shape[1], bool)
            self.mask_num_[self.numericals] = True
            self.imputer_num.fit(X[:, self.mask_num_])
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X:

        Returns
        -------


        """
        X_imputed = np.copy(X)
        if self.categoricals is not None:
            X_imputed[:, self.mask_cat_] = self.imputer_cat.transform(
                X[:, self.mask_cat_]
            )
        if self.numericals is not None:
            X_imputed[:, self.mask_num_] = self.imputer_num.transform(
                X[:, self.mask_num_]
            )
        return np.float32(X_imputed)


class CustomSelection(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    threshold:

    max_corr:

    max_number:

    predictive_task:

    Attributes
    ----------
    features_

    """

    def __init__(self, threshold=None, max_corr=0.8, max_number=None, predictive_task="classification"):
        self.threshold = threshold
        self.max_corr = max_corr
        self.max_number = max_number
        self.predictive_task = predictive_task

    def fit(self, X, y, modalities=None):
        """
        Parameters
        ----------
        X:

        y:

        modalities:

        Returns
        -------
        """
        # Xmasked = X[np.sum(np.isnan(X), axis=1) == 0, :]
        # ymasked = y[np.sum(np.isnan(X), axis=1) == 0]
        # self.features_ = np.arange(Xmasked.shape[1])
        self.features_ = np.arange(X.shape[1])
        scores = np.zeros(X.shape[1])

        if self.predictive_task == "survival":
            for i in range(X.shape[1]):
                mask = np.isnan(y["event"]) | np.isnan(y["time"]) | np.isnan(X[:, i])
                c_index = concordance_index_ipcw(y, y[~mask], X[~mask][:, i])[0]
                scores[i] = max(c_index, 1 - c_index)
        elif self.predictive_task == "classification":
            for i in range(X.shape[1]):
                mask = np.isnan(y) | np.isnan(X[:, i])
                # auc = roc_auc_score(ymasked, Xmasked[:, i])
                auc = roc_auc_score(y[~mask], X[~mask][:, i])
                scores[i] = max(auc, 1 - auc)
        else:
            raise ValueError("Only 'survival' or 'classification' are available for predictive_task parameter")
        if self.threshold is not None:
            self.features_ = self.features_[scores >= self.threshold]
            assert len(self.features_) > 0
            scores = scores[scores >= self.threshold]
        self.features_ = self.features_[np.argsort(scores)[::-1]]

        # corr = np.abs(np.corrcoef(Xmasked[:, self.features_], rowvar=False))
        # corr = np.abs(_pearsonccs(Xmasked[:, self.features_], rowvar=False))
        if self.max_corr < 1:
            corr = np.abs(pd.DataFrame(X[:, self.features_]).corr()).values
            delete = []
            for i in range(len(self.features_) - 1):
                if i not in delete:
                    delete += list((i + 1) + np.where(corr[i, i + 1:] > self.max_corr)[0])
            delete = np.unique(delete)
            if len(delete) > 0:
                self.features_ = np.delete(self.features_, delete)

        if modalities is not None and (self.max_number is not None and len(self.features_) > self.max_number):
            n_modalities = len(np.unique(modalities))
            n_select_modalities = self.max_number // n_modalities
            modalities_ordered = modalities[self.features_]
            list_features = []
            for m in np.unique(modalities):
                list_features += list(self.features_[modalities_ordered == m][:n_select_modalities])
            self.features_ = np.array(list_features)
        else:
            if self.max_number is not None and len(self.features_) > self.max_number:
                self.features_ = self.features_[:self.max_number]
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X:

        Returns
        -------
        """
        return X[:, self.features_]


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    features:

    strategy:

    Attributes
    ----------
    scaler_:

    """

    def __init__(self, features=None, strategy='standardize'):
        self.features = features
        self.strategy = strategy

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X:

        y:

        Returns
        -------
        """
        if self.strategy == 'standardize':
            self.scaler_ = StandardScaler()
        elif self.strategy == 'robust':
            self.scaler_ = RobustScaler()
        elif self.strategy == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError("")

        # deal with cases where X is empty ?
        if X.shape[1] == 0:
            self.scaler_ = None
        else:
            if self.features is None:
                self.scaler_.fit(X)
            else:
                self.scaler_.fit(X[:, self.features])
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X:

        Returns
        -------

        """
        # deal with cases where X is empty ?
        if self.scaler_ is None:
            Xnew = np.copy(X)
        else:
            if self.features is None:
                Xnew = self.scaler_.transform(X)
            else:
                Xnew = np.copy(X)
                Xnew[:, self.features] = self.scaler_.transform(Xnew[:, self.features])
        return Xnew


class CustomLogTransform(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    features:

    Attributes
    ----------
    fitted_:

    """

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X:

        y:

        Returns
        -------
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X:

        Returns
        -------

        """
        if self.features is None:
            Xnew = np.log(X + 1)
        else:
            Xnew = np.copy(X)
            Xnew[:, self.features] = np.log(Xnew[:, self.features] + 1)
        return Xnew


class CustomPCA(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    n_components:

    whiten:

    Attributes
    ----------
    pca_:
    """

    def __init__(self, n_components, whiten):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X:

        y:

        Returns
        -------
        """
        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        # Missing values are disregarded in fit
        self.pca_.fit(X[np.sum(np.isnan(X), axis=1) == 0, :])
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X:

        Returns
        -------

        """
        return self.pca_.transform(X)
