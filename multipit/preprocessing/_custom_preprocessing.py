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
    Custom imputer for missing values which deals with categorical variables with most frequent imputation and with
    numerical variables with median imputation

    Parameters
    ----------
    categoricals: list of integers.
        List of indexes associated to the categorical columns with missing values. If None, no categorical column is
        considered. The default is None.

    numericals: list of integers.
        List of indexes associated to the numerical columns with missing values. If None, no numerical column is
        considered. The default is None.

    Attributes
    ----------
    mask_cat_: 1D array of booleans.
        Boolean mask indicating the categorical columns with missing values.

    mask_num_: 1D array of booleans.
        Boolean mask indicating the numerical columns with missing values.
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
        Fit the custom imputer.
        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)

        y: Ignored

        Returns
        -------
        self: object
            Fitted estimator.
        """
        self.mask_cat_, self.mask_num_ = np.zeros(X.shape[1], bool), np.zeros(
            X.shape[1], bool
        )
        if self.categoricals is not None:
            self.mask_cat_[self.categoricals] = True
            self.imputer_cat.fit(X[:, self.mask_cat_])
        if self.numericals is not None:
            self.mask_num_[self.numericals] = True
            self.imputer_num.fit(X[:, self.mask_num_])
        return self

    def transform(self, X):
        """
        Impute missing values in X.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)

        Returns
        -------
        X_imputed: 2D array of shape (n_samples, n_features)
            X with imputed values

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
    Custom univariate selection for classification (based on AUC) and survival (based on C-index) tasks.

    Parameters
    ----------
    threshold: float in [O.5, 1].
        Threshold for the metric (either AUC or C-index). Features associated to a metric lower than this threshold will
        not be selected. If None, no threshold is applied. The default is None.

    max_corr: float in [0, 1]
        This parameter sets the threshold for the Pearson correlation. When analyzing feature performance, starting from
        the top-performing feature, all features with a Pearson correlation above this threshold are excluded.
        Subsequently, the algorithm considers the second-best performing feature among those that were not filtered out,
        and continues this process iteratively. If max_corr=1, no threshold is applied. The default is 0.8.

    max_number: int.
        Maximum number of selected features. If the number of remaining features after the different filtering steps is
        lower than max_number or if max_number is None all the remaining features are kept. The default is None.

    predictive_task: string in {"classification", "survival"}
        If predictive_task = "classification", the AUC is used and if predictive_task = "survival" the C-index is used.
        The defaults is "classification".

    Attributes
    ----------
    features_: list of integers.
        List of indexes corresponding to the selected features.
    """

    def __init__(
        self,
        threshold=None,
        max_corr=0.8,
        max_number=None,
        predictive_task="classification",
    ):
        self.threshold = threshold
        self.max_corr = max_corr
        self.max_number = max_number
        self.predictive_task = predictive_task

    def fit(self, X, y, modalities=None):
        """
        Fit the custom selection.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features).

        y: 1D array or structured array (see sksurv.util.Surv class) of shape (n_samples).
            If prediction_task = "classification" y corresponds to a binary outcome. If prediction_task = "survival"
            y corresponds to the event indicator and the observed time.

        modalities: 1D array of shape (n_features).
            This parameter deals with scenarios where features from different modalities are concatenated. It comprises
            labels indicating the membership of each feature to a specific modality. If `max_number` is not None, and
            the remaining features outnumber `max_number`, the algorithm selects the top `max_number/n_modalities`
            performing features within each modality. If None the different modalities are ignored. The default is None.

        Returns
        -------
        self: object
            Fitted estimator.
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
            raise ValueError(
                "Only 'survival' or 'classification' are available for predictive_task parameter"
            )
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
                    delete += list(
                        (i + 1) + np.where(corr[i, i + 1 :] > self.max_corr)[0]
                    )
            delete = np.unique(delete)
            if len(delete) > 0:
                self.features_ = np.delete(self.features_, delete)

        if modalities is not None and (
            self.max_number is not None and len(self.features_) > self.max_number
        ):
            n_modalities = len(np.unique(modalities))
            n_select_modalities = self.max_number // n_modalities
            modalities_ordered = modalities[self.features_]
            list_features = []
            for m in np.unique(modalities):
                list_features += list(
                    self.features_[modalities_ordered == m][:n_select_modalities]
                )
            self.features_ = np.array(list_features)
        else:
            if self.max_number is not None and len(self.features_) > self.max_number:
                self.features_ = self.features_[: self.max_number]
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
    A custom data scaler that allows for different scaling strategies and can be applied on a subset of features.

    Parameters
    ----------
    features: 1D array of shape (n_features,)
        Indices or labels of the features to be scaled. If None, all features are considered for scaling. The default is
        None.

    strategy: {'standardize', 'robust', 'minmax'}
        The strategy used for scaling. The default is 'standardize'.


    Attributes
    ----------
    scaler_: object
        Fitted scaler based on the specified strategy.

    """

    def __init__(self, features=None, strategy="standardize"):
        self.features = features
        self.strategy = strategy

    def fit(self, X, y=None):
        """
        Fit the custom scaler.

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

        if self.strategy == "standardize":
            self.scaler_ = StandardScaler()
        elif self.strategy == "robust":
            self.scaler_ = RobustScaler()
        elif self.strategy == "minmax":
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(
                "Only 'standardize', 'robust', or 'minmax' are available for the scaling strategy"
            )

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
        Transform the input data using the fitted scaler.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        Xnew : 2D array of shape (n_samples, n_features)
            Transformed data.
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
    A custom transformer for applying a logarithmic transformation to specified features.

    Parameters
    ----------
    features : 2D array of shape (n_features,)
        Indices or labels of the features to be transformed. If None, logarithmic transformation is applied to all
        features. The default is None.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the transformer has been fitted.
    """

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        """
        Fit the transformer to the provided data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Apply a logarithmic transformation to the input data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        Xnew : 2D array of shape (n_samples, n_features)
            Transformed data after applying the logarithmic transformation.
        """

        if self.features is None:
            Xnew = np.log(X + 1)
        else:
            Xnew = np.copy(X)
            Xnew[:, self.features] = np.log(Xnew[:, self.features] + 1)
        return Xnew


class CustomPCA(BaseEstimator, TransformerMixin):
    """
    A custom transformer applying PCA on input data (dealing with nan values).

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If `None`, all components are kept.

    whiten : bool
        When True, the components are whitened. The default is False

    Attributes
    ----------
    pca_ : PCA
        Fitted PCA object based on the provided parameters.
    """

    def __init__(self, n_components, whiten):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        """
        Fit the PCA transformer.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Training data.

        y : Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        # Missing values are disregarded in fit
        self.pca_.fit(X[np.sum(np.isnan(X), axis=1) == 0, :])
        return self

    def transform(self, X):
        """
        Apply PCA transformation to the input data.

        Parameters
        ----------
        X : 2D array of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_pca : 2D array of shape (n_samples, n_components)
            Transformed data after applying PCA.
        """

        return self.pca_.transform(X)
