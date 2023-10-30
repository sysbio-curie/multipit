import inspect
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from lifelines.statistics import logrank_test
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from multipit.preprocessing import CustomSelection


def _clone(estim):
    if estim is None:
        return None
    return clone(estim)


class EarlyTransform(BaseEstimator, TransformerMixin):
    """
    Transform each modality individually and concatenate the results. For each modality only the samples with this
    modality available are transformed. The others are filled with NaN values.

    Parameters
    ----------
    modalities: dict
        Dictionary with each key corresponding to the modality name and the value corresponding to the list of indexes
        for the features/columns of that modality (e.g., {'modality_1': [0, 1, 2], 'modality_2': [3, 4, 5, 6]}).

    transformers: dict
        Dictionary with each key corresponding to the modality name and the value corresponding to the transformer to
        apply to that modality. If the transformer is None, no transformation is applied to the data associated with
        this modality (e.g., {'modality_1':  sklearn.preprocessing.StandardScaler(), 'modality_2': None}.
        
    Attributes
    ----------
    transformed_modalities_: dict
        Dictionary with each key corresponding to the modality name and the value corresponding to the list of new
        indexes for the features/columns associated with this modality for the transformed data.
    """

    def __init__(self, modalities, transformers):
        self.modalities = modalities
        self.transformers = transformers
        self.transformed_modalities_ = {}

    def fit(self, X, y=None):
        """
        Fit each transformer on the samples for which the associated modality is available.

        Parameters
        ----------
         X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        y: array or structured array of shape (n_samples,)
            Target for supervised transformation (e.g., feature selection). The default is None.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        for modality, transformer in self.transformers.items():
            if transformer is not None:
                temp = X[:, self.modalities[modality]]
                mask = np.isnan(temp).sum(axis=1) == temp.shape[1]
                if y is not None:
                    transformer.fit(temp[~mask], y=y[~mask])
                else:
                    transformer.fit(temp[~mask])
        return self

    def transform(self, X):
        """
        Apply the fitted transformers to their associated modality and concatenate the results.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        Returns
        -------
        transformed_data: array of shape (n_samples, n_new_features)
            Concatenation of the transformed data modalities.
        """
        l_transformed = []
        n = 0
        for modality, transformer in self.transformers.items():
            features = self.modalities[modality]
            if transformer is not None:
                temp = X[:, features]
                mask = np.isnan(temp).sum(axis=1) == len(features)
                X_temp = transformer.transform(temp[~mask])
                X_transformed = np.full((X.shape[0], X_temp.shape[1]), np.nan)
                X_transformed[~mask, :] = X_temp
            else:
                X_transformed = np.copy(X)[:, features]
            l_transformed.append(X_transformed)
            self.transformed_modalities_[modality] = np.arange(n, n + X_transformed.shape[1])
            n += X_transformed.shape[1]
        return np.hstack(l_transformed)


class EarlyFusionClassifier(BaseEstimator):
    """
    Early fusion classifier for multimodal integration

    Parameters
    ----------
    estimator: estimator
        Estimator to apply to the transformed and concatenated multimodal data.

    modalities: dict
        Dictionary with each key corresponding to the modality name and the value corresponding to the list of indexes
        for the features/columns of that modality (e.g., {'modality_1': [0, 1, 2], 'modality_2': [3, 4, 5, 6]}).

    transformers: dict
        Dictionary with each key corresponding to the modality name and the value corresponding to the transformer to
        apply to that modality. If the transformer is None, no transformation is applied to the data associated with
        this modality (e.g., {'modality_1':  sklearn.preprocessing.StandardScaler(), 'modality_2': None}.

    n_jobs: int.
        Number of jobs to run in parallel for collecting the predictions for calibration. The default is None.

    calibration: bool.
        If True the earlyfusion predictions are collected with a cross-validation scheme and a univariate logistic
        regression model is fitted to these predictions. The default is True.

    cv: cross-validation generator
        cross-validation scheme for calibration (if `calibration` is True). The default is None

    balance_features: bool.
        If True a vector of weights of size (n_transformed_features,) is created, where each feature is associated to a
        weight 1/modality_size (modality_size:number of features associated to the modality this feature belongs to). It
        will be used as input to the fit method of the estimator, to balance the different modalities (i.e. equal
        chances to select features from the different modalities). The fit method of the estimator must have a
        `feature_weight' parameter. The default is False.

    select_features: bool.
        If True univariate feature selection will be used as a preprocessing step to select features from the different
        modalities. The default is False.

    select_equal_sizes: bool.
        If True and `select_features` is True equak numbers of features will be selected for each modality. The default
        is True.

    max_features: int.
        Maximum number of features to select with univariate selection. If `select_equal_sizes` is True,
        max_features/n_modalities (n_modalities: total number of modalities) features will be selected for each
        modality. If None all the features (filtered with max_corr and threshold_select) are selected. The default is
        None.

    max_corr: float in ]0, 1].
        Maximum correlation threshold. During the selection step, going from the most imformative feature to the least
        imformative one, remove the less important features whose correlation is higher than this threshold. If None no
        correlation-based filtering is performed. The default is None.

    threshold_select: float.
        Minimum performance value. Only select features whose performance is higher than this threshold. If None no
        performance-based filtering is performed. The default is None.

    Attributes
    ----------
    earlytransform_: EarlyTransform object
        Fitted early transformer (see multipit.multi_model.earlyfusion.EarlyTransform class).

    select_transform_: CustomSelection object or None
        Fitted custom selection object (see multipit.preprocessing.CustomSelection class) if `select_features` is True
        (None otherwise).

    fitted_estimator_: estimator object
        Fitted estimator.

    calibrator_: sklear.linear_model.LogisticRegression object or None
        Fitted calibrator if `calibration` is True (None otherwise).
    """

    def __init__(self, estimator, modalities, transformers, n_jobs=None, calibration=True, cv=None,
                 balance_features=False, select_features=False, select_equal_sizes=True, max_features=None,
                 max_corr=None, threshold_select=None):
        self.estimator = estimator
        self.modalities = modalities
        self.transformers = transformers
        self.calibration = calibration
        self.n_jobs = n_jobs
        self.cv = cv
        self.balance_features = balance_features
        self.select_features = select_features
        self.select_equal_sizes = select_equal_sizes
        self.max_features = max_features
        self.max_corr = max_corr
        self.threshold_select = threshold_select

        self.calibrator_ = None
        self.select_transform_ = None
        self.earlytransform_ = None

    def _collect_preds(self, X, y, train, test):
        X_train, y_train, X_test = X[train, :], y[train], X[test, :]
        transformer = EarlyTransform(modalities=self.modalities,
                                     transformers={moda: _clone(transform) for moda, transform in
                                                   self.transformers.items()})
        X_train_transformed = transformer.fit(X_train, y_train).transform(X_train)
        if self.select_features:
            select_transform = CustomSelection(threshold=self.threshold_select,
                                               max_corr=self.max_corr,
                                               max_number=self.max_features)
            if self.select_equal_sizes:
                modalities = []
                for i, mod in enumerate(transformer.transformed_modalities_.values()):
                    modalities += [i] * len(mod)
                modalities = np.array(modalities)
            else:
                modalities = None

            X_train_transformed = (select_transform.fit(X_train_transformed, y_train, modalities=modalities)
                                   .transform(X_train_transformed)
                                   )
        mask = np.isnan(X_train_transformed).sum(axis=1) == X_train_transformed.shape[1]
        clf = clone(self.estimator).fit(X_train_transformed[~mask, :], y_train[~mask])
        X_test_transformed = transformer.transform(X_test)
        mask_test = np.isnan(X_test_transformed).sum(axis=1) == X_test_transformed.shape[1]
        if self.select_features:
            X_test_transformed = select_transform.transform(X_test_transformed)
        return test[~mask_test], clf.predict_proba(X_test_transformed[~mask_test])[:, 1]

    def fit(self, X, y):
        """
        Fit the early fusion classifier

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        y: array of shape (n_samples,)
            Target to predict.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.earlytransform_ = EarlyTransform(modalities=self.modalities,
                                              transformers={moda: _clone(transform) for moda, transform in
                                                            self.transformers.items()})
        X_transformed = self.earlytransform_.fit(X, y).transform(X)
        if self.select_features:
            self.select_transform_ = CustomSelection(threshold=self.threshold_select,
                                                     max_corr=self.max_corr,
                                                     max_number=self.max_features,
                                                     predictive_task="classification")
            modalities = []
            for i, mod in enumerate(self.earlytransform_.transformed_modalities_.values()):
                modalities += [i] * len(mod)
            X_transformed = (self.select_transform_.fit(X_transformed, y, modalities=np.array(modalities))
                             .transform(X_transformed)
                             )
        mask = np.isnan(X_transformed).sum(axis=1) == X_transformed.shape[1]  # samples without any modality !
        if self.balance_features:
            fweights = []
            for _, val in self.earlytransform_.transformed_modalities_.items():
                fweights += [1 / len(val)] * len(val)
            self.fitted_estimator_ = clone(self.estimator).fit(X_transformed[~mask, :],
                                                               y[~mask],
                                                               feature_weights=fweights)
        else:
            self.fitted_estimator_ = clone(self.estimator).fit(X_transformed[~mask, :], y[~mask])

        if self.calibration:  # only on samples with a least one modality !
            predictions = np.full((len(y), 1), np.nan)  # np.zeros((len(y), 1))
            parallel = Parallel(n_jobs=self.n_jobs)
            collected_predictions = parallel(delayed(self._collect_preds)(X=X,
                                                                          y=y,
                                                                          train=train,
                                                                          test=test)
                                             for train, test in self.cv.split(X, y))
            for indexes, preds in collected_predictions:
                predictions[indexes, :] = preds.reshape(-1, 1)
            drop_empty_mask = np.isnan(predictions).reshape(-1)
            self.calibrator_ = LogisticRegression(class_weight='balanced').fit(predictions[~drop_empty_mask],
                                                                               y[~drop_empty_mask])
        return self

    def predict_proba(self, X):
        """
        Early fusion probability estimates

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        Returns
        -------
        probas: array of shape (n_samples, 2).
            Probability of the samples for each class. If no modality are availbale for the sample, returns 0.5 for
            both classes.
        """
        X_transformed = self.earlytransform_.transform(X)
        if self.select_features:
            X_transformed = self.select_transform_.transform(X_transformed)

        bool_mask = np.isnan(X_transformed).sum(axis=1) == X_transformed.shape[1]
        if self.calibration:
            probas = self.calibrator_.predict_proba(self.fitted_estimator_.predict_proba(
                X_transformed)[:, 1].reshape(-1, 1))
        else:
            probas = self.fitted_estimator_.predict_proba(X_transformed)
        return np.where((~bool_mask).reshape(-1, 1), probas, 0.5)

    def find_logrank_threshold(self, X, ysurv, percentile_min=30, percentile_max=70):
        """
          Find the best cutoff that optimize the stratification of samples with respect to survival data (using logrank
          test).

          Parameters
          ----------
          X: array of shape (n_samples, n_features)
              Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
              with NaNs values for each sample.

          ysurv: structured array of shape (n_samples,) see sksurv.util.Surv (from scikit-survival)
              Structured array for survival data associated with X.

          percentile_min: int in [0, 100]
              Minimum value of the percentile range used to explore various cutoff values for predicted probabilities

          percentile_max: int in [0, 100]
              Maximum value of the percentile range used to explore various cutoff values for predicted probabilities

          Returns
          -------
          cutoff: float.
              Best cutoff for the predicted probabilities that otpimize the log-rank test.
          """
        risk_score = self.predict_proba(X)[:, 1]
        cutoffs, pvals = [], []
        bool_mask = risk_score == 0.5
        risk_score_new, y_new = risk_score[~bool_mask], ysurv[~bool_mask]
        for p in np.arange(percentile_min, percentile_max + 1):
            c = np.percentile(risk_score_new, p)
            group1 = risk_score_new <= c
            group2 = risk_score_new > c
            test = logrank_test(durations_A=y_new[group1]['time'],
                                durations_B=y_new[group2]['time'],
                                event_observed_A=1 * (y_new[group1]['event']),
                                event_observed_B=1 * (y_new[group2]['event']),
                                )
            cutoffs.append(c)
            pvals.append(test.summary['p'].values[0])
        return cutoffs[np.argmin(pvals)]


class EarlyFusionSurvival(BaseEstimator):
    """
      Early fusion classifier for multimodal integration

      Parameters
      ----------
      estimator: estimator
          Estimator to apply to the transformed and concatenated multimodal data.

      modalities: dict
          Dictionary with each key corresponding to the modality name and the value corresponding to the list of indexes
          for the features/columns of that modality (e.g., {'modality_1': [0, 1, 2], 'modality_2': [3, 4, 5, 6]}).

      transformers: dict
          Dictionary with each key corresponding to the modality name and the value corresponding to the transformer to
          apply to that modality. If the transformer is None, no transformation is applied to the data associated with
          this modality (e.g., {'modality_1':  sklearn.preprocessing.StandardScaler(), 'modality_2': None}.

      n_jobs: int.
          Number of jobs to run in parallel for collecting the predictions for calibration. The default is None.

      calibration: bool.
          If True the earlyfusion predictions are collected with a cross-validation scheme and the mean and std are
          estimated for further standardization. The default is True.

      cv: cross-validation generator
          cross-validation scheme for calibration (if `calibration` is True). The default is None

      balance_features: bool.
          If True a vector of weights of size (n_transformed_features,) is created, where each feature is associated to a
          weight 1/modality_size (modality_size:number of features associated to the modality this feature belongs to). It
          will be used as input to the fit method of the estimator, to balance the different modalities (i.e. equal
          chances to select features from the different modalities). The fit method of the estimator must have a
          `feature_weight' parameter. The default is False.

      select_features: bool.
          If True univariate feature selection will be used as a preprocessing step to select features from the different
          modalities. The default is False.

      select_equal_sizes: bool.
          If True and `select_features` is True equak numbers of features will be selected for each modality. The default
          is True.

      max_features: int.
          Maximum number of features to select with univariate selection. If `select_equal_sizes` is True,
          max_features/n_modalities (n_modalities: total number of modalities) features will be selected for each
          modality. If None all the features (filtered with max_corr and threshold_select) are selected. The default is
          None.

      max_corr: float in ]0, 1].
          Maximum correlation threshold. During the selection step, going from the most imformative feature to the least
          imformative one, remove the less important features whose correlation is higher than this threshold. If None no
          correlation-based filtering is performed. The default is None.

      threshold_select: float.
          Minimum performance value. Only select features whose performance is higher than this threshold. If None no
          performance-based filtering is performed. The default is None.

      Attributes
      ----------
      earlytransform_: EarlyTransform object
          Fitted early transformer (see multipit.multi_model.earlyfusion.EarlyTransform class).

      select_transform_: CustomSelection object or None
          Fitted custom selection object (see multipit.preprocessing.CustomSelection class) if `select_features` is True
          (None otherwise).

      fitted_estimator_: estimator object
          Fitted estimator.

      calibrate_mean_: float.
        Estimated mean for calibration if `calibration` is True (None otherwise).

      calibrate_std_: float.
        Estimated std for calibration if `calibration` is True (None otherwise).
      """

    def __init__(self, estimator, modalities, transformers, n_jobs=1, balance_features=False, select_features=False,
                 calibration=True, select_equal_sizes=True, cv=None, max_features=None, max_corr=None,
                 threshold_select=None):
        self.estimator = estimator
        self.modalities = modalities
        self.transformers = transformers
        self.n_jobs = n_jobs
        self.balance_features = balance_features
        self.select_features = select_features
        self.select_equal_sizes = select_equal_sizes
        self.calibration = calibration
        self.cv = cv
        self.max_features = max_features
        self.max_corr = max_corr
        self.threshold_select = threshold_select

        self.calibrate_mean_, self.calibrate_std_ = None, None
        self.select_transform_ = None
        self.earlytransform_ = None

    def _collect_preds(self, X, y, train, test):
        X_train, y_train, X_test = X[train, :], y[train], X[test, :]
        transformer = EarlyTransform(modalities=self.modalities,
                                     transformers={moda: _clone(transform) for moda, transform in
                                                   self.transformers.items()})
        X_train_transformed = transformer.fit(X_train, y_train).transform(X_train)
        if self.select_features:
            select_transform = CustomSelection(threshold=self.threshold_select,
                                               max_corr=self.max_corr,
                                               max_number=self.max_features,
                                               predictive_task='survival')
            if self.select_equal_sizes:
                modalities = []
                for i, mod in enumerate(transformer.transformed_modalities_.values()):
                    modalities += [i] * len(mod)
                modalities = np.array(modalities)
            else:
                modalities = None
            X_train_transformed = (select_transform.fit(X_train_transformed, y_train, modalities=modalities)
                                   .transform(X_train_transformed)
                                   )
        mask = np.isnan(X_train_transformed).sum(axis=1) == X_train_transformed.shape[1]
        clf = clone(self.estimator).fit(X_train_transformed[~mask, :], y_train[~mask])
        X_test_transformed = transformer.transform(X_test)
        mask_test = np.isnan(X_test_transformed).sum(axis=1) == X_test_transformed.shape[1]
        if self.select_features:
            X_test_transformed = select_transform.transform(X_test_transformed)
        return test[~mask_test], clf.predict(X_test_transformed[~mask_test])

    def fit(self, X, y):
        """
        Fit the earlyusion survival model.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        y: structured array of shape (n_samples, ) see sksurv.util.Surv (from scikit-survival).
            Structured array for survival target/outcome

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.earlytransform_ = EarlyTransform(modalities=self.modalities,
                                              transformers={moda: _clone(transform) for moda, transform in
                                                            self.transformers.items()})
        X_transformed = self.earlytransform_.fit(X, y).transform(X)
        if self.select_features:
            self.select_transform_ = CustomSelection(threshold=self.threshold_select,
                                                     max_corr=self.max_corr,
                                                     max_number=self.max_features,
                                                     predictive_task='survival')
            modalities = []
            for i, mod in enumerate(self.earlytransform_.transformed_modalities_.values()):
                modalities += [i] * len(mod)
            X_transformed = (self.select_transform_.fit(X_transformed, y, modalities=np.array(modalities))
                             .transform(X_transformed)
                             )
        mask = np.isnan(X_transformed).sum(axis=1) == X_transformed.shape[1]

        self.fitted_estimator_ = clone(self.estimator).fit(X_transformed[~mask, :], y[~mask])
        # if self.balance_features:
        #     fweights = []
        #     for _, val in self.earlytransform_.transformed_modalities_.items():
        #         fweights += [1/len(val)]*len(val)
        #     self.fitted_estimator_ = clone(self.estimator).fit(X_transformed[~mask, :],
        #                                                        y[~mask],
        #                                                        feature_weights=fweights)
        # else:
        #     self.fitted_estimator_ = clone(self.estimator).fit(X_transformed[~mask, :], y[~mask])
        if self.calibration:  # only on samples with a least one modality !
            predictions = np.full((len(y), 1), np.nan)  # np.zeros((len(y), 1))
            parallel = Parallel(n_jobs=self.n_jobs)
            collected_predictions = parallel(delayed(self._collect_preds)(X=X,
                                                                          y=y,
                                                                          train=train,
                                                                          test=test)
                                             for train, test in self.cv.split(X, y))
            for indexes, preds in collected_predictions:
                predictions[indexes, :] = preds.reshape(-1, 1)
            drop_empty_mask = np.isnan(predictions).reshape(-1)
            self.calibrate_mean_ = np.mean(predictions[~drop_empty_mask])
            self.calibrate_std_ = np.std(predictions[~drop_empty_mask])
        return self

    def predict(self, X):
        """
        Predict risk scores

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        Returns
        -------
        risk_scores: array of shape (n_samples,).
            Predictied risk scores. If no modality are availbale for the sample, returns 0.
        """
        X_transformed = self.earlytransform_.transform(X)
        if self.select_features:
            X_transformed = self.select_transform_.transform(X_transformed)
        bool_mask = np.isnan(X_transformed).sum(axis=1) == X_transformed.shape[1]
        if self.calibration:
            preds = (self.fitted_estimator_.predict(X_transformed) - self.calibrate_mean_) / self.calibrate_std_
        else:
            preds = self.fitted_estimator_.predict(X_transformed)
        return np.where(~bool_mask, preds, 0)

    def find_logrank_threshold(self, X, y, percentile_min=30, percentile_max=70):
        """
        Find the best cutoff that optimize the stratification of samples with respect to survival data (using logrank
        test).

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        y: structured array of shape (n_samples,) see sksurv.util.Surv (from scikit-survival)
            Structured array for survival data associated with X.

        percentile_min: int in [0, 100]
            Minimum value of the percentile range used to explore various cutoff values for predicted probabilities

        percentile_max: int in [0, 100]
            Maximum value of the percentile range used to explore various cutoff values for predicted probabilities

        Returns
        -------
        cutoff: float.
            Best cutoff for the predicted probabilities that otpimize the log-rank test.
        """
        risk_score = self.predict(X)
        cutoffs, pvals = [], []
        bool_mask = risk_score == 0
        risk_score_new, y_new = risk_score[~bool_mask], y[~bool_mask]
        for p in np.arange(percentile_min, percentile_max + 1):
            c = np.percentile(risk_score_new, p)
            group1 = risk_score_new <= c
            group2 = risk_score_new > c
            test = logrank_test(durations_A=y_new[group1]['time'],
                                durations_B=y_new[group2]['time'],
                                event_observed_A=1 * (y_new[group1]['event']),
                                event_observed_B=1 * (y_new[group2]['event']),
                                )
            cutoffs.append(c)
            pvals.append(test.summary['p'].values[0])
        return cutoffs[np.argmin(pvals)]
