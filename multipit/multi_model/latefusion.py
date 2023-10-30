from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines.statistics import logrank_test
from sklearn.base import clone, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


class LateFusionClassifier(BaseEstimator):
    """
    Late fusion classifier for multimodal integration.

    Parameters
    ----------
    estimators: list of (str, estimator, list, dict) tuples.
        List of unimodal estimators to fit and fuse. Each unimodal estimator is associate with a tuple
        (`name`, `estimator` , `features`, `tune_dict`) where `name` is a string and corresponds to the name of the
        estimator, `estimator` ia scikit-learn estimator inheriting from BaseEstimator (can be a Pipeline), `features`
        is a list of indexes corresponding to the columns of the data associated with the modality of intereset, and
        `tune_dict` is either a dictionnary or a tuple (dict, n_iterations) for hyperparameter tuning and GridSearch or
        RandomSearch strategy respectively.

    cv: cross-validation generator
        cross-validation scheme for hyperparameter tuning (if `tuning` is not None) and/or calibration (if `calibration`
        is not None). The default is None

    score: str or callable.
        Score to use for tuning the unimodal models or weighting them at the late fusion step (i.e. sum of the unimodal
        predictions weighted by the performance of each unimodal model estimated with cross-validation).
        See sklearn.model_selection.cross_val_score for more details. The default is None.

    random_score: float.
        Random score for classification. Used when weighting the unimodal models with their estimated score. Weights
        will be max(score - random_score, 0). Unimodal models whose estimated performance is below the random_score
        will not be taken into account. The default is 0.5

    sup_weights: bool.
        Whether to use weights associated with the cross-validation performance of each unimodal model. If false no
        weights are used when fusing the unimodal predictions. The default is False.

    missing_threshold: float in ]0, 1].
        Minimum frequency of missing values to consider a whole modality missing (e.g., if `missing_threshold = 0.9` it
        means that for each sample and each modality at least 90% of the features associated with this modality must be
        missing to consider the whole modality missing). The default is 0.9.

    tuning: str or None.
        Strategy for tuning each model. Either 'gridsearch' for GridSearchCV or 'randomsearch' for RandomSearchCV. If
        None no hyperparameter tuning will be performed. The default is None.

    n_jobs: int.
        Number of jobs to run in parallel for hyperparameter tuning, collecting the predictions for calibration, or
        estimating the performance of each unimodal model with cross-validation. The default is None.

    calibration: str or None.
        Calibration strategy.
            * `calibration = 'late'` means that the fusion is made before calibration. The predictions of each
            multimodal combination are collected with cross-validation and a univariate logistic regression model is
            fitted to these predictions.
            * `calibration = 'early'` means that each unimodal model is calibrated prior to the late fusion. The
            unimodal predictions are collected with a cross-validation scheme and univariate logistic regression models
            are fitted.
            * `calibration = None` means that no calibration is performed.

    Attributes
    ----------
    best_params_: list of dict or empty list.
        List of best parameters for each unimodal predictor (output of GridSearchCV or RandomSearchCV). It follows the
        same order as the one of `estimators` list. If `tuning` is None returns an empty list (i.e., no hyperparameter
        tuning is performed).

    weights_: list of float.
        List of the weights associated to each modality and used at the late fusion stage for weighted sum.

    fitted_estimators_: list of estimators.
        List of fitted unimodal estimators.

    fitted_meta_estimators_: dictionary of estimators.
        Dictionary of meta-estimators for calibration. If `calibration = "early"` the keys correspond to the indexes of
        each unimodal estimator (i.e., from 0 to n_estimators-1) and the values correspond to the logistic regression
        estimators fitted to calibrate the unimodal models. If `calibration = "late"` the keys correspond to tuples
        characterizing each multimodal combination (e.g., (1, 3, 5)) and the values correspond th the logistic regression
        estimators fitted to clibrate the multimodal models.
    """

    def __init__(self, estimators, cv=None, score=None, random_score=0.5, sup_weights=False,
                 missing_threshold=0.9, tuning=None, n_jobs=None, calibration="late"):
        self.estimators = estimators
        self.cv = cv
        self.score = score
        self.random_score = random_score
        self.sup_weights = sup_weights
        self.missing_threshold = missing_threshold
        self.tuning = tuning
        self.n_jobs = n_jobs
        self.calibration = calibration

        self.best_params_ = []
        self.weights_ = []
        self.fitted_estimators_ = []
        self.fitted_meta_estimators_ = {}

    def fit(self, X, y):
        """
        Fit the latefusion classifier.

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
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        weights = np.zeros((X.shape[0], len(self.estimators)))
        i = 0
        for name, estim, features, grid in self.estimators:
            Xnew = X[:, features]
            # bool_mask = ~(np.sum(np.isnan(Xnew), axis=1) > self.missing_threshold * len(features))
            bool_mask = ~(np.sum(pd.isnull(Xnew), axis=1) > self.missing_threshold * len(features))
            Xnew, ynew = Xnew[bool_mask, :], y[bool_mask]
            # Fit unimodal estimator
            self._fit_estim(Xnew, ynew, estim=estim, features=features, grid=grid, name=name)

            # Collect predictions and weights for further calibration
            if self.calibration is not None:
                weights[bool_mask, i] = max(self.weights_[-1] - self.random_score, 0)
                parallel = Parallel(n_jobs=self.n_jobs)
                collected_predictions = parallel(delayed(_collect)(Xdata=X,
                                                                   ydata=y,
                                                                   estimator=estim,
                                                                   bmask=bool_mask,
                                                                   feat=features,
                                                                   train=train,
                                                                   test=test)
                                                 for train, test in self.cv.split(X, y))
                for indexes, preds in collected_predictions:
                    predictions[indexes, i] = preds
            i += 1

        # Calibrate models
        if self.calibration is not None:
            if self.calibration == "early":
                self._fit_early_calibration(predictions=predictions, y=y)
            elif self.calibration == "late":
                self._fit_late_calibration(predictions=predictions, weights=weights, y=y)
            else:
                raise ValueError("'early', 'late' or None are the only values available for calibration parameter")

        self.weights_ = np.array(self.weights_) - self.random_score
        self.weights_ = np.where(self.weights_ > 0, self.weights_, 0)
        return self

    def _fit_estim(self, X, y, estim, features, grid, name):
        """
        Fit a unimodal estimator.
        """
        if (self.tuning is not None) and (len(grid) > 0):
            if self.tuning == 'gridsearch':
                search = GridSearchCV(estimator=clone(estim), param_grid=grid, cv=self.cv, scoring=self.score,
                                      n_jobs=self.n_jobs)
            elif self.tuning == 'randomsearch':
                search = RandomizedSearchCV(estimator=clone(estim), param_distributions=grid[1], n_iter=grid[0],
                                            scoring=self.score, n_jobs=self.n_jobs, cv=self.cv)

            search.fit(X, y)

            if self.sup_weights:
                self.weights_.append(search.best_score_)
            else:
                self.weights_.append(1.)

            temp = search.best_estimator_
            self.best_params_.append(search.best_params_)
            # print("Best params " + name + " :", search.best_params_)
            # print("Best score " + name + " :", search.best_score_)
        else:
            if self.sup_weights:
                self.weights_.append(np.mean(
                    cross_val_score(estimator=clone(estim), X=X, y=y, cv=self.cv, scoring=self.score,
                                    n_jobs=self.n_jobs)))
            else:
                self.weights_.append(1.)
            temp = clone(estim).fit(X, y)

        self.fitted_estimators_.append((name, temp, features))
        return

    def _fit_early_calibration(self, predictions, y):
        """
        Calibrate only each unimodal predictor.
        """
        for i in range(len(self.estimators)):
            probas = predictions[:, i]
            mask = (probas > 0).reshape(-1)
            self.fitted_meta_estimators_[i] = LogisticRegression(class_weight='balanced').fit(
                probas[mask].reshape(-1, 1), y[mask])
        return

    def _fit_late_calibration(self, predictions, weights, y):
        """
        Calibrate each combination of modalities.
        """
        for i in range(1, len(self.estimators) + 1):
            for comb in combinations(range(len(self.estimators)), i):
                probas = predictions[:, np.array(comb)]
                if len(comb) == 1:
                    mask = (probas > 0).reshape(-1)
                else:
                    w = weights[:, np.array(comb)]
                    mask = np.any(probas > 0, axis=1).reshape(-1)
                    temp = np.sum(w, axis=1)
                    w[temp > 0] = w[temp > 0] / (temp[temp > 0].reshape(-1, 1))
                    probas = np.sum(probas * w, axis=1)

                self.fitted_meta_estimators_[comb] = LogisticRegression(class_weight='balanced').fit(
                    probas[mask].reshape(-1, 1), y[mask])
        return

    def predict_proba(self, X, estim_ind=None):
        """
        Late fusion probability estimates

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        estim_ind: tuple of integers.
            Tuple representing a multimodal combination (e.g. (i, j, k) corresponds to the combination of the ith, the
            jth and the kth estimators in self.fitted_estimators_). If None all the multimodal combination with all the
            fitted unimodal predictors is considered.

        Returns
        -------
        probas: array of shape (n_samples, 2).
            Probability of the samples for each class. If no modality are availbale for the sample, returns 0.5 for
            both classes.
        """
        fitted_estimators = [self.fitted_estimators_[i] for i in estim_ind] if estim_ind is not None \
            else self.fitted_estimators_
        fitted_weights = np.array([self.weights_[i] for i in estim_ind]) if estim_ind is not None else self.weights_

        # Collect predictions for each modality
        preds = np.zeros((X.shape[0], len(fitted_estimators)))
        weights = np.zeros((X.shape[0], len(fitted_weights)))
        for j, item in enumerate(fitted_estimators):
            Xpred = X[:, item[2]].copy()
            # bool_mask = ~(np.sum(np.isnan(Xpred), axis=1) > self.missing_threshold * len(item[2]))
            bool_mask = ~(np.sum(pd.isnull(Xpred), axis=1) > self.missing_threshold * len(item[2]))
            weights[:, j] = np.where(bool_mask, fitted_weights[j], 0)
            preds[bool_mask, j] = item[1].predict_proba(Xpred[bool_mask, :])[:, 1]

        # Calibrate the predictions and predict probas
        if self.calibration is not None:
            if self.calibration == "late":
                probas = self._predict_calibrate_late(preds, weights, estim_ind)
            elif self.calibration == "early":
                probas = self._predict_calibrate_early(preds, weights, estim_ind)
            else:
                raise ValueError("'early', 'late' or None are the only values available for calibration parameter")
        else:
            probas = self._predict_uncalibrated(preds, weights)
        return np.hstack([1 - probas, probas])

    @staticmethod
    def _predict_uncalibrated(preds, weights):
        """
        Return weighted sum of available unimodal predictions
        """
        temp = np.sum(weights, axis=1)
        weights[temp > 0] = weights[temp > 0] / (temp[temp > 0].reshape(-1, 1))
        probas = np.sum(preds * weights, axis=1)
        return np.where(temp == 0, 0.5, probas).reshape(-1, 1)

    def _predict_calibrate_early(self, preds, weights, estim_ind):
        """
        Return weighted sum of available and calibrated unimodal predictions
        """
        temp = np.sum(weights, axis=1)
        weights[temp > 0] = weights[temp > 0] / (temp[temp > 0].reshape(-1, 1))
        list_meta_estimators = [self.fitted_meta_estimators_[i] for i in estim_ind] if estim_ind is not None \
            else list(self.fitted_meta_estimators_.values())
        for j, meta in enumerate(list_meta_estimators):
            preds[:, j] = np.where(weights[:, j] != 0, meta.predict_proba(preds[:, j].reshape(-1, 1))[:, 1], 0)
        probas = np.sum(preds * weights, axis=1)
        return np.where(temp == 0, 0.5, probas).reshape(-1, 1)

    def _predict_calibrate_late(self, preds, weights, estim_ind):
        """
        Return calibrated weighted sum of availbale unimodal predictions
        """
        temp = np.sum(weights, axis=1)
        weights[temp > 0] = weights[temp > 0] / (temp[temp > 0].reshape(-1, 1))
        probas = np.sum(preds * weights, axis=1)
        meta_estimator = self.fitted_meta_estimators_[estim_ind] if estim_ind is not None \
            else list(self.fitted_meta_estimators_.values())[-1]
        return np.where(temp == 0, 0.5, meta_estimator.predict_proba(probas.reshape(-1, 1))[:, 1]).reshape(-1, 1)

    def find_logrank_threshold(self, X, ysurv, estim_ind, percentile_min=30, percentile_max=70):
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

        estim_ind: tuple of integers.
            Tuple representing a multimodal combination (e.g. (i, j, k) corresponds to the combination of the ith, the
            jth and the kth estimators in self.fitted_estimators_). If None all the multimodal combination with all the
            fitted unimodal predictors is considered.

        percentile_min: int in [0, 100]
            Minimum value of the percentile range used to explore various cutoff values for predicted probabilities

        percentile_max: int in [0, 100]
            Maximum value of the percentile range used to explore various cutoff values for predicted probabilities
            
        Returns
        -------
        cutoff: float.
            Best cutoff for the predicted probabilities that otpimize the log-rank test.
        """
        risk_score = self.predict_proba(X, estim_ind=estim_ind)[:, 1]
        bool_mask = risk_score == 0.5
        cutoffs, pvals = [], []
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


def _collect(Xdata, ydata, estimator, bmask, feat, train, test):
    Xtrain, Xtest, ytrain, ytest = Xdata[np.intersect1d(np.where(bmask)[0], train), :], \
                                   Xdata[np.intersect1d(np.where(bmask)[0], test), :], \
                                   ydata[np.intersect1d(np.where(bmask)[0], train)], \
                                   ydata[np.intersect1d(np.where(bmask)[0], test)]
    tempbis = clone(estimator).fit(Xtrain[:, feat], ytrain)
    return np.intersect1d(np.where(bmask)[0], test), tempbis.predict_proba(Xtest[:, feat])[:, 1]


class LateFusionSurvival(BaseEstimator):
    """
    Late fusion survival model for multimodal integration.

    Parameters
    ----------
     estimators: list of (str, estimator, list, dict) tuples.
        List of unimodal estimators to fit and fuse. Each unimodal estimator is associate with a tuple
        (`name`, `estimator` , `features`, `tune_dict`) where `name` is a string and corresponds to the name of the
        estimator, `estimator` ia scikit-learn estimator inheriting from BaseEstimator (can be a Pipeline), `features`
        is a list of indexes corresponding to the columns of the data associated with the modality of intereset, and
        `tune_dict` is either a dictionnary or a tuple (dict, n_iterations) for hyperparameter tuning and GridSearch or
        RandomSearch strategy respectively.

    cv: cross-validation generator
        cross-validation scheme for hyperparameter tuning (if `tuning` is not None) and/or calibration (if `calibration`
        is not None). The default is None

    score: str or callable.
        Score to use for tuning the unimodal models or weighting them at the late fusion step (i.e. sum of the unimodal
        predictions weighted by the performance of each unimodal model estimated with cross-validation).
        See sklearn.model_selection.cross_val_score for more details. The default is None.

    random_score: float.
        Random score for classification. Used when weighting the unimodal models with their estimated score. Weights
        will be max(score - random_score, 0). Unimodal models whose estimated performance is below the random_score
        will not be taken into account. The default is 0.5

    sup_weights: bool.
        Whether to use weights associated with the cross-validation performance of each unimodal model. If false no
        weights are used when fusing the unimodal predictions. The default is False.

    missing_threshold: float in ]0, 1].
        Minimum frequency of missing values to consider a whole modality missing (e.g., if `missing_threshold = 0.9` it
        means that for each sample and each modality at least 90% of the features associated with this modality must be
        missing to consider the whole modality missing). The default is 0.9.

    tuning: str or None.
        Strategy for tuning each model. Either 'gridsearch' for GridSearchCV or 'randomsearch' for RandomSearchCV. If
        None no hyperparameter tuning will be performed. The default is None.

    n_jobs: int.
        Number of jobs to run in parallel for hyperparameter tuning, collecting the predictions for calibration, or
        estimating the performance of each unimodal model with cross-validation. The default is None.

    calibration: bool.
        If True each unimodal model is associated with a tuple (mean, std) estimated on predictions collected with
        cross-validation. The predictions of each unimodal model are then standardized before the late fusion step.

    Attributes
    ----------
    best_params_: list of dict or empty list.
        List of best parameters for each unimodal predictor (output of GridSearchCV or RandomSearchCV). It follows the
        same order as the one of `estimators` list. If `tuning` is None returns an empty list (i.e., no hyperparameter
        tuning is performed).

    weights_: list of float.
        List of the weights associated to each modality and used at the late fusion stage for weighted sum.

    fitted_estimators_: list of estimators.
        List of fitted unimodal estimators.
    """

    def __init__(self, estimators, cv, score=None, random_score=0.5, sup_weights=True, missing_threshold=0.9,
                 tuning=None, n_jobs=None, calibration=True):
        self.estimators = estimators
        self.cv = cv
        self.score = score
        self.random_score = random_score
        self.sup_weights = sup_weights
        self.missing_threshold = missing_threshold
        self.tuning = tuning
        self.n_jobs = n_jobs
        self.calibration = calibration

        self.weights_ = []
        self.fitted_estimators_ = []
        self.best_params_ = []

    def _fit_estim(self, X, y, estim, features, grid, name):

        if (self.tuning is not None) and (len(grid) > 0):
            if self.tuning == 'gridsearch':
                search = GridSearchCV(estimator=clone(estim), param_grid=grid, cv=self.cv, scoring=self.score,
                                      n_jobs=self.n_jobs)

            elif self.tuning == 'randomsearch':
                search = RandomizedSearchCV(estimator=clone(estim), param_distributions=grid[1], n_iter=grid[0],
                                            scoring=self.score, n_jobs=self.n_jobs, cv=self.cv)

            search.fit(X, y)

            if self.sup_weights:
                self.weights_.append(search.best_score_)
            else:
                self.weights_.append(1.)

            temp = search.best_estimator_
            self.best_params_.append(search.best_params_)
            # print("Best params " + name + " :", search.best_params_)
            # print("Best score " + name + " :", search.best_score_)
        else:
            if self.sup_weights:
                self.weights_.append(np.mean(
                    cross_val_score(estimator=clone(estim), X=X, y=y, cv=self.cv, scoring=self.score)))
            else:
                self.weights_.append(1.)
            temp = clone(estim).fit(X, y)

        # self.fitted_estimators_.append((name, temp, features))
        return temp

    def fit(self, X, y):
        """
        Fit the latefusion survival model.

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
        for name, estim, features, grid in self.estimators:
            Xnew = X[:, features]
            bool_mask = ~(np.sum(np.isnan(Xnew), axis=1) > self.missing_threshold * len(features))
            Xnew, ynew = Xnew[bool_mask, :], y[bool_mask]

            fitted_estim = self._fit_estim(Xnew, ynew, estim=estim, features=features, grid=grid, name=name)
            if self.calibration:
                parallel = Parallel(n_jobs=self.n_jobs)
                collected_predictions = parallel(delayed(_collect_surv)(Xdata=X,
                                                                        ydata=y,
                                                                        estimator=estim,
                                                                        bmask=bool_mask,
                                                                        feat=features,
                                                                        train=train,
                                                                        test=test)
                                                 for train, test in self.cv.split(X, y))
                temp = np.concatenate(collected_predictions)
                mean, std = np.mean(temp), np.std(temp)
            else:
                mean, std = None, None
            self.fitted_estimators_.append((name, fitted_estim, features, (mean, std)))

        self.weights_ = np.array(self.weights_) - self.random_score
        self.weights_ = np.where(self.weights_ > 0, self.weights_, 0)
        # if np.sum(self.weights_) > 0:
        #    self.weights_ = self.weights_/np.sum(self.weights_)
        return self

    def predict(self, X, estim_ind=None):
        """
        Predict risk scores

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Multimodal array, concatenation of the features from the different modalities. Missing modalities are filled
            with NaNs values for each sample.

        estim_ind: tuple of integers.
            Tuple representing a multimodal combination (e.g. (i, j, k) corresponds to the combination of the ith, the
            jth and the kth estimators in self.fitted_estimators_). If None all the multimodal combination with all the
            fitted unimodal predictors is considered.

        Returns
        -------
        risk_scores: array of shape (n_samples,).
            Predictied risk scores. If no modality are availbale for the sample, returns 0.
        """
        if estim_ind is not None:
            fitted_estimators = [self.fitted_estimators_[i] for i in estim_ind]
        else:
            fitted_estimators = self.fitted_estimators_

        preds = np.zeros((X.shape[0], len(fitted_estimators)))
        weights = np.zeros((X.shape[0], len(fitted_estimators)))
        for j, item in enumerate(fitted_estimators):
            Xpred = X[:, item[2]].copy()
            bool_mask = ~(np.sum(np.isnan(Xpred), axis=1) > self.missing_threshold * len(item[2]))
            weights[:, j] = np.where(bool_mask, self.weights_[j], 0)
            if self.calibration:
                mean = item[3][0]
                std = item[3][1] if item[3][1] != 0 else 1
                preds[bool_mask, j] = (item[1].predict(Xpred[bool_mask, :]) - mean) / std
            else:
                preds[bool_mask, j] = item[1].predict(Xpred[bool_mask, :])
        temp = np.sum(weights, axis=1)
        weights[temp > 0] = weights[temp > 0] / (temp[temp > 0].reshape(-1, 1))
        return np.sum(preds * weights, axis=1)

    def find_logrank_threshold(self, X, y, estim_ind, percentile_min=30, percentile_max=70):
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

        estim_ind: tuple of integers.
            Tuple representing a multimodal combination (e.g. (i, j, k) corresponds to the combination of the ith, the
            jth and the kth estimators in self.fitted_estimators_). If None all the multimodal combination with all the
            fitted unimodal predictors is considered.

        percentile_min: int in [0, 100]
            Minimum value of the percentile range used to explore various cutoff values for predicted probabilities

        percentile_max: int in [0, 100]
            Maximum value of the percentile range used to explore various cutoff values for predicted probabilities

        Returns
        -------
        cutoff: float.
            Best cutoff for the predicted probabilities that otpimize the log-rank test.
        """
        risk_score = self.predict(X, estim_ind=estim_ind)
        bool_mask = risk_score == 0
        cutoffs, pvals = [], []
        risk_score_new, y_new = risk_score[~bool_mask], y[~bool_mask]
        for p in np.arange(percentile_min, percentile_max+1):
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


def _collect_surv(Xdata, ydata, estimator, bmask, feat, train, test):
    Xtrain, Xtest, ytrain, ytest = Xdata[np.intersect1d(np.where(bmask)[0], train), :], \
                                   Xdata[np.intersect1d(np.where(bmask)[0], test), :], \
                                   ydata[np.intersect1d(np.where(bmask)[0], train)], \
                                   ydata[np.intersect1d(np.where(bmask)[0], test)]
    tempbis = clone(estimator).fit(Xtrain[:, feat], ytrain)
    return tempbis.predict(Xtest[:, feat])
