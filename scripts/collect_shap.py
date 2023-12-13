import argparse
import inspect
import os
import sys

# import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import shap
from joblib import delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from init_scripts import PredictionTask
from utils import read_yaml, write_yaml, ProgressParallel

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from multipit.multi_model.latefusion import LateFusionClassifier


def main(params):
    """ """

    # 0. Read config file and save it in the results
    config = read_yaml(params.config)
    save_name = config["save_name"]
    if save_name is None:
        run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        save_name = "exp_" + run_id
    save_dir = os.path.join(params.save_path, save_name)
    os.mkdir(save_dir)
    write_yaml(config, os.path.join(save_dir, "config.yaml"))

    # 1. fix random seeds for reproducibility
    seed = config["latefusion"]["seed"]
    np.random.seed(seed)

    # 2. Load data and define pipelines for each modality
    ptask = PredictionTask(config, survival=False, integration="late")
    ptask.load_data()
    X, y = ptask.data_concat.values, ptask.labels.loc[ptask.data_concat.index].values
    ptask.init_pipelines_latefusion()

    parallel = ProgressParallel(
        n_jobs=config["parallelization"]["n_jobs_repeats"],
        total=config["latefusion"]["n_repeats"],
    )
    list_shap = parallel(
        delayed(_fun_parallel)(
            ptask,
            X,
            y,
            r,
            disable_infos=(config["parallelization"]["n_jobs_repeats"] is not None)
            and (config["parallelization"]["n_jobs_repeats"] > 1),
        )
        for r in range(config["latefusion"]["n_repeats"])
    )

    shap_explain = {"clinical": [], "radiomics": [], "pathomics": [], "RNA": []}
    coefs_LR = {"clinical": [], "radiomics": [], "pathomics": [], "RNA": []}

    for results in list_shap:
        for moda, shapley in results[0].items():
            shap_explain[moda].append(shapley)

    for key, val in shap_explain.items():
        df_shap = pd.concat(val, axis=0, join="outer")
        df_shap.to_csv(os.path.join(save_dir, "Shap_" + key + ".csv"))

    if config["classifier"]["type"] == "LR":
        for results in list_shap:
            for moda, coefs in results[1].items():
                coefs_LR[moda].append(coefs)

        for key, val in coefs_LR.items():
            coefficients = np.stack(val, axis=-1)
            np.save(os.path.join(save_dir, "coef_LR_" + key + ".npy"), coefficients)


def _fun_parallel(prediction_task, X, y, r, disable_infos):
    """
    Collect SHAP values for several unimodal classifiers with cross-validation

    Parameters
    ----------
    prediction_task: PredictionTask object

    X: 2D array of shape (n_samples, n_features)
        Concatenation of the different modalities

    y: 1D array of shape (n_samples,)
        Binary outcome

    r: int
        Repeat number

    disable_infos: bool

    Returns
    -------
    shap_dict:

    coefs_dict:
    """

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    late_clf = LateFusionClassifier(
        estimators=prediction_task.late_estimators,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.seed(r)),
        **prediction_task.config["latefusion"]["args"]
    )

    shap_dict = {name: [] for name, *_ in late_clf.estimators}

    if prediction_task.config["classifier"]["type"] == "LR":
        coef_dict = {name: [] for name, *_ in late_clf.estimators}

    for fold_index, (train_index, test_index) in tqdm(
        enumerate(cv.split(np.zeros(len(y)), y)),
        leave=False,
        total=cv.get_n_splits(np.zeros(len(y))),
        disable=disable_infos,
    ):
        X_train, y_train, X_test, y_test = (
            X[train_index, :],
            y[train_index],
            X[test_index, :],
            y[test_index],
        )
        # Fit late fusion on the training set of the fold
        clf = clone(late_clf)
        clf.fit(X_train, y_train)
        # Collect SHAP values on the test set of the fold for each unimodal classifier
        for ind, (name, estim, features) in enumerate(clf.fitted_estimators_):
            X_background = X_train[:, features]
            bool_mask = ~(
                np.sum(np.isnan(X_background), axis=1)
                > clf.missing_threshold * len(features)
            )
            X_explain = X_test[:, features]
            bool_mask_explain = ~(
                np.sum(np.isnan(X_explain), axis=1)
                > clf.missing_threshold * len(features)
            )
            if clf.calibration is not None:
                explainer = shap.Explainer(
                    lambda x: (
                        clf.fitted_meta_estimators_[(ind,)].predict_proba(
                            estim.predict_proba(x)[:, 1].reshape(-1, 1)
                        )
                    ),
                    X_background[bool_mask, :],
                )
            else:
                explainer = shap.Explainer(
                    lambda x: estim.predict_proba(x), X_background[bool_mask, :]
                )
            shap_values = explainer(X_explain[bool_mask_explain, :])
            shap_df = pd.DataFrame(
                shap_values.values[:, :, 1],
                columns=prediction_task.data_concat.columns[features],
                index=prediction_task.data_concat.index.values[
                    test_index[bool_mask_explain]
                ],
            )
            shap_df["fold_index"] = fold_index
            shap_df["repeat"] = r
            shap_dict[name].append(shap_df)
            # Also collect coefficients for logistic regreression
            if prediction_task.config["classifier"]["type"] == "LR":
                if name == "RNA":
                    temp = np.zeros((1, 40))
                    temp[:, : estim[-1].coef_.shape[1]] = estim[-1].coef_
                    coef_dict[name].append(temp)
                else:
                    coef_dict[name].append(estim[-1].coef_)

    if prediction_task.config["classifier"]["type"] == "LR":
        coefs_dict = {name: np.vstack(value) for name, value in coef_dict.items()}
    else:
        coefs_dict = None

    shap_dict = {
        name: pd.concat(value, axis=0, join="outer")
        for name, value in shap_dict.items()
    }

    return shap_dict, coefs_dict


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Collect Shap")
    args.add_argument(
        "-c",
        "--config",
        type=str,
        help="config file path",
    )
    args.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="save path",
    )
    main(params=args.parse_args())
