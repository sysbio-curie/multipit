import argparse
import inspect
import os
import sys

# import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
from tqdm import tqdm

from _init_scripts_tcga import PredictionTask
from _utils import read_yaml, write_yaml, ProgressParallel

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from multipit.multi_model.latefusion import LateFusionClassifier


def main(params):
    """
    Repeated cross-validation experiment for classification with late fusion
    """

    # Uncomment for disabling ConvergenceWarning
    # warnings.simplefilter("ignore")
    # os.environ["PYTHONWARNINGS"] = 'ignore'

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

    # 3. Perform repeated cross-validation
    parallel = ProgressParallel(
        n_jobs=config["parallelization"]["n_jobs_repeats"],
        total=config["latefusion"]["n_repeats"],
    )
    results_parallel = parallel(
        delayed(_fun_repeats)(
            ptask,
            X,
            y,
            r,
            disable_infos=(config["parallelization"]["n_jobs_repeats"] is not None)
            and (config["parallelization"]["n_jobs_repeats"] > 1),
        )
        for r in range(config["latefusion"]["n_repeats"])
    )

    # 4. Save results
    if config["permutation_test"]:
        perm_predictions = np.zeros(
            (
                len(y),
                len(ptask.names),
                config["n_permutations"],
                config["latefusion"]["n_repeats"],
            )
        )
        list_data_preds= []
        for p, res in enumerate(results_parallel):
            list_data_preds.append(res[0])
            perm_predictions[:, :, :, p] = res[1]
        perm_labels = results_parallel[-1][3]

        np.save(os.path.join(save_dir, "permutation_labels.npy"), perm_labels)
        np.save(os.path.join(save_dir, "permutation_predictions.npy"), perm_predictions)
        data_preds = pd.concat(list_data_preds, axis=0)
        data_preds.to_csv(os.path.join(save_dir, "predictions.csv"))
    else:
        list_data_preds = []
        for p, res in enumerate(results_parallel):
            list_data_preds.append(res[0])
        data_preds = pd.concat(list_data_preds, axis=0)
        data_preds.to_csv(os.path.join(save_dir, "predictions.csv"))


def _fun_repeats(prediction_task, X, y, r, disable_infos):
    """
    Train and test a late fusion model for classification with cross-validation

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
    df_pred: pd.DataFrame of shape (n_samples, n_models+3)
        Predictions collected over the test sets of the cross-validation scheme for each multimodal combination

    df_thrs: pd.DataFrame of shape (n_samples, n_models+2), None
        Thresholds that optimize the log-rank test on the training set for each fold and each multimodal combination.

    permut_predictions: 3D array of shape (n_samples, n_models, n_permutations)
        Predictions collected over the test sets of the cross_validation scheme for each multimodal combination and each random permutation of the labels.

    permut_labels: 2D array of shape (n_samples, n_permutations)
        Permuted labels
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True)  # , random_state=np.random.seed(i))
    X_preds = np.zeros((len(y), 3 + len(prediction_task.names)))

    late_clf = LateFusionClassifier(
        estimators=prediction_task.late_estimators,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.seed(r)),
        **prediction_task.config["latefusion"]["args"]
    )

    # 1. Cross-validation scheme
    for fold_index, (train_index, test_index) in tqdm(
        enumerate(cv.split(np.zeros(len(y)), y)),
        leave=False,
        total=cv.get_n_splits(np.zeros(len(y))),
        disable=disable_infos,
    ):
        X_train, y_train, X_test = (
            X[train_index, :],
            y[train_index],
            X[test_index, :],
        )

        # Fit late fusion on the training set of the fold
        clf = clone(late_clf)
        clf.fit(X_train, y_train)
        # Collect predictions on the test set of the fold for each multimodal combination
        for c, idx in enumerate(prediction_task.indices):
            X_preds[test_index, c] = clf.predict_proba(X_test, estim_ind=idx)[:, 1]

        X_preds[test_index, -3] = fold_index

    X_preds[:, -2] = r
    X_preds[:, -1] = y

    df_pred = (
        pd.DataFrame(
            X_preds,
            columns=prediction_task.names + ["fold_index", "repeat", "label"],
            index=prediction_task.data_concat.index,
        )
        .reset_index()
        .rename(columns={"bcr_patient_barcode": "samples"})
        .set_index(["repeat", "samples"])
    )

    # 2. Perform permutation test
    permut_predictions = None
    permut_labels = None
    if prediction_task.config["permutation_test"]:
        permut_labels = np.zeros((len(y), prediction_task.config["n_permutations"]))
        permut_predictions = np.zeros(
            (
                len(y),
                len(prediction_task.names),
                prediction_task.config["n_permutations"],
            )
        )
        for prm in range(prediction_task.config["n_permutations"]):
            X_perm = np.zeros((len(y), len(prediction_task.names)))
            random_state = check_random_state(prm)
            sh_ind = random_state.permutation(len(y))
            yshuffle = np.copy(y)[sh_ind]
            permut_labels[:, prm] = yshuffle
            for fold_index, (train_index, test_index) in tqdm(
                enumerate(cv.split(np.zeros(len(y)), y)),
                leave=False,
                total=cv.get_n_splits(np.zeros(len(y))),
                disable=disable_infos,
            ):
                X_train, yshuffle_train, X_test = (
                    X[train_index, :],
                    yshuffle[train_index],
                    X[test_index, :],
                )
                clf = clone(late_clf)
                clf.fit(X_train, yshuffle_train)

                for c, idx in enumerate(prediction_task.indices):
                    X_perm[test_index, c] = clf.predict_proba(X_test, estim_ind=idx)[
                        :, 1
                    ]
            permut_predictions[:, :, prm] = X_perm
    return df_pred, permut_predictions, permut_labels


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Late fusion")
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
