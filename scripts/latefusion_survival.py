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
from sklearn.utils import check_random_state
from sksurv.util import Surv
from tqdm import tqdm

from init_scripts import PredictionTask
from utils import read_yaml, write_yaml, ProgressParallel

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from multipit.multi_model.latefusion import LateFusionSurvival
from multipit.utils.custom.cv import CensoredKFold


def main(params):
    """
    Repeated cross-validation experiment for survival prediction with late fusion
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
    ptask = PredictionTask(config, survival=True, integration="late")
    ptask.load_data()
    X = ptask.data_concat.values
    y = Surv().from_arrays(
        event=ptask.labels.loc[ptask.data_concat.index, "event"].values,
        time=ptask.labels.loc[ptask.data_concat.index, "time"].values,
    )
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
        list_data_preds, list_data_thrs = [], []
        for p, res in enumerate(results_parallel):
            list_data_preds.append(res[0])
            list_data_thrs.append(res[1])
            perm_predictions[:, :, :, p] = res[2]
        perm_labels = results_parallel[-1][3]
        np.save(os.path.join(save_dir, "permutation_labels.npy"), perm_labels)
        np.save(os.path.join(save_dir, "permutation_predictions.npy"), perm_predictions)
        data_preds = pd.concat(list_data_preds, axis=0)
        data_preds.to_csv(os.path.join(save_dir, "predictions.csv"))
        if config["collect_thresholds"]:
            data_thrs = pd.concat(list_data_thrs, axis=0)
            data_thrs.to_csv(os.path.join(save_dir, "thresholds.csv"))
    else:
        list_data_preds, list_data_thrs = [], []
        for p, res in enumerate(results_parallel):
            list_data_preds.append(res[0])
            list_data_thrs.append(res[1])
        data_preds = pd.concat(list_data_preds, axis=0)
        data_preds.to_csv(os.path.join(save_dir, "predictions.csv"))
        if config["collect_thresholds"]:
            data_thrs = pd.concat(list_data_thrs, axis=0)
            data_thrs.to_csv(os.path.join(save_dir, "thresholds.csv"))


def _fun_repeats(prediction_task, X, y, r, disable_infos):
    """
    Train and test a late fusion model for survival task with cross-validation

    Parameters
    ----------
    prediction_task: PredictionTask object

    X: 2D array of shape (n_samples, n_features)
        Concatenation of the different modalities

    y: Structured array of size (n_samples,)
        Event indicator and observed time for each sample

    r: int
        Repeat number

    disable_infos: bool

    Returns
    -------
    df_pred: pd.DataFrame of shape (n_samples, n_models+4)
        Predictions collected over the test sets of the cross-validation scheme for each multimodal combination

    df_thrs: pd.DataFrame of shape (n_samples, n_models+2), None
        Thresholds that optimize the log-rank test on the training set for each fold and each multimodal combination.

    permut_predictions: 3D array of shape (n_samples, n_models, n_permutations)
        Predictions collected over the test sets of the cross_validation scheme for each multimodal combination and each random permutation of the labels.

    permut_labels: 3D array of shape (n_samples, n_permutations, 2)
        Permuted event indicators and observed times.
    """
    cv = CensoredKFold(n_splits=10, shuffle=True)  # , random_state=np.random.seed(i))
    X_preds = np.zeros((len(y), 4 + len(prediction_task.names)))
    X_thresholds = (
        np.zeros((len(y), 2 + len(prediction_task.names)))
        if prediction_task.config["collect_thresholds"]
        else None
    )
    late_clf = LateFusionSurvival(
        estimators=prediction_task.late_estimators,
        cv=CensoredKFold(n_splits=10, shuffle=True, random_state=np.random.seed(r)),
        **prediction_task.config["latefusion"]["args"]
    )

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

        clf = clone(late_clf)
        clf.fit(X_train, y_train)

        for c, idx in enumerate(prediction_task.indices):
            X_preds[test_index, c] = clf.predict(X_test, estim_ind=idx)
            if prediction_task.config["collect_thresholds"]:
                X_thresholds[test_index, c] = clf.find_logrank_threshold(
                    X_train, y_train, estim_ind=idx
                )

        X_preds[test_index, -4] = fold_index
        if prediction_task.config["collect_thresholds"]:
            X_thresholds[test_index, -2] = fold_index

    X_preds[:, -3] = r
    if prediction_task.config["collect_thresholds"]:
        X_thresholds[:, -1] = r
    X_preds[:, -2] = y["time"]
    X_preds[:, -1] = y["event"]
    df_pred = (
        pd.DataFrame(
            X_preds,
            columns=prediction_task.names
            + ["fold_index", "repeat", "label.time", "label.event"],
            index=prediction_task.data_concat.index,
        )
        .reset_index()
        .rename(columns={"index": "samples"})
        .set_index(["repeat", "samples"])
    )

    if prediction_task.config["collect_thresholds"]:
        df_thrs = (
            pd.DataFrame(
                X_thresholds,
                columns=prediction_task.names + ["fold_index", "repeat"],
                index=prediction_task.data_concat.index,
            )
            .reset_index()
            .rename(columns={"index": "samples"})
            .set_index(["repeat", "samples"])
        )
    else:
        df_thrs = None

    permut_predictions = None
    permut_labels = None
    if prediction_task.config["permutation_test"]:
        permut_labels = np.zeros((len(y), prediction_task.config["n_permutations"], 2))
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
            permut_labels[:, prm, 0] = yshuffle["time"]
            permut_labels[:, prm, 1] = yshuffle["event"]
            for fold_index, (train_index, test_index) in tqdm(
                enumerate(cv.split(np.zeros(len(y)), y)),
                leave=False,
                total=cv.get_n_splits(np.zeros(len(y))),
                disable=disable_infos,
            ):
                X_train, y_train, X_test, y_test = (
                    X[train_index, :],
                    yshuffle[train_index],
                    X[test_index, :],
                    yshuffle[test_index],
                )
                clf = clone(late_clf)
                clf.fit(X_train, y_train)

                for c, idx in enumerate(prediction_task.indices):
                    X_perm[test_index, c] = clf.predict(X_test, estim_ind=idx)
            permut_predictions[:, :, prm] = X_perm
    return df_pred, df_thrs, permut_predictions, permut_labels


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
