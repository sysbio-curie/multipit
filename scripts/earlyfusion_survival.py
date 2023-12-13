import argparse
import inspect
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import delayed
from sksurv.util import Surv
from tqdm import tqdm

from init_scripts import PredictionTask
from utils import read_yaml, write_yaml, ProgressParallel

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from multipit.multi_model.earlyfusion import EarlyFusionSurvival
from multipit.utils.custom.cv import CensoredKFold


def main(params):
    """
    Repeated cross-validation experiment for survival prediction with early fusion
    """

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
    seed = config["earlyfusion"]["seed"]
    np.random.seed(seed)

    # 2. Load data and define pipelines for each modality
    ptask = PredictionTask(config, survival=True, integration="early")
    ptask.load_data()
    X = ptask.data_concat.values
    y = Surv().from_arrays(
        event=ptask.labels.loc[ptask.data_concat.index, "event"].values,
        time=ptask.labels.loc[ptask.data_concat.index, "time"].values,
    )
    ptask.init_pipelines_earlyfusion()

    # 5. Define function to apply for each cross-validation scheme

    # 6. Perform repeated cross-validation
    parallel = ProgressParallel(
        n_jobs=config["parallelization"]["n_jobs_repeats"],
        total=config["earlyfusion"]["n_repeats"],
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
        for r in range(config["earlyfusion"]["n_repeats"])
    )

    # 7. Save results
    list_data_preds, list_data_thrs = [], []
    for res in results_parallel:
        list_data_preds.append(res[0])
        list_data_thrs.append(res[1])
    data_preds = pd.concat(list_data_preds, axis=0)
    data_preds.to_csv(os.path.join(save_dir, "predictions.csv"))
    if config["collect_thresholds"]:
        data_thrs = pd.concat(list_data_thrs, axis=0)
        data_thrs.to_csv(os.path.join(save_dir, "thresholds.csv"))


def _fun_repeats(prediction_task, X, y, r, disable_infos):
    """
    Train and test an early fusion model for survival task with cross-validation

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
    """
    cv = CensoredKFold(n_splits=10, shuffle=True)  # , random_state=np.random.seed(i))
    X_preds = np.zeros((len(y), 4 + len(prediction_task.names)))
    X_thresholds = (
        np.zeros((len(y), 2 + len(prediction_task.names)))
        if prediction_task.config["collect_thresholds"]
        else None
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

        cv_inner = CensoredKFold(
            n_splits=10, shuffle=True, random_state=np.random.seed(r)
        )
        for c, models in enumerate(prediction_task.names):
            t = {
                model: prediction_task.early_transformers[model]
                for model in models.split("+")
            }
            early_surv = EarlyFusionSurvival(
                estimator=prediction_task.early_estimator,
                transformers=t,
                modalities={
                    model: prediction_task.dic_modalities[model]
                    for model in models.split("+")
                },
                cv=cv_inner,
                **prediction_task.config["earlyfusion"]["args"]
            )
            if len(models.split("+")) == 1:
                early_surv.set_params(**{"select_features": False})

            early_surv.fit(X_train, y_train)
            X_preds[test_index, c] = early_surv.predict(X_test)
            if prediction_task.config["collect_thresholds"]:
                X_thresholds[test_index, c] = early_surv.find_logrank_threshold(
                    X_train, y_train
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

    return df_pred, df_thrs


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Early fusion")
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
