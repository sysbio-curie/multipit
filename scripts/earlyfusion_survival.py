import argparse
import inspect
import os
import sys
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from tqdm import tqdm

from utils import (
    read_yaml,
    write_yaml,
    ProgressParallel,
    encode_biopsy_site,
    process_radiomics,
)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from multipit.data.loader import load_TIPIT_multimoda
from multipit.preprocessing import CustomImputer, CustomOmicsImputer
from multipit.multi_model.earlyfusion import EarlyFusionSurvival
from multipit.utils.custom.cv import CensoredKFold
from multipit.utils.custom.survival import CustomRandomForest


def main(params):
    # 0. Read config file and save it in the results
    config = read_yaml(params.config)
    save_name = config["save_name"]
    if save_name is None:
        run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        save_name = "exp_" + run_id

    save_dir = os.path.join(params.save_path, save_name)
    os.mkdir(save_dir)
    write_yaml(config, os.path.join(save_dir, "config.yaml"))

    # 0. fix random seeds for reproducibility
    seed = config["earlyfusion"]["seed"]
    np.random.seed(seed)

    # 2. Load data
    *list_data, labels, gene_names, target_surv = load_TIPIT_multimoda(
        clinical_file=config["clinical_data"]["clinical_file"],
        radiomics_file=config["radiomics_data"]["radiomics_file"],
        pathomics_file=config["pathomics_data"]["pathomics_file"],
        rna_file=config["rna_data"]["rna_file"],
        order=["clinicals", "radiomics", "pathomics", "RNA"],
        outcome=config["target"],
        return_survival=config["target"],
    )
    list_data[-1] = encode_biopsy_site(list_data[-1])
    list_data[1] = process_radiomics(
        list_data[1], config["radiomics_data"]["preprocessing"]["f_log_transform"]
    )

    c, rad, p, o = (
        list_data[0].shape[1],
        list_data[1].shape[1],
        list_data[2].shape[1],
        list_data[3].shape[1],
    )
    dic_modalities = {
        "clinicals": np.arange(0, c),
        "radiomics": np.arange(c, c + rad),
        "pathomics": np.arange(c + rad, c + rad + p),
        "RNA": np.arange(-o, 0),
    }

    data_concat = pd.concat(list_data, axis=1, join="outer")
    X = data_concat.values
    y = Surv().from_arrays(
        event=labels.loc[data_concat.index, "event"].values,
        time=labels.loc[data_concat.index, "time"].values,
    )

    # 3. Define all possible models (i.e. multimodal combinations)
    models = ["clinicals", "radiomics", "pathomics", "RNA"]
    names, indices = [], []
    for i in range(1, 5):
        for comb in combinations(range(4), i):
            indices.append(comb)
            names.append("+".join([models[c] for c in comb]))

    # 4. Build transformers and estimator
    if config["survival_model"]["type"] == "RF":
        estimator = Pipeline(
            steps=[("RF", CustomRandomForest(**config["survival_model"]["args"]))]
        )
    elif config["survival_model"]["type"] == "Cox":
        estimator = Pipeline(
            steps=[
                ("final_imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("Cox", CoxnetSurvivalAnalysis(**config["survival_model"]["args"])),
            ]
        )

    transformers = {
        "radiomics": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("rad_impute", CustomImputer(**config["radiomics_data"]["imputation"])),
            ]
        ),
        "RNA": Pipeline(
            steps=[
                ("omics_process", CustomOmicsImputer(site_feature=-1)),
                ("scaler", RobustScaler()),
                ("omics_impute", CustomImputer(**config["rna_data"]["imputation"])),
            ]
        ),
        "pathomics": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "path_impute",
                    CustomImputer(**config["pathomics_data"]["imputation"]),
                ),
            ]
        ),
        "clinicals": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "clinicals_impute",
                    CustomImputer(**config["clinical_data"]["imputation"]),
                ),
            ]
        ),
    }

    # 5. Define function to apply for each cross-validation scheme
    def _fun_repeats(r, disable_infos):
        cv = CensoredKFold(
            n_splits=10, shuffle=True
        )  # , random_state=np.random.seed(i))
        X_preds = np.zeros((len(y), 4 + len(names)))
        X_thresholds = np.zeros((len(y), 2 + len(names)))
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
            for c, models in enumerate(names):
                t = {model: transformers[model] for model in models.split("+")}
                early_surv = EarlyFusionSurvival(
                    estimator=estimator,
                    transformers=t,
                    modalities={
                        model: dic_modalities[model] for model in models.split("+")
                    },
                    cv=cv_inner,
                    **config["earlyfusion"]["args"]
                )
                if len(models.split("+")) == 1:
                    early_surv.set_params(**{"select_features": False})

                early_surv.fit(X_train, y_train)
                X_preds[test_index, c] = early_surv.predict(X_test)
                X_thresholds[test_index, c] = early_surv.find_logrank_threshold(
                    X_train, y_train
                )
            X_preds[test_index, -4] = fold_index
            X_thresholds[test_index, -2] = fold_index

        X_preds[:, -3] = r
        X_thresholds[:, -1] = r
        X_preds[:, -2] = y["time"]
        X_preds[:, -1] = y["event"]
        df_pred = (
            pd.DataFrame(
                X_preds,
                columns=names + ["fold_index", "repeat", "label.time", "label.event"],
                index=data_concat.index,
            )
            .reset_index()
            .rename(columns={"index": "samples"})
            .set_index(["repeat", "samples"])
        )

        df_thrs = (
            pd.DataFrame(
                X_thresholds,
                columns=names + ["fold_index", "repeat"],
                index=data_concat.index,
            )
            .reset_index()
            .rename(columns={"index": "samples"})
            .set_index(["repeat", "samples"])
        )

        return df_pred, df_thrs

    # 6. Perform repeated cross-validation
    parallel = ProgressParallel(
        n_jobs=config["parallelization"]["n_jobs_repeats"],
        total=config["earlyfusion"]["n_repeats"],
    )
    results_parallel = parallel(
        delayed(_fun_repeats)(
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
    data_thrs = pd.concat(list_data_thrs, axis=0)
    data_thrs.to_csv(os.path.join(save_dir, "thresholds.csv"))


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
