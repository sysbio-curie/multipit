import argparse
import inspect
import os
import sys
# import warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sksurv.util import Surv
from tqdm import tqdm

from utils import read_yaml, write_yaml, ProgressParallel, encode_biopsy_site, process_radiomics

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from multipit.data.loader import load_TIPIT_multimoda
from multipit.preprocessing import CustomImputer, CustomOmicsImputer
from multipit.multi_model.latefusion import LateFusionSurvival
from multipit.utils.custom.cv import CensoredKFold


def main(params):
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

    # 0. fix random seeds for reproducibility
    seed = config["latefusion"]["seed"]
    np.random.seed(seed)

    # 2. Load data
    *list_data, labels, gene_names, target_surv = load_TIPIT_multimoda(
        clinical_file=config["clinical_data"]["clinical_file"],
        radiomics_file=config["radiomics_data"]["radiomics_file"],
        pathomics_file=config["pathomics_data"]["pathomics_file"],
        rna_file=config["rna_data"]["rna_file"],
        order=['clinicals', 'radiomics', 'pathomics', 'RNA'],
        outcome=config["target"],
        return_survival=config["target"],
    )
    list_data[-1] = encode_biopsy_site(list_data[-1])
    list_data[1] = process_radiomics(list_data[1], config["radiomics_data"]["preprocessing"]["f_log_transform"])

    data_concat = pd.concat(list_data, axis=1, join='outer')
    X = data_concat.values
    y = Surv().from_arrays(event=labels.loc[data_concat.index, 'event'].values,
                           time=labels.loc[data_concat.index, 'time'].values)

    # 3. Define all possible models (i.e. multimodal combinations)
    models = ['clinicals', 'radiomics', 'pathomics', 'RNA']
    names, indices = [], []
    for i in range(1, 5):
        for comb in combinations(range(4), i):
            indices.append(comb)
            names.append("+".join([models[c] for c in comb]))

    # 4. Build pipelines

    ### 4.1 Define survival model
    if config["survival_model"]["type"] == "RF":
        base_clf = Pipeline(steps=[('imputer', CustomImputer()),
                                   ('RF', RandomSurvivalForest(**config["survival_model"]["args"]))])
    elif config["survival_model"]["type"] == "Cox":
        base_clf = Pipeline(steps=[('scaler', RobustScaler()),
                                   ("imputer", CustomImputer()),
                                   ('Cox', CoxnetSurvivalAnalysis(**config["survival_model"]["args"]))])
    else:
        raise ValueError("survival_model can only be of type 'RF' or 'Cox'")

    ### 4.2 Define hyperparameter grid for optional tuning
    optim_dict = config["survival_model"]["optim_params"]
    if optim_dict is None:
        optim_dict = {}
    elif config["latefusion"]["args"]["tuning"] == "randomsearch":
        optim_dict = (config["survival_model"]["n_iter_randomcv"], optim_dict)

    ### 4.3 Define preprocessing operations for each modality (e.g. imputation)
    dct_clin_imput = {'__'.join(('imputer', key)): value
                      for key, value in config["clinical_data"]["imputation"].items()}
    pipe_clinical = ("clinicals",
                     as_concordance_index_ipcw_scorer(
                         clone(base_clf).set_params(**dct_clin_imput)
                     ),
                     list(range(list_data[0].shape[1])),
                     optim_dict
                     )

    dct_rad_imput = {'__'.join(('imputer', key)): value
                     for key, value in config["radiomics_data"]["imputation"].items()}
    pipe_radiomics = ("radiomics",
                      as_concordance_index_ipcw_scorer(
                          clone(base_clf).set_params(**dct_rad_imput)
                      ),
                      list(range(list_data[0].shape[1],
                                 list_data[0].shape[1] + list_data[1].shape[1])),
                      optim_dict
                      )

    dct_pat_imput = {'__'.join(('imputer', key)): value
                     for key, value in config["pathomics_data"]["imputation"].items()}
    pipe_pathomics = ("pathomics",
                      as_concordance_index_ipcw_scorer(
                          clone(base_clf).set_params(**dct_pat_imput)
                      ),
                      list(range(list_data[0].shape[1] + list_data[1].shape[1],
                                 list_data[0].shape[1] + list_data[1].shape[1] + list_data[2].shape[1])),
                      optim_dict
                      )

    dct_rna_imput = {'__'.join(('imputer', key)): value
                     for key, value in config["rna_data"]["imputation"].items()}
    pipe_rna = ("RNA",
                as_concordance_index_ipcw_scorer(
                    Pipeline(steps=[('imputer_omics', CustomOmicsImputer(site_feature=-1))
                                    ] + clone(base_clf).set_params(**dct_rna_imput).steps)
                ),
                list(range(-list_data[-1].shape[1], 0)),
                optim_dict
                )

    # 5. Define function to apply for each cross-validation scheme
    def _fun_repeats(r, disable_infos):
        cv = CensoredKFold(n_splits=10, shuffle=True)  # , random_state=np.random.seed(i))
        X_preds = np.zeros((len(y), 4 + len(names)))
        X_thresholds = np.zeros((len(y), 2 + len(names)))
        late_clf = LateFusionSurvival(estimators=[pipe_clinical, pipe_radiomics, pipe_pathomics, pipe_rna],
                                      cv=CensoredKFold(n_splits=10, shuffle=True, random_state=np.random.seed(r)),
                                      **config["latefusion"]["args"])

        for fold_index, (train_index, test_index) in tqdm(enumerate(cv.split(np.zeros(len(y)), y)),
                                                          leave=False,
                                                          total=cv.get_n_splits(np.zeros(len(y))),
                                                          disable=disable_infos):
            X_train, y_train, X_test, y_test = X[train_index, :], y[train_index], X[test_index, :], y[test_index]

            clf = clone(late_clf)
            clf.fit(X_train, y_train)

            for c, idx in enumerate(indices):
                X_preds[test_index, c] = clf.predict(X_test, estim_ind=idx)
                X_thresholds[test_index, c] = clf.find_logrank_threshold(X_train, y_train, estim_ind=idx)

            X_preds[test_index, -4] = fold_index
            X_thresholds[test_index, -2] = fold_index

        X_preds[:, -3] = r
        X_thresholds[:, -1] = r
        X_preds[:, -2] = y['time']
        X_preds[:, -1] = y['event']
        df_pred = (pd.DataFrame(X_preds,
                                columns=names + ["fold_index", "repeat", "label.time", "label.event"],
                                index=data_concat.index
                                )
                   .reset_index()
                   .rename(columns={"index": "samples"})
                   .set_index(['repeat', 'samples'])
                   )

        df_thrs = (pd.DataFrame(X_thresholds,
                                columns=names + ["fold_index", "repeat"],
                                index=data_concat.index
                                )
                   .reset_index()
                   .rename(columns={"index": "samples"})
                   .set_index(['repeat', 'samples'])
                   )

        permut_predictions = None
        permut_labels = None
        if config["permutation_test"]:
            permut_labels = np.zeros((len(y), config["n_permutations"], 2))
            permut_predictions = np.zeros((len(y), len(names), config["n_permutations"]))
            for prm in range(config["n_permutations"]):
                X_perm = np.zeros((len(y), len(names)))
                random_state = check_random_state(prm)
                sh_ind = random_state.permutation(len(y))
                yshuffle = np.copy(y)[sh_ind]
                permut_labels[:, prm, 0] = yshuffle['time']
                permut_labels[:, prm, 1] = yshuffle['event']
                for fold_index, (train_index, test_index) in tqdm(enumerate(cv.split(np.zeros(len(y)), y)),
                                                                  leave=False,
                                                                  total=cv.get_n_splits(np.zeros(len(y))),
                                                                  disable=disable_infos):
                    X_train, y_train, X_test, y_test = X[train_index, :], yshuffle[train_index], X[test_index, :], \
                                                       yshuffle[test_index]
                    clf = clone(late_clf)
                    clf.fit(X_train, y_train)

                    for c, idx in enumerate(indices):
                        X_perm[test_index, c] = clf.predict(X_test, estim_ind=idx)
                permut_predictions[:, :, prm] = X_perm
        return df_pred, df_thrs, permut_predictions, permut_labels

    # 6. Perform repeated cross-validation
    parallel = ProgressParallel(n_jobs=config["parallelization"]["n_jobs_repeats"],
                                total=config["latefusion"]["n_repeats"])
    results_parallel = parallel(delayed(_fun_repeats)(r,
                                                      disable_infos=
                                                      (config["parallelization"]["n_jobs_repeats"] is not None) and
                                                      (config["parallelization"]["n_jobs_repeats"] > 1))
                                for r in range(config["latefusion"]["n_repeats"]))

    # 7. Save results
    if config["permutation_test"]:
        perm_predictions = np.zeros((len(y), len(names), config["n_permutations"], config["latefusion"]["n_repeats"]))
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
        data_thrs = pd.concat(list_data_thrs, axis=0)
        data_thrs.to_csv(os.path.join(save_dir, "thresholds.csv"))
    else:
        list_data_preds, list_data_thrs = [], []
        for p, res in enumerate(results_parallel):
            list_data_preds.append(res[0])
            list_data_thrs.append(res[1])
        data_preds = pd.concat(list_data_preds, axis=0)
        data_preds.to_csv(os.path.join(save_dir, "predictions.csv"))
        data_thrs = pd.concat(list_data_thrs, axis=0)
        data_thrs.to_csv(os.path.join(save_dir, "thresholds.csv"))


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
