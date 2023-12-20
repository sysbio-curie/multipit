import inspect
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from multipit.preprocessing import CustomImputer, CustomOmicsImputer
from multipit.utils.custom.survival import CustomRandomForest
from multipit.utils.custom.classifier import CustomXGBoostClassifier


class PredictionTask:
    """
    Load data and define associated predictive pipelines.

    Parameters
    ----------
    config: dict
        Configuration dictionary

    survival: bool
        Specify whether the prediction task is a classification task or a survival prediction task

    integration: {"late", "early"}
        Integration strategy, either late fusion or early fusion.

    Attributes
    ----------
    list_data: list of pandas Dataframes
        List of dataframe associated with the different modalities

    labels: pandas Dataframe
        target values, either binary values when 'survival = False' or time to event and event indicator (2
        columns) when 'survival = True'.

    target_surv: Structured array (sksurv.util.Surv)
        Additional survival data (if 'return_survival = "OS" or "PFS")

    data_concat: pandas DataFrame of shape (n_samples, n_features)
        Concatenation of all the modalities

    dic_modalities: dict
        Dictionary specifying for each modality the indexes of the columns of data_concat dataframe that are associated
        with it (e.g., {"clinical": [0, 1, 2, 3], 'RNA': [4, 5, 6, 7, 8, 9] ...}).

    names: list of str
        List of the names of the different multimodal combinations (e.g., ["clinical", "RNA", "clinical+RNA"])

    indices: list of tuples
        List of tuples of indexes characterizing the different multimodal combinations (e.g.,
        [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]).

    late_estimators: list of estimators (compatible with scikit-learn), None
        List of estimators (i.e., pre-processing steps + predictive model gathered in a Pipeline object) associated
        with each modality. None if integration = "early"

    early_estimator: estimator (compatible with scikit-learn)
        Estimator to fit on the concatenated and processed multimodal data for early fusion strategy. None if
        integration = "late".

    early_transformers: list of transformers (compatible with scikit-learn)
        List of pre-processing pipelines/transformers associated with each modality.
    """

    def __init__(self, config, survival=False, integration="late"):
        self.config = config
        self.survival = survival
        self.integration = integration

        self.list_data = None
        self.labels = None
        self.target_surv = None
        self.data_concat = None
        self.dic_modalities = None
        self.names = None
        self.indices = None
        self.late_estimators = None
        self.early_estimator = None
        self.early_transformers = None

    def _check_loaded_data(self):
        if self.list_data is None:
            raise NotLoadedError("Data should be loaded first before calling init_pipelines method")

    def _check_integration_strategy(self, strategy):
        if self.integration != strategy:
            raise StrategyError("Wrong init_pipelines method was called for the specified integration strategy ("
                                "integration = " + self.integration + ")")

    def load_data(self):
        """
        Load multimodal data
        """
        # 1. Load TCGA data
        df_clinical = pd.read_csv(self.config["clinical_data"]["clinical_file"], sep=",", index_col=0)
        df_clinical = df_clinical[~df_clinical[["OS.time", "OS"]].isnull().any(axis=1)]
        df_rna = pd.read_csv(self.config["RNA_data"]["RNA_file"], sep=",", index_col=0).loc[df_clinical.index]
        if self.survival:
            self.list_data = [df_rna, df_clinical.drop(columns=["OS.time", "OS"])]
            self.labels = df_clinical[["OS.time", "OS"]].rename(columns={"OS.time": "time", "OS": "event"})
            self.target_surv = None
        else:
            bool_mask = (df_clinical["OS.time"] <= 365) & (df_clinical["OS"] == 0)
            self.list_data = [df_rna[~bool_mask], df_clinical[~bool_mask].drop(columns=["OS.time", "OS"])]
            self.labels = 1*(df_clinical[~bool_mask]["OS.time"] <= 365)
            self.target_surv = None

        # 2. Concat data and save the columns indexes associated with each modality
        self.data_concat = pd.concat(self.list_data, axis=1, join="outer")

        c, o = (self.list_data[0].shape[1], self.list_data[1].shape[1])

        self.dic_modalities = {
            "clinical": np.arange(0, c),
            "RNA": np.arange(-o, 0),
        }

        # 4. Define all possible models (i.e. multimodal combinations)
        models = ["clinical",  "RNA"]
        self.names, self.indices = [], []
        for i in range(1, 3):
            for comb in combinations(range(2), i):
                self.indices.append(comb)
                self.names.append("+".join([models[c] for c in comb]))

        return self

    def init_pipelines_latefusion(self):
        """
        Intialize prediction pipelines for each modality for late fusion strategy
        """
        self._check_integration_strategy(strategy="late")
        self._check_loaded_data()

        self.late_estimators = []
        model = self.config["survival_model"] if self.survival else self.config["classifier"]

        # 1. Define base model (classifier or survival model) common to each modality
        if (not self.survival) and (model["type"] == "xgboost"):
            base_clf = Pipeline(
                steps=[("xgboost", CustomXGBoostClassifier(**model["args"]))]
            )
        elif (not self.survival) and (model["type"] == "LR"):
            base_clf = Pipeline(
                steps=[
                    ("scaler", RobustScaler()),
                    ("imputer", CustomImputer()),
                    ("LR", LogisticRegression(**model["args"])),
                ]
            )
        elif self.survival and (model["type"] == "RF"):
            base_clf = Pipeline(
                steps=[
                    ("imputer", CustomImputer()),
                    ("RF", RandomSurvivalForest(**model["args"])),
                ]
            )
        elif self.survival and (model["type"] == "Cox"):
            base_clf = Pipeline(
                steps=[
                    ("scaler", RobustScaler()),
                    ("imputer", CustomImputer()),
                    ("Cox", CoxnetSurvivalAnalysis(**model["args"])),
                ]
            )
        else:
            raise ValueError(
                "Model can only be of type 'xgboost' or 'LR' for classification and of type 'Cox' or 'RF' for survival"
                " tasks."
            )

        # 2. Define hyperparameter grid for optional tuning
        optim_dict = model["optim_params"]
        if optim_dict is None:
            optim_dict = {}
        elif self.config["latefusion"]["args"]["tuning"] == "randomsearch":
            optim_dict = (model["n_iter_randomcv"], optim_dict)

        # 3. Define preprocessing operations for each modality (e.g. imputation)
        for moda, features in self.dic_modalities.items():
            dct_imput = {}
            if ((not self.survival) and (model["type"] == "LR")) or (
                self.survival and (model["type"] in ["Cox", "RF"])
            ):
                dct_imput = {
                    "__".join(("imputer", key)): value
                    for key, value in self.config[moda + "_data"]["imputation"].items()
                }

            estim = clone(base_clf).set_params(**dct_imput)

            if self.survival:
                estim = as_concordance_index_ipcw_scorer(estim)

            self.late_estimators.append((moda, estim, features, optim_dict))
        return self

    def init_pipelines_earlyfusion(self):
        """
        Initialize the predictive pipeline and the unimodal pre-processing pipelines for early fusion strategy
        """
        self._check_integration_strategy(strategy="early")
        self._check_loaded_data()

        model = self.config["survival_model"] if self.survival else self.config["classifier"]

        # 1. Define model (classifier or survival model) that will be fitted on the concatenated data
        if (not self.survival) and (model["type"] == "xgboost"):
            self.early_estimator = (
                CustomXGBoostClassifier(**model["args"])
                if len(self.config["classifier"]["args"]) > 0
                else CustomXGBoostClassifier()
            )
        elif (not self.survival) and (model["type"] == "LR"):
            self.early_estimator = Pipeline(
                steps=[
                    ("final_imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("LR", LogisticRegression(**model["args"])),
                ]
            )
        elif self.survival and (model["type"] == "RF"):
            self.early_estimator = Pipeline(
                steps=[("RF", CustomRandomForest(**model["args"]))]
            )
        elif self.survival and (model["type"] == "Cox"):
            self.early_estimator = Pipeline(
                steps=[
                    ("final_imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("Cox", CoxnetSurvivalAnalysis(**model["args"])),
                ]
            )

        # 2. Define pre-processing operation for each modality
        self.early_transformers = {moda: None for moda in self.dic_modalities.keys()}
        if (not self.survival) and (model["type"] == "xgboost"):
            self.early_transformers["RNA"] = CustomOmicsImputer(site_feature=-1)
        else:
            for moda in self.dic_modalities.keys():
                if moda == "RNA":
                    self.early_transformers["RNA"] = Pipeline(
                        steps=[
                            ("omics_process", CustomOmicsImputer(site_feature=-1)),
                            ("scaler", RobustScaler()),
                            (
                                "omics_impute",
                                CustomImputer(**self.config["RNA_data"]["imputation"]),
                            ),
                        ]
                    )
                else:
                    self.early_transformers[moda] = Pipeline(
                        steps=[
                            ("scaler", RobustScaler()),
                            (
                                "imputer",
                                CustomImputer(
                                    **self.config[moda + "_data"]["imputation"]
                                ),
                            ),
                        ]
                    )
        return self


class NotLoadedError(ValueError, AttributeError):
    """
    Exception class to raise if data are not loaded.
    """


class StrategyError(ValueError, AttributeError):
    """
    Exception class to raise if the wrong init_pipelines method is called for the specified integration strategy.
    """