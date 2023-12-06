import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


def compute_cindex(df_preds, names, data_train=None):
    def _fun(df, n):
        data_surv = Surv.from_arrays(
            event=df["label.event"].values, time=df["label.time"].values
        )
        if data_train is None:
            data_surv_train = data_surv
        else:
            data_surv_train = data_train
        return (
            df[n]
            .apply(
                lambda col: concordance_index_ipcw(
                    survival_train=data_surv_train,
                    survival_test=data_surv,
                    estimate=col.values,
                )[0],
                axis=0,
            )
            .rename("c_index")
            .to_frame()
            .T
        )

    results = (
        df_preds.groupby("repeat")
        .apply(_fun, n=names)
        .reset_index()
        .rename(columns={"level_1": "metric"})
        .drop(columns="repeat")
    )
    return results


def compute_all_classif(df_preds, names, cindex=False, event=None, time=None):
    results = (
        df_preds.groupby("repeat")
        .apply(_fun_metrics, names=names, cindex=cindex, event=event, time=time)
        .reset_index()
        .rename(columns={"level_1": "metric"})
        .drop(columns="repeat")
    )
    return results


def _fun_metrics(df, names, cindex=False, event=None, time=None):
    df1 = df[names].apply(
        lambda col: pd.Series(
            {
                "sensitivity": sensitivity(df["label"].values, 1 * (col > 0.5).values),
                "specificity": specificity(df["label"].values, 1 * (col > 0.5).values),
                "f1_score": f1_score(df["label"].values, 1 * (col > 0.5).values),
                "balanced_accuracy": balanced_accuracy(
                    df["label"].values, 1 * (col > 0.5).values
                ),
                "mathews_coef": matthews_corrcoef(
                    df["label"].values, 1 * (col > 0.5).values
                ),
                "roc_auc": auc(df["label"].values, col.values),
            }
        ),
        axis=0,
    )

    if cindex:
        df_surv = df[names].copy()
        df_surv["event"] = event
        df_surv["time"] = time
        df_surv = df_surv.dropna(subset=["event", "time"])
        data_surv = Surv.from_arrays(
            event=df_surv["event"].values, time=df_surv["time"].values
        )

        def fun(col):
            return concordance_index_ipcw(
                survival_train=data_surv, survival_test=data_surv, estimate=col.values
            )[0]

        df2 = df_surv[names].apply(fun, axis=0).rename("c_index").to_frame().T
        return pd.concat([df1, df2], axis=0)
    return df1


def sensitivity(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum() / y_true.sum()


def specificity(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum() / (1 - y_true).sum()


def precision(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum() / y_pred.sum()


def f1_score(y_true, y_pred):
    r = sensitivity(y_true, y_pred)
    p = precision(y_true, y_pred)
    return (2 * p * r) / (p + r)


def auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)


def balanced_accuracy(y_true, y_pred):
    sens = sensitivity(y_true, y_pred)
    spe = specificity(y_true, y_pred)
    return 0.5 * (sens + spe)
