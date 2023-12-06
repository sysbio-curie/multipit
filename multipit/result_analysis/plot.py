import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D


def plot_metrics(
    results,
    metrics,
    models=None,
    annotations=None,
    title=None,
    ylim=None,
    y_text=None,
    ax=None,
):
    """
    Plot the results of the repeated cross-validation experiments for different models with a barplot.

    Parameters
    ----------
    results: pandas DataFrame of shape (n_metrics * n_repeats, n_models+1)
        Dataframe containing the different metrics estimated for every cross-validation repeat.

            metric   | model 1 | ... | model_k |
        0 | AUC      |  0.7    | ... | 0.65    |
        1 | accuracy |  0.63   | ... | O.59    |
        2 | AUC      |  0.71   | ... | 0.69    |
        3 | accuracy |  0.61   | ... | O.64    |
           .................................

    metrics: str, list of str
        Metrics to plot. If metrics is a list the different metrics will be plotted side by side for each model.

    models: list of str, None.
        List of model names to plot. If None all the models are displayed. The default is None.

    annotations: dict, None.
        Dictionnary containing annotations to add to subgroups of models/bars. If None, no annotation is added.
        annotations = {"annotation": (index of first bar, index of last bar), ...}. The default is None.

        {"1 modality": (0, 3), "2 modalities": (4, 9), "3 modalities": (10, 13), "4 modalities": (14, 14)}

    title: str, None.
        Title of the plot. If None, no title is displayed. The default is None.

    ylim: Tuple of float > 0, None.
        Bottom and top ylim. If None, it is set to (0.5, 1). The default is None.

    y_text: float > 0, None
        y position of the annotation. If None, it is set to 0.85. Ignored if annotations is None. The default is None.

    ax : matplotlib.axes, None
            The default is None.

    Returns
    -------
    matplotlib.pyplot.figure

    """
    results = results.copy()[["metric"] + models]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 7))
    else:
        fig = ax.get_figure()

    if isinstance(metrics, list):
        df_plot = results[results["metric"].isin(metrics)].melt(id_vars=["metric"])
        sns.barplot(
            data=df_plot,
            x="variable",
            y="value",
            hue="metric",
            hue_order=metrics,
            ax=ax,
            errorbar=None,
        )

        for i, m in enumerate(metrics):
            ax.errorbar(
                x=np.arange(len(models)) + (-0.4 + (1 + 2 * i) * (0.4 / len(metrics))),
                y=results[results["metric"] == m].mean(axis=0).values,
                yerr=results[results["metric"] == m].std(axis=0).values,
                fmt="none",
                ecolor="k",
                capsize=10,
                elinewidth=1,
            )
        ax.legend(bbox_to_anchor=(1.08, 1.005), fontsize=12)

    elif isinstance(metrics, str):
        df_plot = results[results["metric"] == metrics].melt(id_vars=["metric"])
        sns.barplot(
            data=df_plot, x="variable", y="value", hue="variable", legend=False, ax=ax, errorbar=None, palette="tab20"
        )
        ax.errorbar(
            x=np.arange(len(models)),
            y=results[results["metric"] == metrics]
            .mean(axis=0, numeric_only=True)
            .values,
            yerr=results[results["metric"] == metrics]
            .std(axis=0, numeric_only=True)
            .values,
            fmt="none",
            ecolor="k",
            capsize=10,
            elinewidth=1,
        )
    else:
        raise ValueError("")

    if y_text is None:
        y_text = 0.85

    if annotations is not None:
        last_key = list(annotations.keys())[-1]
        for annot, element in annotations.items():
            ax.text(
                element[0] + 0.5 * (element[1] - element[0]),
                y_text,
                annot,
                weight="bold",
                va="bottom",
                ha="center",
                fontsize=12,
            )
            if annot != last_key:
                ax.vlines(element[1] + 0.5, 0, 1, colors="k", linestyles="--")

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0.5, 1.0)

    if title is not None:
        ax.set_title(title, fontsize=14)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.set(xlabel=None, ylabel=None)
    sns.despine()

    # if pvals is not None:
    #     if pairs == "significant":
    #         l_pairs, pvalues = [], []
    #         comb = combinations(models, 2)
    #         for pair in comb:
    #             p = pvals.loc[pair[0], pair[1]]
    #             if p <= 0.05:
    #                 pvalues.append(p)
    #                 l_pairs.append(pair)
    #     elif pairs == "all":
    #         l_pairs, pvalues = [], []
    #         comb = combinations(models, 2)
    #         for pair in comb:
    #             p = pvals.loc[pair[0], pair[1]]
    #             pvalues.append(p)
    #             l_pairs.append(pair)
    #     else:
    #         l_pairs, pvalues = [], []
    #         for pair in pairs:
    #             p = pvals.loc[pair[0], pair[1]]
    #             pvalues.append(p)
    #             l_pairs.append(pair)
    #
    #     if len(l_pairs) > 0:
    #         annotator = Annotator(ax, l_pairs, data=df_plot, x="variable", y="value")
    #         annotator.pvalue_format.config(fontsize=12)
    #         annotator.configure(test_short_name="DeLong", text_format="simple",
    #                             verbose=0, text_offset=1.5, line_width=1)
    #         annotator.set_pvalues_and_annotate(pvalues)

    plt.tight_layout()
    plt.show()
    return fig


def plot_survival(
    predictions, model, target, target_name="0S", ax=None, title=None, xmax=None
):
    """
    Plot survival curves for patients stratified with respect to the predictions of a model (collected with a repeated
    cross-validation scheme).

    Parameters
    ----------
    predictions: pandas DataFrame of shape (n_samples*n_repeats, n_models + 1)
        Dataframe containing the predictions of different models for each sample and for each repeat of the
        cross-validation scheme.

              | samples | model_1 | ... | model_k | repeats |
          0   | name_1  | O.3     | ... | 0.5     |    0    |
                                 ...
          N-1 | name_N  | O.2     | ... | 0.9     |    0    |
          N   | name_1  | O.7     | ... | 0.1     |    1    |
                                 ...

    model: str
        model name.

    target: pandas DataFrame of shape (n_samples, 2)
        Dataframe containing the event indicator and observed time for each sample

               | event | time |
        name_1 |  0    | 1034 |
        name_2 |  1    |  239 |
                    ...

    target_name: str.
        Target name.

    ax : matplotlib.axes, None
        The default is None.

    title: str, None.
        Title of the plot. If None, no title is displayed. The default is None

    xmax: float, None.
        Maximum time to consider. The default is None.

    Returns
    -------
    matplotlib.pyplot.figure
        Generated figure displaying survival curves.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    stats_pred = predictions.groupby("samples").apply(
        lambda df: (1 * (df[model] > 0.5)).mean(axis=0)
    )
    target = target.dropna(subset=["time", "event"])  # .loc[stats_pred.index]

    if xmax is not None:
        target.loc[target["time"] > xmax, "time"] = xmax
        target.loc[target["time"] > xmax, "event"] = False

    # if xmax is not None:
    #     group1 = (stats_pred.loc[target.index] > 0.5) & (target['time'] <= xmax)
    #     group2 = (stats_pred.loc[target.index] <= 0.5) & (target["time"] <= xmax)
    # else:
    #     group1 = stats_pred.loc[target.index] > 0.5
    #     group2 = stats_pred.loc[target.index] <= 0.5

    group1 = stats_pred.loc[target.index] > 0.5
    group2 = stats_pred.loc[target.index] <= 0.5

    kmf1 = KaplanMeierFitter(
        label="Predicted " + target_name + " 1 "
    )  # (n = " + str(group.sum()) + ")")
    kmf1.fit(target[group1]["time"].dropna(), target[group1]["event"].dropna())
    kmf1.plot(ax=ax, show_censors=True)
    kmf2 = KaplanMeierFitter(
        label="Predicted " + target_name + " 0 "
    )  # (n = " + str((~group).sum()) + ")")
    kmf2.fit(target[group2]["time"].dropna(), target[group2]["event"].dropna())
    kmf2.plot(ax=ax, show_censors=True)
    add_at_risk_counts(kmf1, kmf2, ax=ax)

    test = logrank_test(
        durations_A=target[group1]["time"].dropna(),
        durations_B=target[group2]["time"].dropna(),
        event_observed_A=1 * (target[group1]["event"].dropna()),
        event_observed_B=1 * (target[group2]["event"].dropna()),
    )
    pval = test.summary["p"].values[0]
    leg = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="black",
        label="pvalue = " + str(np.round(pval, 10)),
        markersize=7,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(leg)
    ax.legend(handles=handles, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_axisbelow(True)
    sns.despine()
    ax.yaxis.grid(color="gray", linestyle="dashed")
    # ax.set(xlabel="days", ylabel=None)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_xlabel("days", fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    return fig


def plot_shap_values(data, shap_values, n_best=10, figsize=(9, 10)):
    """
    Generate a SHAP values plot for visualizing feature importance.

    Parameters
    ----------
    data : pandas DataFrame
        The input dataset.

    shap_values : numpy array or pandas DataFrame
        SHAP values corresponding to the input data.

    n_best : int, optional
        Number of top features to display. Default is 10.

    figsize : tuple, optional
        Size of the figure (width, height). Default is (9, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure displaying SHAP values.
    """

    data_quantiles = data.apply(
        lambda col: pd.qcut(col.rank(method="first"), q=20, labels=False), axis=0
    )

    # mask_reduced = np.any(shap_values != 0, axis=1)
    # shap_values_reduced = shap_values[mask_reduced]
    # print(shap_values_reduced.shape)
    # quantiles_reduced = data_quantiles[mask_reduced]

    best_features = (
        np.abs(shap_values).mean(axis=0).sort_values(ascending=False).index[:n_best]
    )

    temp = shap_values[best_features].melt()
    temp["quantiles"] = data_quantiles[best_features].melt()["value"]

    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(
        data=temp,
        x="value",
        y="variable",
        hue="quantiles",
        ax=ax,
        orient="h",
        s=5,
        palette="bwr",
    )

    ax.vlines(0, -1, n_best, colors="gray")
    ax.set_ylim(-1, n_best)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.legend().set_visible(False)
    ax.set(xlabel=None, ylabel=None)
    sns.despine(left=True)
    cb = fig.colorbar(
        cm.ScalarMappable(cmap="bwr"), ticks=[0, 1], aspect=80, ax=ax, shrink=0.9
    )
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Feature value", size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    ax.set_xlim(-0.125, 0.125)
    plt.tight_layout()
    plt.show()
    return fig
