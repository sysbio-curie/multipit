from itertools import combinations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D
from statannotations.Annotator import Annotator


def plot_metrics(results, metrics, models, title=None, ylim=None, y_text=None, plot_all=True, ax=None,
                 pvals=None, pairs=None):
    results = results.copy()[["metric"] + models]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 7))
    else:
        fig = ax.get_figure()

    if isinstance(metrics, list):
        df_plot = results[results["metric"].isin(metrics)].melt(id_vars=["metric"])
        sns.barplot(data=df_plot, x="variable", y="value", hue="metric", hue_order=metrics, ax=ax, ci=None)

        for i, m in enumerate(metrics):
            ax.errorbar(x=np.arange(len(models)) + (-0.4 + (1 + 2 * i) * (0.4 / len(metrics))),
                        y=results[results["metric"] == m].mean(axis=0).values,
                        yerr=results[results["metric"] == m].std(axis=0).values,
                        fmt='none',
                        ecolor='k',
                        capsize=10,
                        elinewidth=1)

        ax.legend(bbox_to_anchor=(1.08, 1.005), fontsize=12)

    elif isinstance(metrics, str):
        df_plot = results[results["metric"] == metrics].melt(id_vars=["metric"])
        sns.barplot(data=df_plot, x="variable", y="value", ax=ax, ci=None, palette="tab20")
        ax.errorbar(x=np.arange(len(models)),
                    y=results[results["metric"] == metrics].mean(axis=0, numeric_only=True).values,
                    yerr=results[results["metric"] == metrics].std(axis=0, numeric_only=True).values,
                    fmt='none',
                    ecolor='k',
                    capsize=10,
                    elinewidth=1)
    else:
        raise ValueError("")

    if y_text is None:
        y_text = 0.85

    if plot_all:
        ax.vlines(3.5, 0, 1, colors='k', linestyles='--')
        ax.text(1.5, y_text, "1 modality", weight='bold', va='bottom', ha='center', fontsize=12)
        ax.vlines(9.5, 0, 1, colors='k', linestyles='--')
        ax.text(6.5, y_text, "2 modalities", weight='bold', va='bottom', ha='center', fontsize=12)
        ax.vlines(13.5, 0, 1, colors='k', linestyles='--')
        ax.text(11.5, y_text, "3 modalities", weight='bold', va='bottom', ha='center', fontsize=12)
        # ax.vlines(22.5, 0.45, 1, colors = 'k', linestyles='--')
        ax.text(14.5, y_text, "4 modalities", weight='bold', va='bottom', ha='center', fontsize=12)
    else:
        ax.vlines(0.5, 0, 1, colors='k', linestyles='--')
        ax.text(0, y_text, "1 modality", weight='bold', va='bottom', ha='center', fontsize=12)
        ax.vlines(3.5, 0, 1, colors='k', linestyles='--')
        ax.text(2, y_text, "2 modalities", weight='bold', va='bottom', ha='center', fontsize=12)
        ax.vlines(6.5, 0, 1, colors='k', linestyles='--')
        ax.text(5, y_text, "3 modalities", weight='bold', va='bottom', ha='center', fontsize=12)
        # ax.vlines(, 0.45, 1, colors = 'k', linestyles='--')
        ax.text(7, y_text, "4 modalities", weight='bold', va='bottom', ha='center', fontsize=12)

    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0.5, 0.87)

    if title is not None:
        ax.set_title(title, fontsize=14)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set(xlabel=None, ylabel=None)
    sns.despine()

    if pvals is not None:
        if pairs == "significant":
            l_pairs, pvalues = [], []
            comb = combinations(models, 2)
            for pair in comb:
                p = pvals.loc[pair[0], pair[1]]
                if p <= 0.05:
                    pvalues.append(p)
                    l_pairs.append(pair)
        elif pairs == "all":
            l_pairs, pvalues = [], []
            comb = combinations(models, 2)
            for pair in comb:
                p = pvals.loc[pair[0], pair[1]]
                pvalues.append(p)
                l_pairs.append(pair)
        else:
            l_pairs, pvalues = [], []
            for pair in pairs:
                p = pvals.loc[pair[0], pair[1]]
                pvalues.append(p)
                l_pairs.append(pair)

        if len(l_pairs) > 0:
            annotator = Annotator(ax, l_pairs, data=df_plot, x="variable", y="value")
            annotator.pvalue_format.config(fontsize=12)
            annotator.configure(test_short_name="DeLong", text_format="simple",
                                verbose=0, text_offset=1.5, line_width=1)
            annotator.set_pvalues_and_annotate(pvalues)

    plt.tight_layout()
    plt.show()
    return fig


def plot_rankings(results, models, metrics, plot_all=True, ax=None, title=None):
    # rankings = np.argsort(np.argsort(results[['metric'] + models].iloc[:, 1:]))
    rankings = results[['metric'] + models].iloc[:, 1:].rank(ascending=False, axis=1)
    rankings["metric"] = results["metric"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 10))
    else:
        fig = ax.get_figure()

    df_plot = rankings[rankings["metric"] == metrics].melt(id_vars=["metric"])
    sns.boxplot(data=df_plot[df_plot["metric"] == metrics], y="variable", x="value", ax=ax, orient="h", palette="tab20",
                medianprops={"linewidth": 4, "solid_capstyle": "butt"})

    if plot_all:
        ax.hlines(3.5, -5, 25, colors='k', linestyles='--')
        ax.text(15.5, 1.5, "1 modality", weight='bold', va='center', ha='left', fontsize=12)
        ax.hlines(9.5, -5, 25, colors='k', linestyles='--')
        ax.text(15.5, 6.5, "2 modalities", weight='bold', va='center', ha='left', fontsize=12)
        ax.hlines(13.5, -5, 25, colors='k', linestyles='--')
        ax.text(15.5, 11.5, "3 modalities", weight='bold', va='center', ha='left', fontsize=12)
        ax.vlines(22.5, 0.45, 1, colors='k', linestyles='--')
        ax.text(15.5, 14, "4 modalities", weight='bold', va='center', ha='left', fontsize=12)
        ax.set_xlim(0.5, 19)
        ax.set(xlabel=None, ylabel=None)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.set_xticks(list(range(1, 16)))
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
    else:
        ax.hlines(0.5, -5, 25, colors='k', linestyles='--')
        ax.text(8.5, 0, "1 modality", weight='bold', va='bottom', ha='left', fontsize=12)
        ax.hlines(3.5, -5, 25, colors='k', linestyles='--')
        ax.text(8.5, 2, "2 modalities", weight='bold', va='bottom', ha='left', fontsize=12)
        ax.hlines(6.5, -5, 25, colors='k', linestyles='--')
        ax.text(8.5, 5, "3 modalities", weight='bold', va='bottom', ha='left', fontsize=12)
        ax.vlines(22.5, 0.45, 1, colors='k', linestyles='--')
        ax.text(8.5, 7, "4 modalities", weight='bold', va='bottom', ha='left', fontsize=12)
        ax.set_xlim(0.5, 10.5)
        ax.set(xlabel=None, ylabel=None)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.set_xticks(list(range(1, 9)))
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)

    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(metrics + " ranking", fontsize=14)
    plt.tight_layout()
    return fig


def plot_survival(predictions, model, target, target_name="0S", ax=None, title=None, xmax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    stats_pred = predictions.groupby("samples").apply(lambda df: (1 * (df[model] > 0.5)).mean(axis=0))
    target = target.dropna(subset=["time", "event"])  # .loc[stats_pred.index]

    if xmax is not None:
        target.loc[target['time'] > xmax, 'time'] = xmax
        target.loc[target['time'] > xmax, 'event'] = False

    # if xmax is not None:
    #     group1 = (stats_pred.loc[target.index] > 0.5) & (target['time'] <= xmax)
    #     group2 = (stats_pred.loc[target.index] <= 0.5) & (target["time"] <= xmax)
    # else:
    #     group1 = stats_pred.loc[target.index] > 0.5
    #     group2 = stats_pred.loc[target.index] <= 0.5

    group1 = stats_pred.loc[target.index] > 0.5
    group2 = stats_pred.loc[target.index] <= 0.5

    kmf1 = KaplanMeierFitter(label="Predicted " + target_name + " 1 ")  # (n = " + str(group.sum()) + ")")
    kmf1.fit(target[group1]['time'].dropna(), target[group1]['event'].dropna())
    kmf1.plot(ax=ax, show_censors=True)
    kmf2 = KaplanMeierFitter(label="Predicted " + target_name + " 0 ")  # (n = " + str((~group).sum()) + ")")
    kmf2.fit(target[group2]['time'].dropna(), target[group2]['event'].dropna())
    kmf2.plot(ax=ax, show_censors=True)
    add_at_risk_counts(kmf1, kmf2, ax=ax)

    test = logrank_test(durations_A=target[group1]['time'].dropna(),
                        durations_B=target[group2]['time'].dropna(),
                        event_observed_A=1 * (target[group1]['event'].dropna()),
                        event_observed_B=1 * (target[group2]['event'].dropna()),
                        )
    pval = test.summary['p'].values[0]
    leg = Line2D([0], [0], marker='o', color='w', markerfacecolor="black", label='pvalue = ' + str(np.round(pval, 10)),
                 markersize=7)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(leg)
    ax.legend(handles=handles, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_axisbelow(True)
    sns.despine()
    ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.set(xlabel="days", ylabel=None)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_xlabel("days", fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    return fig, group1


def plot_shap_values(data, shap_values, n_best=10, figsize=(9, 10)):
    data_quantiles = data.apply(lambda col: pd.qcut(col.rank(method="first"), q=20, labels=False), axis=0)

    # mask_reduced = np.any(shap_values != 0, axis=1)
    # shap_values_reduced = shap_values[mask_reduced]
    # print(shap_values_reduced.shape)
    # quantiles_reduced = data_quantiles[mask_reduced]

    best_features = np.abs(shap_values).mean(axis=0).sort_values(ascending=False).index[:n_best]

    temp = shap_values[best_features].melt()
    temp["quantiles"] = data_quantiles[best_features].melt()["value"]

    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(data=temp, x="value", y="variable", hue="quantiles", ax=ax, orient="h", s=5, palette="bwr")

    ax.vlines(0, -1, n_best, colors='gray')
    ax.set_ylim(-1, n_best)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.legend().set_visible(False)
    ax.set(xlabel=None, ylabel=None)
    sns.despine(left=True)
    cb = fig.colorbar(cm.ScalarMappable(cmap='bwr'), ticks=[0, 1], aspect=80, ax=ax, shrink=0.9)
    cb.set_ticklabels(['low', 'high'])
    cb.set_label("Feature value", size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    ax.set_xlim(-0.125, 0.125)
    plt.tight_layout()
    plt.show()
    return fig
