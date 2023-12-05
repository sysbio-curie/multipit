import pandas as pd
from sksurv.util import Surv


def load_TIPIT_multimoda(clinical_file,
                         radiomics_file,
                         pathomics_file,
                         rna_file,
                         order,
                         clinical_features=None,
                         radiomic_features=None,
                         pathomics_features=None,
                         rna_features=None,
                         outcome='OS',
                         survival_outcome=False,
                         return_survival=None):
    """
    Loader for raw TIPIT data.

    Parameters
    ----------
    clinical_file: str
        Path to clinical csv file

    radiomics_file: str, None
        Path to radiomics csv file. If None, no radiomics data are loaded.

    pathomics_file: str, None
        Path to pathomics csv file. If None, no pathomics data are loaded.

    rna_file: str, None
        Path to RNA csv file. If None, no RNA data are loaded.

    clinical_features: list of str, None
        List of clinical features to select. If None all the clinical features are considered. The default is None.

    radiomic_features:
        List of radiomics features to select. If None all the radiomics features are considered. The default is None.

    pathomics_features:
        List of pathomics features to select. If None all the pathomics features are considered. The default is None.

    rna_features:
        List of RNA features to select. If None all the RNA features are considered. The default is None.

    order: list of str in {'clinicals', 'radiomics', 'pathomics', 'RNA'}.
        Defines the order of the concatenation (e.g., ["clinicals", "RNA", "pathomics", "radiomics"] modalities will be
        concatenated from left to right).

    outcome: str in {"OS", "PFS", "RECIST"}
        - "OS" corresponds to either death at 1 year (binary outcome) if 'survival_outcome = False' or overall survival
        if 'survival_outcome = True'.
        - "PFS" corresponds to either 6 months progression (binary outcome) if 'survival_outcome = False' or progression
        -free survival if 'survival_outcome = True'.
        - "RECIST" corresponds to stable/progression vs partial response/complete response (binary outcome) and is only
        available when 'survival_outcome = False".
        The default is "OS".

    survival_outcome: bool
        See above. The default is False.

    return_survival: str in {"OS", "PFS"}, None
        If return survival is not None, returns a pandas DataFrame with "time" and "event" columns related to either OS
        or PFS. The default is None.

    Returns
    -------
    output: tuple of pandas DataFrames.
        Dataframes containing the features of the loaded modalities, in the same order as the one specified with 'order'
        parameter.

    target: pandas Dataframe.
        target values, either binary values when 'survival_outcome = False' or time to event and event indicator (2
        columns) when 'survival_outcome = True'.

    target_survival: Structured array (sksurv.util.Surv), None
        Additional survival data (if 'return_survival = "OS" or "PFS")

    """
    # 1. Load raw data and concatenate them
    assert clinical_file is not None, "clinical data should always be provided"
    df_clinicals = pd.read_csv(clinical_file, index_col=0, sep=';')
    df_radiomics = pd.read_csv(radiomics_file, index_col=0, sep=';') if radiomics_file is not None else None
    df_pathomics = pd.read_csv(pathomics_file, index_col=0, sep=';') if pathomics_file is not None else None
    df_RNA = pd.read_csv(rna_file, index_col=0, sep=";") if rna_file is not None else None

    list_data = [df for df in [df_clinicals, df_pathomics, df_RNA, df_radiomics] if df is not None]
    df_total = pd.concat(list_data, axis=1, join='outer') if len(list_data) > 1 else list_data[0].copy()

    # 2. Collect outcome/target (either OS, PFS or Best Response (for classification))
    if survival_outcome:
        if outcome == 'OS':
            bool_mask = df_total['OS'].isnull()
            df_total = df_total[~bool_mask]
            target = df_total[['OS', 'Statut Vital']].rename(columns={"OS": "time", "Statut Vital": "event"})
            target["event"] = 1*(target["event"] == "Decede")
        elif outcome == 'PFS':
            bool_mask = df_total['PFS'].isnull()
            df_total = df_total[~bool_mask]
            target = df_total[['PFS', 'Progression']].rename(columns={"PFS": "time", "Progression": "event"})
            target["event"] = 1*(target["event"] == 'Yes')
        else:
            raise ValueError("outcome can only be 'OS' or 'PFS' when survival_outcome is True")
    else:
        if outcome == 'OS':
            bool_mask = (df_total['OS'].isnull()) | ((df_total['OS'] <= 365) & (df_total['Statut Vital'] == "Vivant"))
            df_total = df_total[~bool_mask]
            target = 1 * (df_total['OS'] <= 365)

        elif outcome == 'PFS':
            bool_mask = (df_total['PFS'].isnull()) | ((df_total['PFS'] <= 180) & (df_total['Progression'] == "No"))
            df_total = df_total[~bool_mask]
            target = (1 * (df_total['PFS'] <= 180))

        elif outcome == "RECIST":
            bool_mask = df_total['Best response'].isnull()
            df_total = df_total[~bool_mask]
            target = 1 * ((df_total['Best response'] == 'Partielle') | (df_total['Best response'] == 'Complete'))

        else:
            raise ValueError("outcome can only be 'OS','PFS' or 'RECIST' when survival_outcome is False")

    # 3. Select specific features for each modality
    datasets = {key: None for key in ['clinicals', 'radiomics', 'pathomics', 'RNA']}
    target_survival = None

    if clinical_features is not None:
        datasets['clinicals'] = df_total[clinical_features]
    else:
        datasets['clinicals'] = df_total[df_clinicals.columns].drop(columns=['OS', 'PFS', 'Statut Vital',
                                                                             'Progression', 'Best response'],
                                                                    errors='ignore')
    if return_survival == "PFS":
        target_survival = Surv().from_arrays(time=df_total["PFS"].values,
                                             event=(1*(df_total["Progression"] == "Yes")).values)
    elif return_survival == "OS":
        target_survival = Surv().from_arrays(time=df_total["OS"].values,
                                             event=(1*(df_total["Statut Vital"] == "Decede")).values)

    if df_radiomics is not None:
        datasets['radiomics'] = df_total[radiomic_features] if radiomic_features is not None \
            else df_total[df_radiomics.columns]
    if df_pathomics is not None:
        datasets['pathomics'] = df_total[pathomics_features] if pathomics_features is not None \
            else df_total[df_pathomics.columns]
    if df_RNA is not None:
        datasets['RNA'] = df_total[rna_features] if rna_features is not None else df_total[df_RNA.columns]

    # 4. Return each dataset and the target in the right order
    output = tuple()
    for modality in order:
        assert datasets[modality] is not None, "order specifies a modality but the input file for loading the raw " \
                                               "data is not given "
        output = output + (datasets[modality],)

    return output + (target, target_survival)
