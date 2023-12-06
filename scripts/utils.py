import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel


def encode_biopsy_site(df_rna):
    d = {}
    for site in df_rna["Biopsy site"].unique():
        if pd.isnull(site) | (site == "Non disponible"):
            d[site] = np.nan
        elif site in ["PRIMITIF", "META_PULM", "META_PULM_HL", "META_PULM_CL"]:
            d[site] = 0
        elif site in ["META_PLEVRE", "META_PLEVRE_HL", "META_PLEVRE_CL"]:
            d[site] = 1
        elif site.split('_')[0] == 'ADP':
            d[site] = 2
        elif site == 'META_OS':
            d[site] = 3
        elif site == 'META_FOIE':
            d[site] = 4
        elif site == 'META_SURRENALE':
            d[site] = 5
        elif site == 'META_BRAIN':
            d[site] = 6
        else:
            d[site] = 7
    return df_rna.replace({"Biopsy site": d})


def process_radiomics(df_rad, transformed_features):
    df_rad[transformed_features] = np.log(df_rad[transformed_features] + 1)
    return df_rad


def write_yaml(content, fname):
    content = _clean_nested_dict(content)
    with open(fname, "w") as yaml_file:
        yaml.safe_dump(
            content, yaml_file, default_flow_style=None
        )  # , default_flow_style=False)


def read_yaml(fname):
    with open(fname) as yaml_file:
        return yaml.safe_load(yaml_file)


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def _clean_nested_dict(d):
    if isinstance(d, (np.int8, np.int16, np.int32, np.int64)):
        return int(d)
    if isinstance(d, (np.float16, np.float32, np.float64)):
        return float(d)
    if isinstance(d, list):
        return [_clean_nested_dict(x) for x in d]
    if isinstance(d, dict):
        for key, value in d.items():
            d.update({key: _clean_nested_dict(value)})
    return d