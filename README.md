# multipit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides a set of Python tools to perform multimodal learning with tabular data. It contains the code used in our study: 

"Integration of clinical, pathological, radiological, and transcriptomic data to predict first-line immunotherapy outcome in metastatic non-small cell lung cancer"

## Installation
### Dependencies
- lifelines (>= 0.27.4)
- matplotlib (>= 3.5.1)
- numpy (>= 1.21.5)
- pandas (= 1.5.3)
- pyyaml (>= 6.0)
- scikit-learn (>= 1.2.0)
- scikit-survival (>= 0.21.0)
- seaborn (=0.13.0)
- shap (>= 0.41.0)
- xgboost (>= 1.7.5)
### Install from source
Clone the repository:
```
git clone https://github.com/ncaptier/multipit.git 
```

## Key features
* **Early and late fusion implementations**: 4 estimators compatible with scikit-learn and scikit-surv to fuse several tabular modalities in a single multimodal model.
  * [`multipit.multi_model.EarlyFusionClassifier`](multipit/multi_model/earlyfusion.py) and [`multipit.multi_model.LateFusionClassifier`](multipit/multi_model/latefusion.py) for binary classification.
  * [`multipit.multi_model.EarlyFusionSurvival`](multipit/multi_model/earlyfusion.py) and [`multipit.multi_model.LateFusionSurvival`](multipit/multi_model/latefusion.py) for survival prediction.
   

* **Scripts to reproduce the experiments of our study**: Scripts to perform late fusion an early fusion of clinical, radiomic, pathomic and transcriptomic features with a repeated cross-validation scheme. Scripts to compute and collect the SHAP values associated with each unimodal predictive model (see [scripts](scripts) folder).
   

* **Plotting functions and notebooks to reproduce the figures of our study**: several functions to plot and compare the performances of different multimodal combinations as well as to display feature importance with SHAP values.
  * [plot_results.ipynb](notebooks/plot_results.ipynb) 
  * [benchmark.ipynb](notebooks/benchmark.ipynb)
  * [plot_shap.ipynb](notebooks/plot_shap.ipynb)

## Deep-multipit

We also provide another Github repository, named [deep-mulitpit](https://github.com/ncaptier/deep-multipit) with a Pytorch implementation of an end-to-end integration strategy with attention weights, inspired by [Vangurie *et al*, 2022](https://www.nature.com/articles/s43018-022-00416-8).

## Run scripts

Modify the configurations in `.yaml` config files (in config/ subfolder) then run the following command in your terminal:

```
python latefusion.py -c config/config_latefusion.yaml -s path/to/results/folder
```

````
python collect_shap_survival.py -c config/config_latefusion_survival.yaml -s path/to/results/folder
````

**Warning:** For Windows OS paths must be written with '\\' or '\\\' separators (instead of '/').

**Note:** In order to modify more deeply the loading of the data or the predictive pipelines, please update the `PredictionTask` class in the file [_init_scripts.py](scripts/_init_scripts.py). 

## Acknowledgements

This repository was created as part of the PhD project of Nicolas Captier in the [Computational Systems Biologie of Cancer group](https://institut-curie.org/team/barillot) and the [ Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.