## Collection of TCGA data

1. Bulk RNA-seq profiles from [TGCA-LUAD](https://portal.gdc.cancer.gov/exploration?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.tissue_or_organ_of_origin%22%2C%22value%22%3A%5B%22lower%20lobe%2C%20lung%22%2C%22lung%2C%20nos%22%2C%22main%20bronchus%22%2C%22middle%20lobe%2C%20lung%22%2C%22overlapping%20lesion%20of%20lung%22%2C%22upper%20lobe%2C%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LUAD%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.samples.sample_type%22%2C%22value%22%3A%5B%22primary%20tumor%22%5D%7D%7D%5D%7D)
(i.e., NSCLC adenocarcinoma) and [TGCA-LUSC](https://portal.gdc.cancer.gov/exploration?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.tissue_or_organ_of_origin%22%2C%22value%22%3A%5B%22lower%20lobe%2C%20lung%22%2C%22lung%2C%20nos%22%2C%22main%20bronchus%22%2C%22middle%20lobe%2C%20lung%22%2C%22overlapping%20lesion%20of%20lung%22%2C%22upper%20lobe%2C%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LUSC%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.samples.sample_type%22%2C%22value%22%3A%5B%22primary%20tumor%22%5D%7D%7D%5D%7D)
(i.e., NSCLC squamous cell carcinoma) were extracted with the [GDC data portal](https://portal.gdc.cancer.gov/), normalized with TPM, and log-transformed.
   

2. MCP counter signatures (1) were computed for each sample (i.e., score the abundance of 10 cell types). Additionally, log expressions of 22 oncogenes associated with lung cancer were extracted as features (KRAS, NRAS, EGFR, MET, BRAF, ROS1, ALK, ERBB2, ERBB4, FGFR1, FGFR2, FGFR3, NTRK1, NTRK2, NTRK3, LTK, RET, RIT1, MAP2K1, DDR2, ALK, and CD274).


3. Clinical data, including Overall Survival data, were extracted from Liu *et al.* (2). Categorical features were binary encoded:
   * **female_vs_male** (sex): 0: man, 1: woman
   * **adeno_vs_squamous** (subtype): 0: squamous cell carcinoma, 1: adenocarcinoma
   * **stageIV_vs_stageIII** (stage): 0: stage III, 1: stage IV


4. Only patients with stage III or IV were considered in this analysis.

### References
1. Becht E, Giraldo NA, Lacroix L, Buttard B, Elarouci N, Petitprez F, et al. Estimating the population abundance of tissue-infiltrating immune and stromal cell populations using gene expression. Genome Biol. 2016 Oct 20;17(1):218.
2. Liu J, Lichtenberg T, Hoadley KA, Poisson LM, Lazar AJ, Cherniack AD, et al. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell. 2018 Apr 5;173(2):400-416.e11.