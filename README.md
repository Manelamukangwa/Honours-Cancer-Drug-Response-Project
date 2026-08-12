# Honours Cancer Drug Response Project

## Overview

This repository contains the machine learning project developed as part of an Honours research project focused on predicting cancer drug response using pharmacogenomic datasets.

The project investigates the use of machine learning to predict drug response from biological and molecular features of cancer cell lines and drugs, including gene expression, DNA methylation, mutations, proteomics and drug fingerprints. It further explores generalisation across datasets and training strategies, and benchmarks a custom pipeline against the DrEval (drevalpy) framework.

## Why This Matters

Cancer drug response varies significantly between patients, even for the same tumour type, making treatment selection a persistent clinical challenge. Machine learning models trained on pharmacogenomic data offer a route toward predicting which drugs are likely to work for a given cell line or patient profile, supporting precision oncology.

However, models that perform well on one dataset often fail to generalise to new cell lines, drugs, or datasets; a problem that limits their real-world clinical utility. This project investigates that generalisation gap directly, evaluating how models trained under different strategies (including deliberately harder splits like Leave-Cell-Out) hold up when tested on unseen data, and comparing dataset-specific versus combined-dataset training approaches. Generalisation is further tested against patient-derived datasets (BeatAML2, PDX-Bruna), moving beyond cell-line data toward a more clinically relevant test of model robustness.

## Project Structure

```text
Honours-Cancer-Drug-Response-Project/
│
├── Projects/
│   └── Cancer-Drug-Response-Prediction/
│       ├── configs/
│       ├── data/
│       ├── docs/
│       ├── experiments/
│       ├── logs/
│       ├── models/
│       ├── notebooks/
│       ├── results/
│       ├── src/
│       └── tests/
│
├── README.md
└── .gitignore
```

## Project Status

The core `toy_pipeline` (Random Forest baseline) is functional and has completed:
- Data cleaning and feature building (gene expression, drug fingerprints)
- Train/test splitting under three strategies: regular (80/20), Leave-Cell-Out (LOC), and Leave-Drug-Out (LDO) — each evaluated using a single fold rather than repeated k-fold splits
- Model training and evaluation
- A comparison table across all four evaluation strategies


## Datasets

The project draws on publicly available pharmacogenomic datasets, including:
- **GDSC1 / GDSC2** — Genomics of Drug Sensitivity in Cancer
- **CTRPv1 / CTRPv2** — Cancer Therapeutics Response Portal
- **CCLE** — Cancer Cell Line Encyclopedia

For validation and generalisation testing beyond cell-line data, the project also incorporates patient-derived datasets:
- **BeatAML2** — patient-derived acute myeloid leukaemia drug response data
- **PDX-Bruna** — patient-derived xenograft drug response data

DrEval-harmonised versions of these datasets (via Zenodo) are used where available, with raw GDSC2 preprocessing (via CurveCurator) also explored as part of the project.

## Technologies

**Programming language**
- Python

**Libraries**
- pandas — data manipulation and preprocessing
- NumPy — numerical computation and array operations
- Matplotlib — data visualisation and plotting
- Seaborn — statistical data visualisation
- scikit-learn — machine learning models, data splitting, cross-validation, and evaluation

**Models**
- Random Forest — ensemble machine learning model (bagging)
- XGBoost — gradient-boosting machine learning model
- Multiple Linear Regression — baseline machine learning model

**Benchmarking**
- DrEval (drevalpy) — external framework used to benchmark pipeline performance

**Version control**
- Git — version control
- GitHub — repository management, collaboration, and project versioning

## Usage

1. Clone the repository
2. Set up the conda environment (see `configs/` for environment specification)
3. Run pipeline scripts in order from `src/`:
   - Data cleaning
   - Feature building
   - Train/test splitting
   - Model training
   - Cross-validation / evaluation

## Reproducibility

The project is organised using version control and a structured directory system to support reproducible research and maintainable code.

## Author

Manela Mutombo-Mukangwa
Honours Research Project
Supervised by Dr. Hocine Bendou and Dr. Adeshina Odugbemi