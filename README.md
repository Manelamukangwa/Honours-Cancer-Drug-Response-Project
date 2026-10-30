# Honours Cancer Drug Response Project

## Overview

This repository contains the machine learning project developed as part of an Honours research project focused on predicting cancer drug response using pharmacogenomic datasets.

The project investigates the use of machine learning to predict drug response from biological and molecular features of cancer cell lines and drugs, such as gene expression. It further explores generalisation across datasets and training strategies, and benchmarks a custom pipeline against the DrEval (drevalpy) framework.

## Why This Matters

Cancer drug response varies significantly between patients, even for the same tumour type, making treatment selection a persistent clinical challenge. Machine learning models trained on pharmacogenomic data offer a route toward predicting which drugs are likely to work for a given cell line or patient profile, supporting precision oncology.

However, models that perform well on one dataset often fail to generalise to new cell lines, drugs, or datasets; a problem that limits their real-world clinical utility. This project investigates that generalisation gap directly, evaluating how models trained under Leave-Cell-Out splitting strategy can hold up when tested on unseen data, and comparing dataset-specific versus combined-dataset training approaches. 

## Project Structure

```text
Honours-Cancer-Drug-Response-Project/
│
├── Projects/
│   └── Cancer-Drug-Response-Prediction/
│       ├── data/
│       ├── experiments/
│       ├── src/
│
├── README.md
└── .gitignore
```

## Project Status
Completed

## Datasets

The project uses publicly available pharmacogenomic datasets, including:
- **GDSC1 / GDSC2** — Genomics of Drug Sensitivity in Cancer
- **CTRPv1 / CTRPv2** — Cancer Therapeutics Response Portal
- **CCLE** — Cancer Cell Line Encyclopedi

DrEval preprocesses versions of these datasets and obtained via Zenodo.

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
- Naive Mean Effector Predictor — baseline machine learning model

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
