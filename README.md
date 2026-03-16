
# Post-Selection Performance Showcase

This repository serves as a comprehensive showcase for statistical rigor and interpretability in Machine Learning. It specifically addresses the selection bias (winner's curse) that occurs when models are selected purely based on data-driven performance metrics.

## Overview

Modern machine learning workflows typically evaluate many candidate models and select the best-performing one. However, the performance reported for the selected model is often **optimistically biased**, because the same evaluation results were used for model selection.

This repository demonstrates how to obtain **statistically valid performance guarantees after model selection** using the **Multiplicity‑Adjusted Bootstrap Tilting (MABT)** method. When the best model is chosen from a variety of candidates (e.g., through hyperparameter optimization or the comparison of different model families), the observed test performance is often optimistically biased. MABT provides valid lower confidence limits for performance after selection, ensuring that the reported results are statistically sound.

MABT produces **lower confidence bounds for prediction performance that remain valid even after selecting the best model from many candidates**.

The method was developed in my dissertation:

**Pascal Rink (2025)**  
*Confidence Limits for Prediction Performance*  
University of Bremen  
https://doi.org/10.26092/elib/3822

The repository provides **practical implementations of the method across multiple machine learning settings**, including:

- classical machine learning (scikit‑learn)
- gradient boosting (XGBoost)
- deep learning (PyTorch)
- transformer models (Hugging Face)
- evaluation under distribution shift

---

## Key Idea

Typical ML evaluation pipeline:

```
Many candidate models
        │
        ▼
Model selection
        │
        ▼
Naive performance estimate
(optimistically biased)
        │
        ▼
MABT adjustment
        │
        ▼
Valid lower confidence bound
```

MABT corrects for the bias introduced by model selection by performing **simultaneous statistical inference across all candidate models**.

---

## Repository Structure

```
postselection-performance-showcase
│
├── 01_xgboost_shap
│   └── xgboost_shap.py
│
├── 02_deep_learning
│   └── fashion_mnist_models.py
│
├── 03_transformers
│   └── agnews_distilbert.py
│
├── 04_distribution_shift
│   └── ctg_shift.py
│
├── requirements.txt
└── LICENSE
```

Each directory contains a **self‑contained experiment** demonstrating how MABT can be applied in different modelling scenarios.

---

# Example Experiments

## 1. Gradient Boosting with Model Interpretability

**File:** `01_xgboost_shap/xgboost_shap.py`

Dataset  

- German Credit dataset (OpenML)

Models

- Logistic Regression
- Random Forest
- XGBoost

Workflow

- preprocessing using **scikit‑learn pipelines**
- training multiple candidate models
- validation‑based model selection
- evaluation on a held‑out test set
- **MABT lower confidence bound for accuracy**

Additional component

This example also demonstrates **SHAP explanations** for model interpretability, illustrating how **performance guarantees can be combined with explainable AI techniques**.

---

## 2. Deep Learning Model Selection

**File:** `02_deep_learning/fashion_mnist_models.py`

Dataset  

- Fashion‑MNIST image classification dataset

Candidate architectures

- small MLP
- large MLP
- small CNN
- large CNN

Framework

- PyTorch

Workflow

1. train several neural network architectures  
2. evaluate models on validation data  
3. select a shortlist of candidate models  
4. evaluate shortlisted models on test data  
5. apply **MABT to obtain a valid confidence bound**

This example demonstrates that the method integrates naturally with **deep learning pipelines**.

---

## 3. Transformer Models for NLP

**File:** `03_transformers/agnews_distilbert.py`

Dataset  

- AG News text classification dataset

Model

- DistilBERT transformer (Hugging Face Transformers)

Candidate models differ in:

- random seed
- learning rate
- number of epochs
- weight decay
- training data size

Workflow

- fine‑tune several transformer models
- generate different prediction strategies
- select the best strategy
- compute **MABT confidence bounds**

This example demonstrates applicability to **modern NLP architectures**.

---

## 4. Distribution Shift Scenario

**File:** `04_distribution_shift/ctg_shift.py`

Dataset  

- Cardiotocography dataset (UCI ML Repository)

Models

- Logistic Regression
- Random Forest
- XGBoost

Scenario

A **covariate shift** between training and testing data is simulated by splitting the dataset based on a feature threshold.

Workflow

1. train models on one distribution  
2. evaluate models under shifted test distribution  
3. select the best model  
4. compute **MABT lower confidence bounds**

This illustrates how MABT can support **robust model evaluation under distribution shift**.

---

## Technical Highlights

Machine Learning

- classical ML pipelines (scikit‑learn)
- gradient boosting (XGBoost)
- deep learning training (PyTorch)
- transformer fine‑tuning (Hugging Face)

Statistical Methods

- post‑selection inference
- bootstrap methods
- simultaneous inference for model evaluation

Engineering Practices

- reproducible ML experiments
- deterministic training setups
- multi‑framework ML experimentation

---

## Reproducibility

All experiments are designed to be reproducible:

- fixed random seeds
- deterministic data splits
- explicit model configurations

Running the code requires the following Python libraries:

- datasets
- mabt
- matplotlib
- numpy
- pandas
- pathlib
- random
- re
- seaborn
- shap
- sklearn
- torch
- torchvision
- transformers
- ucimlrepo
- xgboost

See requirements.txt for details.

---

## Quick Start

Clone repository

```
git clone https://github.com/pascalrink/postselection-performance-showcase
cd postselection-performance-showcase
```

Install dependencies

```
pip install -r requirements.txt
```

Run an example

```
python 01_xgboost_shap/xgboost_shap.py
```

---

## About the MABT Method

The **Multiplicity‑Adjusted Bootstrap Tilting (MABT)** method provides **valid lower confidence bounds for prediction performance after model selection**.

Key characteristics:

- accounts for model selection
- model‑agnostic
- compatible with different ML frameworks
- requires only predictions on an evaluation set
- no retraining needed for inference

Reference implementation:

https://github.com/pascalrink/prediction-performance-ci

---

## Reference

Pascal Rink (2025)  
**Confidence Limits for Prediction Performance**  
University of Bremen

https://doi.org/10.26092/elib/3822

---

## License

MIT License
