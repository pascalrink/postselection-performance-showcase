# Post-Selection Performance Showcase

This repository serves as a comprehensive showcase for statistical rigor and interpretability in Machine Learning. It specifically addresses the selection bias (winner's curse) that occurs when models are selected purely based on data-driven performance metrics.

## Core Methodology: MABT
The central element of this project is the Multiplicity-Adjusted Bootstrap Tilting (MABT) method. When the best model is chosen from a variety of candidates (e.g., through hyperparameter optimization or the comparison of different model families), the observed test performance is often optimistically biased. MABT provides valid lower confidence limits for performance after selection, ensuring that the reported results are statistically sound.

## Project Structure

### 01_xgboost_shap
This module demonstrates the end-to-end workflow for tabular data using the German Credit Dataset.

#### Script Details: xgboost_shap.py
The script implements a robust pipeline designed for regulated environments (such as Finance or Pharma):

1. Data Splitting: Utilizes a three-way split (train/validation/test) to strictly separate the selection process from the final evaluation.
2. Model Family Comparison:
   - Lasso (Logistic Regression with L1-Penalty): As a linear baseline estimator.
   - Random Forest: For ensemble-based bagging.
   - XGBoost: For gradient boosting with second-order Taylor approximation.
3. Top-k Selection: Evaluation of 30 candidate models and selection of the two best models per family for the final performance matrix.
4. MABT Inference: Calculation of the selection-adjusted 95% lower confidence limit for the winning model to obtain a realistic floor for expected performance.
5. SHAP Interpretability:
   - Global: Beeswarm and bar plots to analyze feature importance across the entire dataset.
   - Local: Waterfall plots to explain individual credit decisions (Right to Explanation).
   - Error Analysis: Utilizing SHAP values to deconstruct specific misclassifications.

## Roadmap for Further Modules

### 02_deep_learning

### 03_transformers

### 04_distribution_shift