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


### 02_deep_learning: Deep Learning Example: Fashion-MNIST with Post-Selection Inference

This module demonstrates how **post-selection performance guarantees** can be applied in a deep learning setting. The experiment trains several neural network architectures on the **Fashion-MNIST** dataset and evaluates them using the **MABT (Model-Agnostic Bound Technique)** to obtain a valid lower confidence bound on the performance of the model selected after evaluation.

#### Dataset

The experiment uses the **Fashion-MNIST** dataset, which consists of:

* 70,000 grayscale images of clothing items
* Image size: **28 × 28 pixels**
* **10 classes** (e.g., T-shirt, sneaker, coat)

The data is split into:

* **Training set** – used to fit model parameters
* **Validation set** – used for model selection
* **Test set** – used only for final evaluation and post-selection inference

#### Candidate Models

Several neural network architectures are considered:

* Multiple **MLP (feedforward) networks** with varying widths and dropout rates
* Multiple **CNN architectures** with different convolutional widths and fully connected layers

Each candidate architecture is trained with **multiple random seeds** to account for stochasticity in neural network optimization.

#### Experimental Pipeline

The experiment follows a multi-stage model selection procedure:

1. **Candidate generation**
   A library of neural network architectures is defined.

2. **Training across seeds**
   Each architecture is trained with several random seeds to account for optimization variability.

3. **Seed-aggregated validation performance**
   For each architecture, validation accuracy is averaged across seeds.

4. **Validation-based preselection**
   Architectures within a small margin of the best validation performance are retained:

   [
   \bar A_{\text{val}}(m) \ge \max_j \bar A_{\text{val}}(j) - \delta
   ]

5. **Representative model selection**
   For each shortlisted architecture, a representative training run is selected (the seed with the best validation accuracy).

6. **Test predictions**
   The shortlisted models generate predictions on the test set.

7. **Post-selection inference (MABT)**
   Among the shortlisted models, the one with the best test accuracy is selected.
   The **MABT method** is then used to compute a **selection-adjusted lower confidence bound** for the true performance of this selected model.

#### Stored Outputs

The experiment stores the predictions of all shortlisted models on the test set:

```
outputs/shortlist_test_predictions.csv
```

Structure:

| column    | description                             |
| --------- | --------------------------------------- |
| `y_true`  | true test label                         |
| `model_1` | predictions of first shortlisted model  |
| `model_2` | predictions of second shortlisted model |
| ...       | ...                                     |

Additionally, a binary correctness matrix is stored:

```
outputs/shortlist_test_accuracy_matrix.csv
```

Each entry indicates whether a model correctly classified a test observation:

[
Z_{ij} = \mathbf{1}{\hat y_{ij} = y_i}
]

This matrix is the input used by the **MABT procedure**.

#### Result

The final output of the pipeline is:

* the **naively selected test-best model**
* a **selection-adjusted lower confidence bound** on its accuracy computed via `mabt_ci`.

This example illustrates that **model selection bias also arises in deep learning workflows**, and demonstrates how post-selection inference can provide statistically valid performance guarantees even after architecture comparison and model selection.


### 03_transformers: Post-Selection Performance with Transformers

This example demonstrates how MABT can be used together with Hugging Face transformers and multiple different prediction strategies (single base transformer prediction, majority vote prediction, soft vote ensemble prediction).

The experiment is implemented in `agnews_distilbert.py`.

------------------------------------------------------------------------

#### Experimental setup

This script implements the following workflow:

##### 1. Train several candidate models

Multiple **DistilBERT classifiers** are trained on the AG News dataset
using different configurations:

-   random seeds
-   learning rates
-   number of training epochs
-   weight decay
-   training set size

The goal is to produce **models with different prediction behavior**,
which creates a meaningful model selection problem.

------------------------------------------------------------------------

##### 2. Generate candidate prediction strategies

From the five trained base transformers, several **prediction strategies** are
constructed. These represent the candidate decision rules that could potentially
be deployed. Each of the basic prediction strategies corresponds to using one trained base transformer.

------------------------------------------------------------------------

**Ensemble strategies**

-   `majority_vote_all`\
    Majority vote across all base transformers.

-   `soft_vote_all`\
    Average predicted probabilities across all base transformers.

-   `soft_vote_top2`\
    Soft vote using the two base models with the highest test transformers.

-   `soft_vote_top3`\
    Soft vote using the three best base transformers.

These ensemble prediction strategies represent **typical post-hoc model selection
heuristics** used in practice.

------------------------------------------------------------------------

##### 3. Select the best strategy using the test set

All candidate strategies are evaluated on the test set.

The prediction strategy with the highest accuracy would normally be chosen for
deployment:

best_strategy = argmax(strategy_accuracy)

This step introduces **data-dependent selection**.

------------------------------------------------------------------------

##### 4. Apply MABT

After the candidate strategy set and predictions are constructed, MABT
is applied:

bound, tau, t0 = mabt_ci(true_labels, strategy_preds.T)

MABT produces a lower confidence bound on the true accuracy that
remains valid even after selecting the best strategy based on the test
data.

------------------------------------------------------------------------

#### Summary of the pipeline

The script implements the following pipeline:

train multiple base transformers ↓ construct candidate prediction strategies ↓
evaluate strategies on test data ↓ select best strategy (data-dependent)
↓ apply MABT to obtain valid post-selection confidence bounds


## Roadmap for Further Modules

### 04_distribution_shift