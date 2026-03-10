# %%
# Evaluation of prediction performance after model selection using MABT and SHAP: 
# We demonstrate a statistically rigorous model selection workflow. One of 
# the main problems in ML is the "winner's curse": selecting the best model 
# among a multitude of candidate models based on its performance on the test 
# data, the observed performance is upwards biased (too optimistic). MABT 
# yields a lower confidence bound that corrects for this bias.


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import shap

from mabt import mabt_ci
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# %%
rs = 12345 # global seed for reproducibility


# %%
# Load German credit data set from OpenML, a typical data set 
# from a regulated environment (finance), where interpretability 
# and uncertainty are critical issues.
dataset = fetch_openml(data_id=31, as_frame=True, parser="auto")
df = dataset.frame


# %%

# Map the class labels {"good", "bad"} to {"1", "0"}
df["class"] = df["class"].map({"good": 1, "bad": 0})

X = df.drop("class", axis=1)
y = df["class"]


# %%

# Identify numerical and categorical features
num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_feats = X.select_dtypes(include=["category", "object"]).columns.tolist()


# %%
# We use a strict three-split design (training/validation/test):
#  - training: estimation of model parameter
#  - validation: tuning of hyperparameters and model preselection
#  - test: final model selection and evaluation using MABT

# Split into 80% for learning (training and validation) and 20% for testing (evaluation)
X_learn, X_test, y_learn, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=rs, 
    stratify=y
)

# Split learning set into training and validation set such that 
# we obtain a three-split into 60% for training, 20% for validation 
# and 20% for testing
X_train, X_valid, y_train, y_valid = train_test_split(
    X_learn, y_learn, 
    test_size=0.25, 
    random_state=rs, 
    stratify=y_learn
)


# %%

# Explorative data analysis: plot label counts for target variable 
# and feature heatmap to check for collinearity
def plot_eda(df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(x="class", data=df)
    plt.title("Class distribution of target variable")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        df[num_feats + ["class"]].corr(), 
        annot=True, 
        fmt=".2f"
    )
    plt.title("Correlation matrix of numerical variables")

    plt.show()

plot_eda(df)


# %%
num_transformer = StandardScaler() # for lasso
cat_transformer = OneHotEncoder(handle_unknown="ignore") # make dummies from cat features

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_feats), 
        ("cat", cat_transformer, cat_feats)
    ])


# %%

# Feature engineering and model families
# We will consider three model families:
#  - lasso: logistic regression with l1 penalty (linear), induces sparsity (feature selection)
#  - random forest: combines multiple decision trees and aggregate (bagging) predictions (ensemble; nonlinear)
#  - xgboost: sequential improvement (boosting) using gradient descent
model_families = {
    "Lasso": [LogisticRegression(l1_ratio=1, solver="liblinear", C=c) 
              for c in np.logspace(-3, 1, 10)],
    "Random Forest": [RandomForestClassifier(max_depth=d, n_estimators=n, random_state=rs) 
                      for d in [3, 5, 8, 12, None] for n in [100, 300]], 
    "XGBoost": [XGBClassifier(learning_rate=lr, max_depth=d, n_estimators=n, random_state=rs) 
                for lr in [0.01, 0.1] for d in [3, 5] for n in [50, 100]]
}

# %%
# Obtain accuracy from all 30 candidate models on the validation set, 
# and select the two best-performing models from each family for future 
# considerations
selection_results = []
for family_name, candidate_list in model_families.items():
    family_scores = []
    for i, clf in enumerate(candidate_list):
        pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])
        pipeline.fit(X_train, y_train)
        valid_acc = accuracy_score(y_valid, pipeline.predict(X_valid)) # alternatively: cv on learning data
        family_scores.append({"model_obj": clf, "valid_acc": valid_acc})

    top2_from_fam = sorted(family_scores, key=lambda x: x["valid_acc"], reverse=True)[:2]
    for res in top2_from_fam: # select the two best-performing models from each model family for testing
        res["family"] = family_name
        selection_results.append(res)


# %%
# Final training and performance matrix
test_performance = []
preds_mat = np.empty((len(y_test), len(selection_results)))
for i, res in enumerate(selection_results):
    # Retrain selected models on entire learning data
    test_pipeline = Pipeline([('pre', preprocessor), ('clf', res['model_obj'])])
    test_pipeline.fit(X_learn, y_learn)
    preds = test_pipeline.predict(X_test)
    preds_mat[:, i] = preds # save predictions per model for future use (mabt)
    test_acc = accuracy_score(y_test, preds)
    test_performance.append({
        "id": i, 
        "family": res["family"], 
        "pipeline": test_pipeline, 
        "test_acc": test_acc
    })


# %%
best_model = max(test_performance, key=lambda x: x["test_acc"])
print(f"Best model is {best_model["family"]} with test accuracy {best_model["test_acc"]}")


# %%
# MABT: multiplicity-adjusted bootstrap tilting lower confidence bound for 
# prediction performance post-model selection. Correction is obtained using 
# a maxT-approach (multiple testing, simultaneous inference). We need to 
# account for that because we selected the final model from the test data, 
# which we also use for evaluating the model's predicitive performance.
bound, tau, t0 = mabt_ci(y_test, preds_mat) # valid post-selection lower 95% confidence limit
print(f"Corresponding post-selection 95% lower confidence limit is {bound:.6f}")


# %%

# SHAP: Shapley Addative Explanations (quantify contribution of features)
# In regulated environments (pharma, finance), SHAP can reveal whether 
# protected features (such as age or sex) are used for decision-making, 
# or why a specific credit application was declined
best_pipeline = best_model["pipeline"]
preprocessor = best_pipeline.named_steps["pre"]
classifier = best_pipeline.named_steps["clf"]

ohe_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_feats)
all_features_raw = num_feats + list(ohe_names)

all_features_clean = [re.sub(r'[<>]', '_', f) for f in all_features_raw]

# For SHAP we need the data pre-model training, but post-processing
X_test_transformed = preprocessor.transform(X_test)
X_test_df = pd.DataFrame(X_test_transformed, columns=all_features_clean)

if hasattr(classifier, "get_booster"):
    classifier.get_booster().feature_names = all_features_clean

if best_model["family"] == "Lasso":
    explainer = shap.LinearExplainer(classifier, X_test_df) # linear explainer for lasso
else:
    explainer = shap.TreeExplainer(classifier) # use tree-like structure

shap_values = explainer.shap_values(X_test_df) # compute SHAP values

if isinstance(shap_values, list):
    actual_shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    actual_shap_values = shap_values[:, :, 1]
else:
    actual_shap_values = shap_values

# Beeswarm plot shows importance and direction of feature contribution to getting a credit
plt.figure(figsize=(12, 8))
shap.summary_plot(actual_shap_values, X_test_df, show=False)
plt.title(f"SHAP beeswarm plot: {best_model["family"]} (contribution to getting a credit)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(actual_shap_values, X_test_df, plot_type="bar", show=False)
plt.title(f"Global feature importance (mean absolute SHAP value)")
plt.tight_layout()
plt.show()


# %%

# Analysis of an individual observation
instance_idx = 0
instance_data = X_test_df.iloc[instance_idx]

# Waterfall plot
plt.figure(figsize=(10, 6))

exp = shap.Explanation(
    values=actual_shap_values[instance_idx], 
    base_values=explainer.expected_value[1] if isinstance(
        explainer.expected_value, (list, np.ndarray)
    ) else explainer.expected_value, data=instance_data, feature_names=all_features_clean
)

shap.plots.waterfall(exp, show=False)
plt.title(f"Local explanation for observation {instance_idx} (model: {best_model["family"]})")
plt.show()


# %%

# Analyze misclassifications
test_preds = best_pipeline.predict(X_test)
misclassified_indices = np.where(test_preds != y_test.values)[0]

if len(misclassified_indices) > 0:
    print(f"Number of misclassifications in the test data: {len(misclassified_indices)}")

# Check first misclassification
error_idx = misclassified_indices[0]

print(f"\nError analysis for test index {error_idx}:")
print(f"True class label: {y_test.values[error_idx]}; predicted class label: {test_preds[error_idx]}")

# Extract the top-3 SHAP contributions for this misclassification
error_shap = actual_shap_values[error_idx]
top_indices = np.argsort(np.abs(error_shap))[-3:][::-1]

print("Top 3 features with biggest contribution for misclassifcation:")
for idx in top_indices:
    print(f"- {all_features_clean[idx]}: {error_shap[idx]:.4f}")
# %%
