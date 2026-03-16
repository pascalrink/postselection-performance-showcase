# %%
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

from mabt import mabt_ci
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier


# %%
test_size = 0.3
random_state = 1
shift_feat = "Variance" # feature for shifting dist between train and test
shift_quantile = 0.7


# %%

# Set seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)#
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def softmax(logits):
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def accuracy(true_labels, preds):
    acc = (true_labels == preds).mean()
    return acc


# Load cardiotocography data set from UCI ML repository
def load_ctg_data():
    ctg = fetch_ucirepo(id=193) # cardiotocography data set
    X = ctg.data.features.copy()
    y = ctg.data.targets.copy()
    
    y = y["NSP"].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X.columns = [str(col) for col in X.columns]

    return X, y


# Systematic covariate shift: train and test differ in one feature; 
# lower values go mostly to training, upper tail goes to testing
def shift_split(X, y, shift_feat, q):
    
    threshold = X[shift_feat].quantile(q)

    to_train = X[shift_feat] <= threshold
    to_test = X[shift_feat] > threshold

    X_train = X.loc[to_train]
    y_train = y[to_train]

    X_test = X.loc[to_test]
    y_test = y[to_test]

    return {
        "scenario": "shifted", 
        "threshold": threshold, 
        "X_train": X_train, 
        "y_train": y_train, 
        "X_test": X_test, 
        "y_test": y_test
    }


# Standard stratified split (baseline scenario)
def no_shift_split(X, y, test_size=0.3, random_state=1):
    
    splitter = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(X, y))

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    return {
        "scenario": "no_shift", 
        "X_train": X_train, 
        "y_train": y_train, 
        "X_test": X_test, 
        "y_test": y_test
    }


# Summarize no-shift and shift-scenario
def split_summary(scenario, shift_feat): 
    out_df = pd.DataFrame(
        {
            "split": ["train", "test"], 
            "n": [len(scenario["X_train"]), len(scenario["X_test"])], 

            f"{shift_feat}_mean": [
                scenario["X_train"][shift_feat].mean(), 
                scenario["X_test"][shift_feat].mean()
            ], 

            f"{shift_feat}_median": [
                scenario["X_train"][shift_feat].median(), 
                scenario["X_test"][shift_feat].median()
            ], 

            "class_0_share": [
                np.mean(scenario["y_train"] == 0), 
                np.mean(scenario["y_test"] == 0)
            ], 

            "class_1_share": [
                np.mean(scenario["y_train"] == 1), 
                np.mean(scenario["y_test"] == 1)
            ], 

            "class_2_share": [
                np.mean(scenario["y_train"] == 2), 
                np.mean(scenario["y_test"] == 2)
            ]
        }
    )

    return out_df


# Small NN used as a candidate model
class SmallMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# sklearn-style wrapper to use same fit and evaluate framework
class TorchMLPClassifier:
    def __init__(
        self,
        hidden_seed=1,
        n_epochs=2200,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4
    ):
        self.hidden_seed = hidden_seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.model = None
        self.classes_ = None

    def get_params(self, deep=True):
        return {
            "hidden_seed": self.hidden_seed,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        set_seeds(self.hidden_seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        self.classes_ = np.unique(y)

        X_tensor = torch.tensor(X_scaled)
        y_tensor = torch.tensor(y)

        self.model = SmallMLP(
            input_dim=X_scaled.shape[1],
            n_classes=len(self.classes_)
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        data_set = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model.train()
        for _ in range(self.n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor).numpy()

        return softmax(logits)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


# Candidate models
def make_models(random_state=1):
    logreg_model = Pipeline([
        ("scale", StandardScaler()), 
        ("model", LogisticRegression(
            max_iter=3000, 
            random_state=random_state
        ))
    ])

    rf_model = RandomForestClassifier(
        n_estimators=400, 
        min_samples_leaf=3, 
        random_state=random_state, 
        n_jobs=1
    )

    xgboost_model = XGBClassifier(
        n_estimators=300, 
        max_depth=4, 
        learning_rate=0.05, 
        subsample=0.9, 
        colsample_bytree=0.9, 
        objective="multi:softprob", 
        eval_metric="mlogloss", 
        random_state=random_state, 
        n_jobs=4
    )

    mlp_model = TorchMLPClassifier(
        hidden_seed=random_state, 
        n_epochs=200, 
        batch_size=64, 
        learning_rate=1e-3, 
        weight_decay=1e-4
    )

    return {
        "logreg": logreg_model, 
        "random_forest": rf_model, 
        "xgboost": xgboost_model, 
        "mlp": mlp_model
    }


# Fit and evaluate models
def fit_eval_models(scenario):
    
    models = make_models()

    fitted_models = {}
    model_results = []

    for name, model in models.items():
        print(f"Fit model: {scenario["scenario"]} | {name}")
        fitted_model = clone(model)
        fitted_model.fit(scenario["X_train"], scenario["y_train"])

        probas = fitted_model.predict_proba(scenario["X_test"])
        preds = np.argmax(probas, axis=1)
        acc = accuracy(scenario["y_test"], preds)

        fitted_models[name] = {
            "model": fitted_model, 
            "predictions": preds, 
            "probabilities": probas, 
            "accuracy": acc
        }

        model_results.append(
            {
                "scenario": scenario["scenario"], 
                "model": name, 
                "accuracy": acc
            }
        )

        print(f"Test accuracy: {acc:.4f}")

    model_results = pd.DataFrame(model_results).sort_values("accuracy", ascending=False)
    return fitted_models, model_results


# %%%
if __name__ == "__main__":

    X, y = load_ctg_data()
    class_labels = ["normal", "suspect", "pathologic"]

    # %%
    no_shift_scen = no_shift_split(X, y, test_size=test_size, random_state=random_state)
    shift_scen = shift_split(X, y, shift_feat=shift_feat, q=shift_quantile)

    no_shift_sum = split_summary(no_shift_scen, shift_feat)
    shift_sum = split_summary(shift_scen, shift_feat)

    print("Split summary: no shift")
    print(no_shift_sum)

    print("Split summary: shifted")
    print(shift_sum)

    # %%
    no_shift_models, no_shift_results = fit_eval_models(no_shift_scen)
    shift_models, shift_results = fit_eval_models(shift_scen)

    print("Model results, no shift:")
    print(no_shift_results)

    print("Model results, shifted:")
    print(shift_results)

    # %%
    pred_matrix = np.column_stack([
        shift_models[m]["predictions"]
        for m in shift_models
    ])

    bound, tau, t0 = mabt_ci(shift_scen["y_test"], pred_matrix)
    print(f"MABT lower limit: {bound:.4f}")
# %%
