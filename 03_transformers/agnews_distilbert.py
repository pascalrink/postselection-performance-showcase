# %%
# Train several (base) candidate models, generate multiple prediction 
# strategies, select the best strategy based on test accuracy, and then 
# apply MABT to obtain a valid confidence bound after this data-dependent 
# selection

import numpy as np
import random
import torch

from datasets import load_dataset
from mabt import mabt_ci
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments
)


# %%
model_name = "distilbert-base-uncased" # base language model used for all candidates
n_labels = 4

# Candidate base model configurations. Vary seed, learning rate, training size, 
# and regularization. Creates models with different predictive behavior
base_model_configs = [
    {"name": "model1", "seed": 1, "learning_rate": 1e-4, "epochs": 1, "weight_decay": 0.0, "n_train": 500},
    {"name": "model2", "seed": 2, "learning_rate": 5e-5, "epochs": 1, "weight_decay": 0.0, "n_train": 1000},
    {"name": "model3", "seed": 3, "learning_rate": 2e-5, "epochs": 2, "weight_decay": 0.01, "n_train": 1500},
    {"name": "model4", "seed": 4, "learning_rate": 1e-5, "epochs": 3, "weight_decay": 0.01, "n_train": 2000},
    {"name": "model5", "seed": 5, "learning_rate": 8e-6, "epochs": 4, "weight_decay": 0.1, "n_train": 2000},
]


# %%

# Set all random seeds for reproducibility. Each candidate model uses 
# a different seed
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(labels, preds):
    acc = (labels == preds).mean()
    return acc

# Convert logits to probabilities. Used later for soft voting
def softmax(logits):
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# %%
# Load AG News data set
raw_data = load_dataset("ag_news")


# %%
# Shuffle data sets. No explicit validation set. Selection is 
# intentionally performed on the test data in order to apply 
# MABT
train_data = raw_data["train"].shuffle(seed=1)
test_data = raw_data["test"].shuffle(seed=1)

print(f"Example training row:\n{train_data[0]}")


# %%
# Convert text into model input tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True)


# %%
# Apply tokenization
train_data = train_data.map(tokenize_batch, batched=True)
test_data = test_data.map(tokenize_batch, batched=True)


# %%
# Hugging Face trainer expects label column to be exactly named "labels"
train_data = train_data.rename_column("label", "labels").remove_columns("text")
test_data = test_data.rename_column("label", "labels").remove_columns("text")


# %%
# Store test labels separately for evaluation
true_labels = np.array(test_data["labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# %%
# Train one candidate base model
def train_base_model(config):
    
    print(f"Train base model: {config["name"]} | seed={config["seed"]} | lr={config["learning_rate"]}")
    set_seeds(config["seed"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=n_labels
    )

    # Each model trains on a different subset size (to increase diversity across models)
    train_subset = train_data.shuffle(seed=config["seed"]).select(range(config["n_train"]))

    train_args = TrainingArguments(
        output_dir="./outputs", 
        eval_strategy="epoch", 
        save_strategy="no", 
        logging_strategy="epoch", 
        num_train_epochs=config["epochs"], 
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=32, 
        learning_rate=config["learning_rate"], 
        weight_decay=config["weight_decay"], 
        report_to="none", 
        disable_tqdm=True, # got an error when was "False"
        seed=config["seed"]
    )

    trainer = Trainer(
        model=model, 
        args=train_args, 
        train_dataset=train_subset, 
        eval_dataset=test_data, 
        processing_class=tokenizer, 
        data_collator=data_collator
    )

    print("Start training...")
    trainer.train()

    # Evaluate model prediction on the test data
    pred_output = trainer.predict(test_data)
    logits = pred_output.predictions
    preds = np.argmax(logits, axis=1)
    probas = softmax(logits)
    acc = accuracy(true_labels, preds)

    print(f"Test accuracy: {acc:.4f}")

    return {
        "name": config["name"], 
        "logits": logits, 
        "probabilities": probas, 
        "predictions": preds, 
        "accuracy": acc
    }


# %%
# Train all base models
base_model_results = [train_base_model(config) for config in base_model_configs]


# %%
# Collect prediction, probabilities, and accuracies
all_bm_preds = np.stack([bm["predictions"] for bm in base_model_results], axis=0)
all_bm_probas = np.stack([bm["probabilities"] for bm in base_model_results], axis=0)
all_bm_accs = np.stack([bm["accuracy"] for bm in base_model_results], axis=0)

print(f"Base model accuracies:")
for bm in base_model_results:
    print(f"{bm["name"]}: {bm["accuracy"]:.4f}")


# %%
# Majority voting strategy
def majority_vote(pred_mat, n_labels):
    n_test = pred_mat.shape[1]
    maj_vote_vec = np.empty(n_test, dtype=int)

    for i in range(n_test):
        counts = np.bincount(pred_mat[:, i], minlength=n_labels)
        maj_vote_vec[i] = int(np.argmax(counts))

    return maj_vote_vec


# Soft voting strategy: average predicted probabilities across models
def soft_vote(proba_tensor):
    mean_proba_mat = proba_tensor.mean(axis=0)
    return np.argmax(mean_proba_mat, axis=1)


# %%
strategy_preds = []
strategy_names = []

#%%
# Cadidates 1, to 5: each single base model
for bm in base_model_results:
    strategy_names.append(f"single_{bm["name"]}")
    strategy_preds.append(bm["predictions"])


# %%
# Candidate 6: majority vote over all base models
strategy_names.append("majority_vote_all")
strategy_preds.append(majority_vote(all_bm_preds, n_labels))


# %%
# Candidate 7: soft vote ensemble over all base models
strategy_names.append("soft_vote_all")
strategy_preds.append(soft_vote(all_bm_probas))


# %%
# Candidate 8: soft vote over the top-2 base models
base_order = np.argsort(all_bm_accs)[::-1]
top2_idx = base_order[:2]
strategy_names.append("soft_vote_top2")
strategy_preds.append(soft_vote(all_bm_probas[top2_idx]))


# %%
# Candidate 9: soft vote over the top-3 base models
top3_idx = base_order[:3]
strategy_names.append("soft_vote_top3")
strategy_preds.append(soft_vote(all_bm_probas[top3_idx]))


# %%

# Stack predictions of all candidate strategies
strategy_preds = np.stack(strategy_preds, axis=0)

# Compute accuracy on the test data for each candidate strategy
strategy_accs = np.array([
    accuracy(true_labels, preds) for preds in strategy_preds
])

print("Candidate strategy accuracies:")
for name, acc in zip(strategy_names, strategy_accs):
    print(f" {name}: {acc:.4f}")


# %%
# Multiple candidate prediction strategies. We would put the best-performing 
# one into production, based on test set accuracy. This introduces selection 
# bias. MABT corrects for this post-selection setting and yields statistically 
# valid confidence bounds
bound, tau, t0 = mabt_ci(true_labels, strategy_preds.T) # transpose so preds in cols
print(f"Bound: {bound:.6f}")
# %%
