# %%
# Train several simple MLP and CNN candidate models on Fashion-MNIST,
# select a shortlist based on validation accuracy, and then apply MABT
# to obtain a valid confidence bound after this data-dependent selection

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

from mabt import mabt_ci
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# %%
# Candidate model configurations
model_configs = [
    {
        "name": "mlp_small",
        "model_type": "mlp",
        "hidden_dims": (128, 64),
        "dropout_rate": 0.2,
        "epochs": 10,
        "learning_rate": 1e-3,
    },
    {
        "name": "mlp_large",
        "model_type": "mlp",
        "hidden_dims": (256, 128),
        "dropout_rate": 0.3,
        "epochs": 10,
        "learning_rate": 1e-3,
    },
    {
        "name": "cnn_small",
        "model_type": "cnn",
        "channels": (16, 32),
        "hidden_dim": 64,
        "dropout_rate": 0.3,
        "epochs": 10,
        "learning_rate": 1e-3,
    },
    {
        "name": "cnn_large",
        "model_type": "cnn",
        "channels": (32, 64),
        "hidden_dim": 128,
        "dropout_rate": 0.3,
        "epochs": 10,
        "learning_rate": 1e-3,
    },
]

seeds = [1, 2, 3]
delta = 0.001
min_shortlist_size = 2

# %%
# Set all random seeds for reproducibility
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    return acc


# %%
# Load Fashion-MNIST data and create train/validation/test loaders
def get_fashion_mnist_loaders(data_dir="data", batch_size=128, valid_size=10000, seed=1):
    set_seed(seed)

    transform = transforms.ToTensor()

    learn_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(learn_data) - valid_size
    train_data, valid_data = random_split(
        learn_data,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    class_labels = list(learn_data.classes)
    return train_loader, valid_loader, test_loader, class_labels


# %%
class FashionMLP(nn.Module):
    def __init__(self, hidden_dims=(128, 64), dropout_rate=0.2, num_classes=10):
        super().__init__()

        layers = [nn.Flatten()]
        in_features = 28 * 28

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FashionCNN(nn.Module):
    def __init__(self, channels=(16, 32), hidden_dim=64, dropout_rate=0.3, num_classes=10):
        super().__init__()

        c1, c2 = channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 5 * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# %%
def make_model(config):
    if config["model_type"] == "mlp":
        return FashionMLP(
            hidden_dims=config["hidden_dims"],
            dropout_rate=config["dropout_rate"],
        )

    if config["model_type"] == "cnn":
        return FashionCNN(
            channels=config["channels"],
            hidden_dim=config["hidden_dim"],
            dropout_rate=config["dropout_rate"],
        )

    raise ValueError(f"Unknown model_type: {config['model_type']}")


# %%
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc



def train_model(model, train_loader, valid_loader, device, epochs=10, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4f}"
        )


# %%
# Collect class predictions
def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return y_true, y_pred


# %%
# Train one candidate once for one seed
def train_candidate(config, train_loader, valid_loader, test_loader, device, seed):
    print(f"Train model: {config['name']} | seed={seed}")
    set_seed(seed)

    model = make_model(config)
    train_model(
        model,
        train_loader,
        valid_loader,
        device,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
    )

    y_valid_true, y_valid_pred = collect_predictions(model, valid_loader, device)
    y_test_true, y_test_pred = collect_predictions(model, test_loader, device)

    valid_acc = accuracy(y_valid_true, y_valid_pred)
    test_acc = accuracy(y_test_true, y_test_pred)
    print(f"Validation accuracy: {valid_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return {
        "name": config["name"],
        "seed": seed,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }


# %%
# Save test predictions of all shortlisted models
def save_shortlist_predictions_csv(
    y_true,
    pred_mat,
    model_names,
    output_path="outputs/shortlist_test_predictions.csv",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"y_true": y_true.astype(int)})
    for j, model_name in enumerate(model_names):
        df[model_name] = pred_mat[:, j].astype(int)

    df.to_csv(output_path, index=False)
    print(f"Saved shortlist test predictions to: {output_path.resolve()}")
    return df


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_loader, valid_loader, test_loader, class_labels = get_fashion_mnist_loaders()

# %%
# Train all candidate models for several seeds
all_results = []

for config in model_configs:
    print(f"\nModel group: {config['name']}")

    config_results = []
    for seed in seeds:
        result = train_candidate(config, train_loader, valid_loader, test_loader, device, seed)
        config_results.append(result)

    valid_accs = np.array([result["valid_acc"] for result in config_results])
    test_accs = np.array([result["test_acc"] for result in config_results])

    summary = {
        "name": config["name"],
        "runs": config_results,
        "valid_acc_mean": valid_accs.mean(),
        "test_acc_mean": test_accs.mean(),
    }
    all_results.append(summary)

    print(f"Mean validation accuracy: {summary['valid_acc_mean']:.4f}")
    print(f"Mean test accuracy: {summary['test_acc_mean']:.4f}")


# %%
# Rank candidate groups by mean validation accuracy and build a shortlist
all_results = sorted(all_results, key=lambda x: x["valid_acc_mean"], reverse=True)
best_valid_acc = all_results[0]["valid_acc_mean"]
cutoff = best_valid_acc - delta

shortlist = [result for result in all_results if result["valid_acc_mean"] >= cutoff]
if len(shortlist) < min_shortlist_size:
    shortlist = all_results[:min_shortlist_size]

print("Validation ranking:")
for result in all_results:
    print(
        f"{result['name']}: "
        f"valid_mean={result['valid_acc_mean']:.4f} "
        f"test_mean={result['test_acc_mean']:.4f}"
    )

print(f"Shortlist cutoff: {cutoff:.4f}")
print(f"Shortlist: {[result['name'] for result in shortlist]}")


# %%
# For each shortlisted candidate group, keep the run with the best validation accuracy
chosen_runs = []

for result in shortlist:
    best_run = max(result["runs"], key=lambda x: x["valid_acc"])
    chosen_runs.append(best_run)

    print(
        f"Selected run: {best_run['name']} | "
        f"seed={best_run['seed']} | "
        f"valid_acc={best_run['valid_acc']:.4f} | "
        f"test_acc={best_run['test_acc']:.4f}"
    )


# %%
# Stack predictions of shortlisted models
y_true = chosen_runs[0]["y_test_true"]
pred_mat = np.column_stack([run["y_test_pred"] for run in chosen_runs])
model_names = [f"{run['name']}|seed={run['seed']}" for run in chosen_runs]

save_shortlist_predictions_csv(y_true, pred_mat, model_names)


# %%
# Test ranking inside the shortlist
test_accs = np.array([accuracy(y_true, pred_mat[:, j]) for j in range(pred_mat.shape[1])])
best_idx = int(np.argmax(test_accs))

print("Shortlist test accuracies:")
for name, acc in zip(model_names, test_accs):
    print(f"{name}: {acc:.4f}")

print(f"Best model in test data: {model_names[best_idx]}")


# %%
# Apply MABT to the shortlisted models
bound, tau, t0 = mabt_ci(y_true, pred_mat)
print(f"MABT lower bound: {bound:.4f}")
print(f"tau: {tau}")
print(f"t0: {t0}")


# %%
# Print standard diagnostics for the naive test winner
best_run = chosen_runs[best_idx]
print("Confusion matrix:")
print(confusion_matrix(best_run["y_test_true"], best_run["y_test_pred"]))

print("Classification report:")
print(
    classification_report(
        best_run["y_test_true"],
        best_run["y_test_pred"],
        target_names=class_labels,
    )
)

# %%
