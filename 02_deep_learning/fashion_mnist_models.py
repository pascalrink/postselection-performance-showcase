# %%
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
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_fashion_mnist_loaders(
        data_dir="data",
        batch_size=128,
        valid_size=10000,
        seed=1
):
    set_seed(seed)

    data_dir = Path(data_dir)
    transform = transforms.ToTensor()

    learn_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(learn_data) - valid_size

    train_data, valid_data = random_split(
        learn_data,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    class_labels = list(learn_data.classes)

    return train_loader, valid_loader, test_loader, class_labels


def inspect_one_batch(loader, class_labels):
    images, labels = next(iter(loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Dtype images: {images.dtype}")
    print(f"Dtype labels: {labels.dtype}")
    print(f"Pixel range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"First 10 labels: {labels[:10].tolist()}")
    print(f"Class labels: {class_labels}")


class FashionMLP(nn.Module):
    def __init__(
            self,
            input_dim=28 * 28,
            hidden_dims=(256, 128, 64),
            num_classes=10,
            dropout_rate=0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.network(x)
        return logits


def inspect_mlp_forward_pass(loader, device):
    model = FashionMLP().to(device)
    criterion = nn.CrossEntropyLoss()

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = criterion(logits, labels)

    print(f"Input shape: {images.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss value: {loss.item():.4f}")


class FashionCNN(nn.Module):
    def __init__(
            self,
            num_classes=10,
            channels=(32, 64),
            hidden_dim=128,
            dropout_rate=0.3
    ):
        super().__init__()

        c1, c2 = channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(c2 * 5 * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x


def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device
):
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


def evaluate(
        model,
        loader,
        criterion,
        device
):
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


def train_model(
        model,
        train_loader,
        valid_loader,
        device,
        epochs=10,
        lr=1e-3,
        verbose=True
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        valid_loss, valid_acc = evaluate(
            model,
            valid_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        if verbose:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} "
                f"valid_acc={valid_acc:.4f}"
            )

    return history


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


def compute_confusion_matrix(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(confusion_mat)

    acc = np.mean(y_true == y_pred)
    print(f"Test accuracy: {acc:.4f}")

    return confusion_mat, acc


def print_classification_report(y_true, y_pred, class_labels):
    print(classification_report(y_true, y_pred, target_names=class_labels))


def save_shortlist_predictions_csv(
        y_true,
        pred_mat,
        model_names,
        output_path="outputs/shortlist_test_predictions.csv"
):
    """
    Speichert die Testvorhersagen ALLER vorselektierten Modelle als CSV.

    Spalten:
    - y_true
    - eine Spalte pro vorselektiertem Modell
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"y_true": y_true.astype(int)})

    for j, model_name in enumerate(model_names):
        df[model_name] = pred_mat[:, j].astype(int)

    df.to_csv(output_path, index=False)

    print(f"\nSaved shortlist test predictions to: {output_path.resolve()}")
    print(f"CSV shape: {df.shape}")

    return df


def save_shortlist_accuracy_matrix_csv(
        y_true,
        pred_mat,
        model_names,
        output_path="outputs/shortlist_test_accuracy_matrix.csv"
):
    """
    Speichert die 0/1-Korrektheitsmatrix ALLER vorselektierten Modelle.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acc_mat = (pred_mat == y_true[:, None]).astype(int)

    df = pd.DataFrame(acc_mat, columns=model_names)
    df.insert(0, "y_true", y_true.astype(int))

    df.to_csv(output_path, index=False)

    print(f"Saved shortlist accuracy matrix to: {output_path.resolve()}")
    print(f"CSV shape: {df.shape}")

    return df


def get_candidate_specs():
    return [
        {
            "name": "mlp_small",
            "model_class": FashionMLP,
            "model_kwargs": {
                "hidden_dims": (128, 64),
                "dropout_rate": 0.2,
            },
            "epochs": 10,
            "lr": 1e-3,
        },
        {
            "name": "mlp_standard",
            "model_class": FashionMLP,
            "model_kwargs": {
                "hidden_dims": (256, 128, 64),
                "dropout_rate": 0.3,
            },
            "epochs": 10,
            "lr": 1e-3,
        },
        {
            "name": "mlp_wide",
            "model_class": FashionMLP,
            "model_kwargs": {
                "hidden_dims": (512, 256, 128),
                "dropout_rate": 0.3,
            },
            "epochs": 10,
            "lr": 1e-3,
        },
        {
            "name": "cnn_small",
            "model_class": FashionCNN,
            "model_kwargs": {
                "channels": (16, 32),
                "hidden_dim": 64,
                "dropout_rate": 0.3,
            },
            "epochs": 10,
            "lr": 1e-3,
        },
        {
            "name": "cnn_standard",
            "model_class": FashionCNN,
            "model_kwargs": {
                "channels": (32, 64),
                "hidden_dim": 128,
                "dropout_rate": 0.3,
            },
            "epochs": 10,
            "lr": 1e-3,
        },
        {
            "name": "cnn_wide_lowdrop",
            "model_class": FashionCNN,
            "model_kwargs": {
                "channels": (32, 64),
                "hidden_dim": 256,
                "dropout_rate": 0.1,
            },
            "epochs": 10,
            "lr": 1e-3,
        },
    ]


def train_candidate_once(
        spec,
        train_loader,
        valid_loader,
        test_loader,
        device,
        seed,
        verbose=False,
):
    set_seed(seed)

    model = spec["model_class"](**spec["model_kwargs"])

    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=spec["epochs"],
        lr=spec["lr"],
        verbose=verbose,
    )

    y_valid_true, y_valid_pred = collect_predictions(model, valid_loader, device)
    valid_acc = np.mean(y_valid_true == y_valid_pred)

    y_test_true, y_test_pred = collect_predictions(model, test_loader, device)
    test_acc = np.mean(y_test_true == y_test_pred)

    return {
        "seed": seed,
        "model": model,
        "history": history,
        "y_valid_true": y_valid_true,
        "y_valid_pred": y_valid_pred,
        "valid_acc": valid_acc,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "test_acc": test_acc,
    }


def train_candidate_over_seeds(
        spec,
        train_loader,
        valid_loader,
        test_loader,
        device,
        seeds,
        verbose=False,
):
    print(f"\n=== Candidate: {spec['name']} ===")

    runs = []

    for seed in seeds:
        print(f"Running seed {seed} ...")
        run_result = train_candidate_once(
            spec=spec,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            seed=seed,
            verbose=verbose,
        )
        runs.append(run_result)

        print(
            f"seed={seed} | "
            f"valid_acc={run_result['valid_acc']:.4f} | "
            f"test_acc={run_result['test_acc']:.4f}"
        )

    valid_accs = np.array([run["valid_acc"] for run in runs])
    test_accs = np.array([run["test_acc"] for run in runs])

    agg_result = {
        "name": spec["name"],
        "spec": spec,
        "runs": runs,
        "valid_acc_mean": float(valid_accs.mean()),
        "valid_acc_std": float(valid_accs.std(ddof=1)) if len(valid_accs) > 1 else 0.0,
        "test_acc_mean": float(test_accs.mean()),
        "test_acc_std": float(test_accs.std(ddof=1)) if len(test_accs) > 1 else 0.0,
    }

    print(
        f"Aggregate {spec['name']} | "
        f"valid_mean={agg_result['valid_acc_mean']:.4f} "
        f"(sd={agg_result['valid_acc_std']:.4f}) | "
        f"test_mean={agg_result['test_acc_mean']:.4f} "
        f"(sd={agg_result['test_acc_std']:.4f})"
    )

    return agg_result


def select_candidates_by_margin(
        results,
        delta=0.002,
        min_shortlist_size=2
):
    """
    Seed-aggregierte Vorselektion per Margin.

    Falls die Margin nur sehr wenige Modelle liefert, wird auf die besten
    `min_shortlist_size` Kandidaten aufgefüllt, damit die Shortlist
    tatsächlich mehrere Modelle enthält.
    """
    results_sorted = sorted(
        results,
        key=lambda x: x["valid_acc_mean"],
        reverse=True
    )

    best_valid_mean = results_sorted[0]["valid_acc_mean"]
    cutoff = best_valid_mean - delta

    shortlist = [
        res for res in results_sorted
        if res["valid_acc_mean"] >= cutoff
    ]

    if len(shortlist) < min_shortlist_size:
        shortlist = results_sorted[:min_shortlist_size]

    print("\n=== Seed-aggregated validation ranking ===")
    for rank, res in enumerate(results_sorted, start=1):
        print(
            f"{rank:02d}. {res['name']:18s} "
            f"valid_mean={res['valid_acc_mean']:.4f} "
            f"valid_sd={res['valid_acc_std']:.4f} "
            f"test_mean={res['test_acc_mean']:.4f} "
            f"test_sd={res['test_acc_std']:.4f}"
        )

    print(f"\nBest aggregated validation accuracy: {best_valid_mean:.4f}")
    print(f"Margin delta: {delta:.4f}")
    print(f"Shortlist cutoff: {cutoff:.4f}")
    print(f"Minimum shortlist size: {min_shortlist_size}")

    print("\n=== Validation shortlist ===")
    for res in shortlist:
        print(
            f"- {res['name']} "
            f"(valid_mean={res['valid_acc_mean']:.4f}, "
            f"test_mean={res['test_acc_mean']:.4f})"
        )

    return shortlist


def choose_representative_run(candidate_result, rule="best_valid"):
    runs = candidate_result["runs"]

    if rule == "best_valid":
        best_run = max(runs, key=lambda x: x["valid_acc"])
    elif rule == "median_valid":
        sorted_runs = sorted(runs, key=lambda x: x["valid_acc"])
        best_run = sorted_runs[len(sorted_runs) // 2]
    else:
        raise ValueError(f"Unknown representative rule: {rule}")

    return best_run


def build_pred_matrix_from_shortlist(shortlist, representative_rule="best_valid"):
    """
    Baut die Vorhersagematrix aus ALLEN vorselektierten Modellen.

    Für jeden shortlist-Kandidaten wird genau ein repräsentativer Run gewählt,
    und alle diese Vorhersagevektoren werden spaltenweise zusammengefügt.
    """
    chosen_runs = []

    for candidate_result in shortlist:
        run = choose_representative_run(
            candidate_result,
            rule=representative_rule
        )
        chosen_runs.append({
            "candidate_name": candidate_result["name"],
            "seed": run["seed"],
            "y_test_true": run["y_test_true"],
            "y_test_pred": run["y_test_pred"],
            "valid_acc": run["valid_acc"],
            "test_acc": run["test_acc"],
        })

    y_true = chosen_runs[0]["y_test_true"]
    pred_columns = [run["y_test_pred"] for run in chosen_runs]
    pred_mat = np.column_stack(pred_columns)
    model_names = [
        f"{run['candidate_name']}|seed={run['seed']}"
        for run in chosen_runs
    ]

    return y_true, pred_mat, model_names, chosen_runs


# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, class_labels = get_fashion_mnist_loaders(
        batch_size=128,
        valid_size=10000,
        seed=1
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")

    inspect_one_batch(train_loader, class_labels)
    inspect_mlp_forward_pass(train_loader, device)


    # %%
    # --------------------------------------------------
    # 1. Kandidatenmenge
    # --------------------------------------------------
    candidate_specs = get_candidate_specs()

    # --------------------------------------------------
    # 2. Mehrere Seeds pro Kandidat
    # --------------------------------------------------
    seeds = [1, 2, 3]
    all_results = []

    for spec in candidate_specs:
        result = train_candidate_over_seeds(
            spec=spec,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            seeds=seeds,
            verbose=False,
        )
        all_results.append(result)

    # %%
    # --------------------------------------------------
    # 3. Seed-aggregierte Vorselektion per Margin
    #    mit garantierter Mindestgröße der Shortlist
    # --------------------------------------------------
    shortlist = select_candidates_by_margin(
        results=all_results,
        delta=0.001,
        min_shortlist_size=2
    )

    # %%
    # --------------------------------------------------
    # 4. Vorhersagematrix ALLER vorselektierten Modelle
    # --------------------------------------------------
    y_true, pred_mat, shortlist_names, chosen_runs = build_pred_matrix_from_shortlist(
        shortlist,
        representative_rule="best_valid"
    )

    print("\n=== Shortlist runs passed to MABT ===")
    for run_name, run in zip(shortlist_names, chosen_runs):
        print(
            f"- {run_name}: "
            f"valid_acc={run['valid_acc']:.4f}, "
            f"test_acc={run['test_acc']:.4f}"
        )

    print(f"\nPrediction matrix shape: {pred_mat.shape}")
    print(f"Number of shortlisted models: {len(shortlist_names)}")

    # %%
    # --------------------------------------------------
    # 5. CSV-Export der Testvorhersagen ALLER vorselektierten Modelle
    # --------------------------------------------------
    # save_shortlist_predictions_csv(
    #     y_true=y_true,
    #     pred_mat=pred_mat,
    #     model_names=shortlist_names,
    #     output_path="outputs/shortlist_test_predictions.csv"
    # )

    # save_shortlist_accuracy_matrix_csv(
    #     y_true=y_true,
    #     pred_mat=pred_mat,
    #     model_names=shortlist_names,
    #     output_path="outputs/shortlist_test_accuracy_matrix.csv"
    # )

    # %%
    # --------------------------------------------------
    # 6. Naive Testwahl innerhalb der Shortlist
    # --------------------------------------------------
    shortlist_test_accs = np.mean(pred_mat == y_true[:, None], axis=0)
    best_idx = int(np.argmax(shortlist_test_accs))
    best_name = shortlist_names[best_idx]

    print("\n=== Naive test ranking within shortlist ===")
    for name, acc in zip(shortlist_names, shortlist_test_accs):
        print(f"- {name}: test_acc={acc:.4f}")

    print(f"\nNaively selected test winner within shortlist: {best_name}")

    # %%
    # --------------------------------------------------
    # 7. MABT auf der shortlist-basierten Kandidatenmenge
    # --------------------------------------------------
    imported_results = pd.read_csv("outputs/shortlist_test_predictions.csv")
    y_true = imported_results["y_true"]
    pred_mat = imported_results.drop(columns="y_true")


    # %%
    bound, tau, t0 = mabt_ci(y_true, pred_mat)

    print("\n=== MABT results ===")
    print(f"Adjusted 95% lower bound: {bound:.4f}")
    print(f"tau = {tau}")
    print(f"t0 = {t0}")

    # %%
    # --------------------------------------------------
    # 8. Diagnose für den naiven Testsieger
    # --------------------------------------------------
    winner_run = chosen_runs[best_idx]

    print(f"\n=== Detailed report for selected winner: {best_name} ===")
    confusion_mat, acc = compute_confusion_matrix(
        winner_run["y_test_true"],
        winner_run["y_test_pred"]
    )
    print_classification_report(
        winner_run["y_test_true"],
        winner_run["y_test_pred"],
        class_labels
    )