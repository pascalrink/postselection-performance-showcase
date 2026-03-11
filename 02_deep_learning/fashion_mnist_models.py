# %%
import numpy as np
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
    transform = transforms.ToTensor() # rescale pixel values to [0.0, 1.0]

    learn_data = datasets.FashionMNIST(
        root=data_dir, 
        train=True, # obtain designated training set
        download=True, 
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=data_dir, 
        train=False, # obtain designated test set
        download=True, 
        transform=transform
    )

    train_size = len(learn_data) - valid_size

    # Split learning set into training and validation set
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
        pin_memory=torch.cuda.is_available
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
        lr=1e-3
):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

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

        print(
            f"Epoch: {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4f} "
        )


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
    print(
        classification_report(y_true, y_pred, target_names=class_labels)
    )


class FashionCNN(nn.Module):

    def __init__(self, num_classes: int = 10):

        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(

            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x


# %%
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, class_labels = get_fashion_mnist_loaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")

    inspect_one_batch(train_loader, class_labels)
    inspect_mlp_forward_pass(train_loader, device)

    model = FashionMLP()

    train_model(
        model, 
        train_loader, 
        valid_loader, 
        device, 
        epochs=10
    )

    y_true, y_pred = collect_predictions(model, test_loader, device)
    confusion_mat, acc = compute_confusion_matrix(y_true, y_pred)
    print_classification_report(y_true, y_pred, class_labels)

    # %%
    print("\nTraining CNN")

    cnn_model = FashionCNN()

    train_model(
        cnn_model, 
        train_loader, 
        valid_loader, 
        device, 
        epochs=10
    )

    y_true_cnn, y_pred_cnn = collect_predictions(
        cnn_model, 
        test_loader, 
        device
    )

    conf_mat_cnn, acc_cnn = compute_confusion_matrix(y_true_cnn, y_pred_cnn)
    print_classification_report(y_true_cnn, y_pred_cnn, class_labels)


    # %%
    pred_mat = np.empty((len(y_pred), 2))
    pred_mat[:, 0] = y_pred
    pred_mat[:, 1] = y_pred_cnn

    bound, tau, t0 = mabt_ci(y_true, pred_mat)
    print(f"Adjusted 95% lower bound: {bound:.4f}")
# %%
