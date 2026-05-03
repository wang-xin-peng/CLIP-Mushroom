import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class LinearProbe(nn.Module):
    """Linear classifier on top of frozen CLIP image features."""

    def __init__(self, feature_dim=512, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(clip_wrapper, train_dataset, val_dataset, num_classes,
                       epochs=20, lr=1e-3, batch_size=64, device=None):
    """Train a linear probe on frozen CLIP features.

    Pre-extracts all features first for speed, then trains only the linear layer.

    Args:
        clip_wrapper: CLIPWrapper instance
        train_dataset: FungiDataset with integer labels
        val_dataset: FungiDataset with integer labels
        num_classes: number of seen classes
        epochs: training epochs
        lr: learning rate
        batch_size: batch size for training
        device: torch device

    Returns:
        model: trained LinearProbe
        history: dict with train_loss and val_acc per epoch
    """
    device = device or clip_wrapper.device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Pre-extract training features (freeze CLIP encoder)
    print("Pre-extracting training features...")
    train_features, train_labels = [], []
    clip_wrapper.model.eval()
    with torch.no_grad():
        for batch in train_loader:
            px = batch["pixel_values"].to(device)
            feats = clip_wrapper.encode_images(px).cpu()
            train_features.append(feats)
            train_labels.append(batch["label"])
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    print(f"  Extracted {len(train_features)} features, dim={train_features.shape[1]}")

    # Pre-extract validation features
    print("Pre-extracting validation features...")
    val_features, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            px = batch["pixel_values"].to(device)
            feats = clip_wrapper.encode_images(px).cpu()
            val_features.append(feats)
            val_labels.append(batch["label"])
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    print(f"  Extracted {len(val_features)} validation features")

    # Train linear probe
    model = LinearProbe(feature_dim=train_features.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        # Shuffle training data each epoch
        perm = torch.randperm(train_features.size(0))
        train_features_shuffled = train_features[perm]
        train_labels_shuffled = train_labels[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_features), batch_size):
            batch_feats = train_features_shuffled[i:i + batch_size].to(device)
            batch_labels = train_labels_shuffled[i:i + batch_size].to(device)

            outputs = model(batch_feats)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(val_features.to(device))
            _, predicted = torch.max(outputs, 1)
            total += val_labels.size(0)
            correct += (predicted.cpu() == val_labels).sum().item()
        val_acc = 100.0 * correct / total

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {history['train_loss'][-1]:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/linear_probe_best.pt")
            print(f"  -> New best model saved (val_acc={val_acc:.2f}%)")

    # Load best model
    model.load_state_dict(torch.load("checkpoints/linear_probe_best.pt", weights_only=True))
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    return model, history
