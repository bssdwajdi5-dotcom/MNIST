"""Train a CNN on MNIST and report test accuracy.

Usage:
    python3 train.py [--epochs 5] [--batch-size 128] [--lr 1e-3] [--out checkpoints/mnist_cnn.pt]
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_CLASSES = 10


class MNISTNet(nn.Module):
    """2-conv CNN: 1x28x28 -> 10 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28 -> 14
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(self.dropout(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=eval_tf)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/mnist_cnn.pt"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    train_loader, test_loader = build_loaders(args.batch_size)
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "seconds": elapsed,
            }
        )
        print(
            f"epoch {epoch}/{args.epochs}  loss={loss:.4f}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  ({elapsed:.1f}s)"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "args": vars(args) | {"out": str(args.out)},
                },
                args.out,
            )

    total = time.time() - start
    print(f"best test accuracy: {best_acc:.4f}  |  total {total:.1f}s")
    print(f"checkpoint saved to {args.out}")

    history_path = args.out.with_suffix(".history.json")
    history_path.write_text(
        json.dumps({"best_test_acc": best_acc, "history": history}, indent=2)
    )
    print(f"history saved to {history_path}")


if __name__ == "__main__":
    main()
