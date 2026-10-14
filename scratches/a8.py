import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader, TensorDataset


class MnistMLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_b, y_b in loader:
            logits = model(x_b)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    return (correct / total) * 100


def main() -> None:
    X_raw, y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X = X_raw.astype(np.float32) / 255.0
    y = y_raw.astype(np.int64)

    X_train, y_train = torch.tensor(X[:60000]), torch.tensor(y[:60000])
    X_test, y_test = torch.tensor(X[60000:]), torch.tensor(y[60000:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=64, shuffle=False
    )

    model = MnistMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    criterion = nn.CrossEntropyLoss()
    epochs = 50
    print("Training...")

    for epoch in range(epochs):
        model.train()
        loss = 0.0
        preds = 0.0
        samples = 0

        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            local_loss = criterion.forward(logits, yb)
            local_loss.backward()
            opt.step()

            loss += local_loss.item() * xb.size(0)
            preds += (torch.argmax(logits, dim=1) == yb).sum().item()
            samples += yb.size(0)

        train_loss = loss / samples
        train_acc = (preds / samples) * 100
        test_acc = evaluate(model, test_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )


if __name__ == "__main__":
    main()
