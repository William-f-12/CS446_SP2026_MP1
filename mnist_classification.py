import argparse
import gzip
import os
import struct
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mini_torch as torch
import mini_torch.nn as nn
from mini_torch import Tensor
from mini_torch.nn import CrossEntropyLoss
from mini_torch.optim import SGD
from mini_torch.utils.data import DataLoader, TensorDataset

# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from torch.utils.data import DataLoader, TensorDataset

# =========================
# MNIST loader
# =========================


def read_idx_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def read_idx_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n)


def load_mnist_local(root="data/mnist"):
    trX = read_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    trY = read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz"))
    teX = read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    teY = read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
    return trX, trY, teX, teY


# =========================
# Utils
# =========================


def one_hot(y: np.ndarray, num_classes=10) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
    return out


def accuracy_from_logits(logits: np.ndarray, y_int: np.ndarray) -> float:
    pred = logits.argmax(axis=1)
    return float((pred == y_int).mean())


# ============================================================
# Neural network (model definition)
# ============================================================
class MNISTMLP(nn.Module):
    # You can pass in any arguments you want to the constructor.
    def __init__(
        self,
    ):
        super().__init__()
        # first hidden layer
        self.linear1 = nn.Linear(784, 2048)
        self.activation1 = nn.ReLU()
        # second hidden layer
        self.linear2 = nn.Linear(2048, 1024)
        self.activation2 = nn.ReLU()
        # output layer
        self.linear3 = nn.Linear(1024, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x


def main():
    np.random.seed(0)

    Xtr_img, ytr, Xte_img, yte = load_mnist_local("data/mnist")

    Xtr = Xtr_img.reshape(-1, 784).astype(np.float32) / 255.0
    Xte = Xte_img.reshape(-1, 784).astype(np.float32) / 255.0

    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    ytr_oh = one_hot(ytr, 10)
    yte_oh = one_hot(yte, 10)

    Xtr_t = torch.tensor(Xtr, requires_grad=False)
    ytr_t = torch.tensor(ytr_oh, requires_grad=False)
    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    # ============================================================
    # instantiate model / loss / optimizer
    # ============================================================
    model = MNISTMLP()
    criterion = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.003)

    Xte_t = torch.tensor(Xte, requires_grad=False)
    yte_t = torch.tensor(yte_oh, requires_grad=False)

    epochs = 20  # you can change this if you want

    last_te_acc = 0.0  # will be updated each epoch

    # ============================================================
    # training loop
    # ============================================================
    for ep in range(1, epochs + 1):
        for xb, yb in train_dl:
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        te_logits = model(Xte_t).detach().numpy()
        te_acc = accuracy_from_logits(te_logits, yte)

        last_te_acc = te_acc

        shifted = te_logits - np.max(te_logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        te_loss = -np.mean(np.sum(yte_oh * log_probs, axis=1))

        print(
            f"Epoch {ep:02d}/{epochs} | test_loss={te_loss:.4f} | test_acc={te_acc:.4f}"
        )

    return last_te_acc


if __name__ == "__main__":
    main()
