import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import time, sys

import torch.nn.functional as F
import torch.nn as nn

from models_mnist import MnistModel
from utils import get_default_device


def split_indices(n, val_pct):
    n_val = int(n * val_pct)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_func, valid_dl, metric=None):
    with torch.no_grad():
        results = [
            loss_batch(model, loss_func, xb, yb, metric=metric) for xb, yb in valid_dl
        ]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def fit(epochs, lr, model, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None):
    losses, metrics = [], []

    if opt_fn is None:
        opt_fn = torch.optim.SGD
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        losses.append(val_loss)
        metrics.append(val_metric)

        if metric is None:
            print(f"Epoch [{epoch}/{epochs}], Loss: {val_loss}")
        else:
            print(
                f"Epoch [{epoch}/{epochs}], Loss: {val_loss}, {metric.__name__}: {val_metric}"
            )

    return losses, metrics


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


def train_model(input_size, hidden_size, num_layers):
    dataset = MNIST(root="data/", download=True, transform=ToTensor())
    dataset = [(x[0][0].view(784), x[1]) for x in dataset]

    train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)
    print(len(train_indices), len(val_indices))

    batch_size = 100
    device = get_default_device()
    print("device:", device)

    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)
    train_dl = DeviceDataLoader(train_dl, device)

    valid_sampler = SubsetRandomSampler(val_indices)
    valid_dl = DataLoader(dataset, batch_size, sampler=valid_sampler)
    valid_dl = DeviceDataLoader(valid_dl, device)

    num_classes = 10
    model = MnistModel(input_size, hidden_size, num_layers, out_size=num_classes).to(
        device
    )

    losses, metrics = fit(5, 0.5, model, F.cross_entropy, train_dl, valid_dl, accuracy)

    return model


if __name__ == "__main__":
    input_size = int(28 * 28 / int(sys.argv[1]))
    hidden_size = int(sys.argv[2])
    num_layers = int(sys.argv[3])
    model_name = sys.argv[4]
    model = train_model(input_size, hidden_size, num_layers)

    torch.save(model.state_dict(), f"saved/{model_name}")
    print(f"The model is saved in ./saved/{model_name}")
