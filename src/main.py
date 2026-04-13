import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].astype(np.float64) / 255.0   # (10000, 3072)
    X = X.T                                         # (3072, 10000)

    y = np.array(batch[b'labels'])                  # (10000,)
    K = 10
    n = X.shape[1]

    Y = np.zeros((K, n), dtype=np.float64)
    Y[y, np.arange(n)] = 1

    return X, Y, y

def NormalizeData(X, mean_X, std_X):
    return (X - mean_X) / std_X

def softmax(s):
    s_shifted = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shifted)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)

def ComputeAccuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    acc = np.mean(y_pred == y)
    return acc

def PlotHistory(history, title="", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Cost
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_cost"], label="train cost")
    plt.plot(epochs, history["val_cost"], label="val cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()

def MaxAbsoluteError(a, b):
    return np.max(np.abs(a - b))


def MaxRelativeError(a, b, eps=1e-10):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))


def InitNetwork(d, m, K, seed=42):
    """
    d = input dimension (3072 for CIFAR-10)
    m = hidden layer size (50 by default)
    K = number of classes (10)
    """
    rng = np.random.default_rng(seed)
    
    network = {}
    network['W'] = [None] * 2
    network['b'] = [None] * 2
    
    # Layer 1: (m x d), std = 1/sqrt(d)
    network['W'][0] = rng.standard_normal((m, d)) / np.sqrt(d)
    network['b'][0] = np.zeros((m, 1))
    
    # Layer 2: (K x m), std = 1/sqrt(m)
    network['W'][1] = rng.standard_normal((K, m)) / np.sqrt(m)
    network['b'][1] = np.zeros((K, 1))
    
    return network