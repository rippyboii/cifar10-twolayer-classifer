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

def ApplyNetwork(X, network):
    """
    X: (d, n)
    Returns fp_data dict with all intermediate values
    """
    W1, b1 = network['W'][0], network['b'][0]
    W2, b2 = network['W'][1], network['b'][1]
    
    # Layer 1
    s1 = W1 @ X + b1          # (m, n)
    h  = np.maximum(0, s1)    # ReLU  (m, n)
    
    # Layer 2
    s  = W2 @ h + b2          # (K, n)
    P  = softmax(s)            # (K, n)
    
    # store everything — backward pass will need s1 and h
    fp_data = {
        's1': s1,
        'h':  h,
        's':  s,
        'P':  P
    }
    
    return fp_data

def BackwardPass(X, Y, fp_data, network, lam):
    """
    X       : (d, n)   input images
    Y       : (K, n)   one-hot true labels
    fp_data : dict     from ApplyNetwork (has P, h, s1)
    network : dict     current parameters
    lam     : float    regularization strength
    """
    n  = X.shape[1]
    P  = fp_data['P']
    h  = fp_data['h']
    s1 = fp_data['s1']
    W2 = network['W'][1]
    W1 = network['W'][0]

    # error at the ouitput layer
    G = P - Y                                        # (K, n)

    # gradients for 2nd layer
    grad_W2 = (G @ h.T) / n + 2 * lam * W2          # (K, m)
    grad_b2 = np.sum(G, axis=1, keepdims=True) / n  # (K, 1)

    G = W2.T @ G                                     # (m, n)
    G = G * (s1 > 0)                                 # ReLU gate (m, n)

    # gradients for 1st layer
    grad_W1 = (G @ X.T) / n + 2 * lam * W1          # (m, d)
    grad_b1 = np.sum(G, axis=1, keepdims=True) / n  # (m, 1)

    grads = {
        'W': [grad_W1, grad_W2],
        'b': [grad_b1, grad_b2]
    }

    return grads

def ComputeLoss(P, y):
    n = P.shape[1]
    p_correct = P[y, np.arange(n)]
    L = -np.mean(np.log(p_correct))
    return L

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    reg = lam * (np.sum(network['W'][0]**2) + np.sum(network['W'][1]**2))
    return loss + reg



