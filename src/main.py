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

def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, network, lam):
    n       = X.shape[1]
    n_batch = GDparams['n_batch']
    eta     = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    history = {
        "train_loss": [], "val_loss": [],
        "train_cost": [], "val_cost": [],
        "train_acc":  [], "val_acc":  []
    }

    for epoch in range(n_epochs):
        # shuffle indices each epoch
        idx = np.random.permutation(n)
        X, Y, y = X[:, idx], Y[:, idx], y[idx]

        for j in range(0, n, n_batch):
            X_batch = X[:, j:j+n_batch]
            Y_batch = Y[:, j:j+n_batch]

            fp      = ApplyNetwork(X_batch, network)
            grads   = BackwardPass(X_batch, Y_batch, fp, network, lam)

            # update parameters
            for i in range(2):
                network['W'][i] -= eta * grads['W'][i]
                network['b'][i] -= eta * grads['b'][i]

        # record metrics once per epoch
        fp_train = ApplyNetwork(X, network)
        fp_val   = ApplyNetwork(X_val, network)

        train_loss = ComputeLoss(fp_train['P'], y)
        val_loss   = ComputeLoss(fp_val['P'],   y_val)
        train_cost = ComputeCost(fp_train['P'], y,    network, lam)
        val_cost   = ComputeCost(fp_val['P'],   y_val, network, lam)
        train_acc  = ComputeAccuracy(fp_train['P'], y)
        val_acc    = ComputeAccuracy(fp_val['P'],   y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cost'].append(train_cost)
        history['val_cost'].append(val_cost)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
              f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

    return network, history


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "Datasets" / "cifar-10-python" / "cifar-10-batches-py"

    # load data
    trainX, trainY, trainy = LoadBatch(data_dir / "data_batch_1")
    mean_X = np.mean(trainX, axis=1, keepdims=True)
    std_X  = np.std(trainX,  axis=1, keepdims=True)
    trainX = NormalizeData(trainX, mean_X, std_X)

    # small network and small data for the check
    rng     = np.random.default_rng(42)
    d_small = 5
    n_small = 3
    m       = 6
    K       = 10

    small_net = InitNetwork(d_small, m, K, seed=42)

    X_small = trainX[0:d_small, 0:n_small]
    Y_small = trainY[:, 0:n_small]
    y_small = trainy[0:n_small]

    # compare gradients lam=0
    from torch_gradient_computations import ComputeGradsWithTorch
    fp      = ApplyNetwork(X_small, small_net)
    my_grads    = BackwardPass(X_small, Y_small, fp, small_net, lam=0.0)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

    print("-- Gradient check (lam=0) --")
    for i in range(2):
        print(f"  Layer {i+1} W: abs={MaxAbsoluteError(my_grads['W'][i], torch_grads['W'][i]):.2e} "
              f"rel={MaxRelativeError(my_grads['W'][i], torch_grads['W'][i]):.2e}")
        print(f"  Layer {i+1} b: abs={MaxAbsoluteError(my_grads['b'][i], torch_grads['b'][i]):.2e} "
              f"rel={MaxRelativeError(my_grads['b'][i], torch_grads['b'][i]):.2e}")
    
    print("\n-- Overfit sanity check (100 examples, lam=0) --")
    X_overfit = trainX[:, 0:100]
    Y_overfit = trainY[:, 0:100]
    y_overfit = trainy[0:100]

    overfit_net    = InitNetwork(d=trainX.shape[0], m=50, K=10, seed=42)
    overfit_params = {'n_batch': 10, 'eta': 0.01, 'n_epochs': 200}

    overfit_net, _ = MiniBatchGD(
        X_overfit, Y_overfit, y_overfit,
        X_overfit, Y_overfit, y_overfit,   # use same data for val
        overfit_params, overfit_net, lam=0.0
    )