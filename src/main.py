import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].astype(np.float64) / 255.0
    X = X.T

    y = np.array(batch[b'labels'])
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
    steps = history["steps"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(steps, history["train_cost"], label="train")
    axs[0].plot(steps, history["val_cost"],   label="val")
    axs[0].set_xlabel("Update step")
    axs[0].set_ylabel("Cost")
    axs[0].set_title("Cost")
    axs[0].legend()

    axs[1].plot(steps, history["train_loss"], label="train")
    axs[1].plot(steps, history["val_loss"],   label="val")
    axs[1].set_xlabel("Update step")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss")
    axs[1].legend()

    axs[2].plot(steps, history["train_acc"], label="train")
    axs[2].plot(steps, history["val_acc"],   label="val")
    axs[2].set_xlabel("Update step")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Accuracy")
    axs[2].legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

def MaxAbsoluteError(a, b):
    return np.max(np.abs(a - b))

def MaxRelativeError(a, b, eps=1e-10):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))

def InitNetwork(d, m, K, seed=42):
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
    W1, b1 = network['W'][0], network['b'][0]
    W2, b2 = network['W'][1], network['b'][1]

    s1 = W1 @ X + b1          # (m, n)
    h  = np.maximum(0, s1)    # ReLU
    s  = W2 @ h + b2          # (K, n)
    P  = softmax(s)            # (K, n)

    fp_data = {'s1': s1, 'h': h, 's': s, 'P': P}
    return fp_data

def BackwardPass(X, Y, fp_data, network, lam):
    n  = X.shape[1]
    P  = fp_data['P']
    h  = fp_data['h']
    s1 = fp_data['s1']
    W2 = network['W'][1]
    W1 = network['W'][0]

    # error at output layer
    G = P - Y                                        # (K, n)

    # gradients for layer 2
    grad_W2 = (G @ h.T) / n + 2 * lam * W2
    grad_b2 = np.sum(G, axis=1, keepdims=True) / n

    # propagate back through W2 and ReLU
    G = W2.T @ G
    G = G * (s1 > 0)                                 # ReLU gate

    # gradients for layer 1
    grad_W1 = (G @ X.T) / n + 2 * lam * W1
    grad_b1 = np.sum(G, axis=1, keepdims=True) / n

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
    reg  = lam * (np.sum(network['W'][0]**2) + np.sum(network['W'][1]**2))
    return loss + reg

def CyclicLearningRate(t, eta_min, eta_max, n_s):
    cycle = t // (2 * n_s)
    pos   = t - 2 * cycle * n_s

    if pos < n_s:
        eta = eta_min + (pos / n_s) * (eta_max - eta_min)
    else:
        eta = eta_max - ((pos - n_s) / n_s) * (eta_max - eta_min)

    return eta

def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, network, lam):
    n        = X.shape[1]
    n_batch  = GDparams['n_batch']
    eta_min  = GDparams['eta_min']
    eta_max  = GDparams['eta_max']
    n_s      = GDparams['n_s']
    n_cycles = GDparams['n_cycles']

    total_steps = 2 * n_s * n_cycles
    log_every   = max(1, 2 * n_s // 10)   # log 10 times per cycle

    history = {
        "train_loss": [], "val_loss": [],
        "train_cost": [], "val_cost": [],
        "train_acc":  [], "val_acc":  [],
        "steps":      []
    }

    t = 0  # global update step counter

    while t < total_steps:
        idx    = np.random.permutation(n)
        X_shuf = X[:, idx]
        Y_shuf = Y[:, idx]
        y_shuf = y[idx]

        for j in range(0, n, n_batch):
            if t >= total_steps:
                break

            X_batch = X_shuf[:, j:j+n_batch]
            Y_batch = Y_shuf[:, j:j+n_batch]

            eta   = CyclicLearningRate(t, eta_min, eta_max, n_s)
            fp    = ApplyNetwork(X_batch, network)
            grads = BackwardPass(X_batch, Y_batch, fp, network, lam)

            for i in range(2):
                network['W'][i] -= eta * grads['W'][i]
                network['b'][i] -= eta * grads['b'][i]

            if t % log_every == 0:
                fp_train = ApplyNetwork(X, network)
                fp_val   = ApplyNetwork(X_val, network)

                train_loss = ComputeLoss(fp_train['P'], y)
                val_loss   = ComputeLoss(fp_val['P'],   y_val)
                train_cost = ComputeCost(fp_train['P'], y,     network, lam)
                val_cost   = ComputeCost(fp_val['P'],   y_val, network, lam)
                train_acc  = ComputeAccuracy(fp_train['P'], y)
                val_acc    = ComputeAccuracy(fp_val['P'],   y_val)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_cost'].append(train_cost)
                history['val_cost'].append(val_cost)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['steps'].append(t)

                print(f"  step {t:4d}/{total_steps} | "
                      f"loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"acc: {train_acc:.4f}/{val_acc:.4f} | "
                      f"eta: {eta:.6f}")

            t += 1

    return network, history


if __name__ == "__main__":
    ROOT     = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "Datasets" / "cifar-10-python" / "cifar-10-batches-py"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # --- Load and normalize training data ---
    trainX, trainY, trainy = LoadBatch(data_dir / "data_batch_1")
    mean_X = np.mean(trainX, axis=1, keepdims=True)
    std_X  = np.std(trainX,  axis=1, keepdims=True)
    trainX = NormalizeData(trainX, mean_X, std_X)

    valX, valY, valy = LoadBatch(data_dir / "data_batch_2")
    valX = NormalizeData(valX, mean_X, std_X)

    # --- Gradient check ---
    d_small, n_small, m, K = 5, 3, 6, 10
    small_net = InitNetwork(d_small, m, K, seed=42)
    X_small   = trainX[0:d_small, 0:n_small]
    Y_small   = trainY[:, 0:n_small]
    y_small   = trainy[0:n_small]

    from torch_gradient_computations import ComputeGradsWithTorch
    fp          = ApplyNetwork(X_small, small_net)
    my_grads    = BackwardPass(X_small, Y_small, fp, small_net, lam=0.0)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

    print("-- Gradient check (lam=0) --")
    for i in range(2):
        print(f"  Layer {i+1} W: abs={MaxAbsoluteError(my_grads['W'][i], torch_grads['W'][i]):.2e} "
              f"rel={MaxRelativeError(my_grads['W'][i], torch_grads['W'][i]):.2e}")
        print(f"  Layer {i+1} b: abs={MaxAbsoluteError(my_grads['b'][i], torch_grads['b'][i]):.2e} "
              f"rel={MaxRelativeError(my_grads['b'][i], torch_grads['b'][i]):.2e}")

    # --- Overfit sanity check ---
    print("\n-- Overfit sanity check (100 examples, lam=0) --")
    X_overfit = trainX[:, 0:100]
    Y_overfit = trainY[:, 0:100]
    y_overfit = trainy[0:100]

    overfit_net    = InitNetwork(d=trainX.shape[0], m=50, K=10, seed=42)
    overfit_params = {
        'n_batch':  10,
        'eta_min':  0.01,
        'eta_max':  0.01,   # fixed LR (min == max)
        'n_s':      500,
        'n_cycles': 4
    }
    overfit_net, _ = MiniBatchGD(
        X_overfit, Y_overfit, y_overfit,
        X_overfit, Y_overfit, y_overfit,
        overfit_params, overfit_net, lam=0.0
    )

    # --- Exercise 3: 1 cycle, replicate Figure 3 ---
    print("\n-- Exercise 3: 1 cycle (replicating Figure 3) --")
    net_ex3    = InitNetwork(d=trainX.shape[0], m=50, K=10, seed=42)
    params_ex3 = {
        'n_batch':  100,
        'eta_min':  1e-5,
        'eta_max':  1e-1,
        'n_s':      500,
        'n_cycles': 1
    }
    net_ex3, hist_ex3 = MiniBatchGD(
        trainX, trainY, trainy,
        valX,   valY,   valy,
        params_ex3, net_ex3, lam=0.01
    )
    PlotHistory(hist_ex3, title="Exercise 3: 1 cycle",
                save_path=figures_dir / "ex3_one_cycle.png")
    
    # --- Exercise 4: 3 cycles, replicate Figure 4 ---
    print("\n-- Exercise 4: 3 cycles (replicating Figure 4) --")
    net_ex4    = InitNetwork(d=trainX.shape[0], m=50, K=10, seed=42)
    params_ex4 = {
        'n_batch':  100,
        'eta_min':  1e-5,
        'eta_max':  1e-1,
        'n_s':      800,
        'n_cycles': 3
    }
    net_ex4, hist_ex4 = MiniBatchGD(
        trainX, trainY, trainy,
        valX,   valY,   valy,
        params_ex4, net_ex4, lam=0.01
    )

    # evaluate on test set
    testX, testY, testy = LoadBatch(data_dir / "test_batch")
    testX = NormalizeData(testX, mean_X, std_X)
    fp_test  = ApplyNetwork(testX, net_ex4)
    test_acc = ComputeAccuracy(fp_test['P'], testy)
    print(f"  Test accuracy after 3 cycles: {test_acc*100:.2f}%")

    PlotHistory(hist_ex4, title="Exercise 4: 3 cycles",
                save_path=figures_dir / "ex4_three_cycles.png")
    
    # --- Coarse lamda search ---
    print("\n-- Coarse lambda search --")

    # load all 5 batches
    all_X, all_Y, all_y = [], [], []
    for i in range(1, 6):
        Xi, Yi, yi = LoadBatch(data_dir / f"data_batch_{i}")
        all_X.append(Xi)
        all_Y.append(Yi)
        all_y.append(yi)
    all_X = np.concatenate(all_X, axis=1)  # (3072, 50000)
    all_Y = np.concatenate(all_Y, axis=1)  # (10, 50000)
    all_y = np.concatenate(all_y, axis=0)  # (50000,)

    # normalize using training set stats
    mean_all = np.mean(all_X, axis=1, keepdims=True)
    std_all  = np.std(all_X,  axis=1, keepdims=True)
    all_X    = NormalizeData(all_X, mean_all, std_all)

    # split: last 5000 = val, rest = train
    big_trainX = all_X[:, :-5000]
    big_trainY = all_Y[:, :-5000]
    big_trainy = all_y[:-5000]
    big_valX   = all_X[:, -5000:]
    big_valY   = all_Y[:, -5000:]
    big_valy   = all_y[-5000:]

    d    = big_trainX.shape[0]
    n    = big_trainX.shape[1]

    # 8 values evenly on log scale from 1e-5 to 1e-1
    l_min, l_max = -5, -1
    lam_values = [10 ** (l_min + (l_max - l_min) * i / 7) for i in range(8)]

    n_batch = 100
    n_s_coarse = int(2 * np.floor(n / n_batch))  # as per PDF formula

    coarse_params = {
        'n_batch':  n_batch,
        'eta_min':  1e-5,
        'eta_max':  1e-1,
        'n_s':      n_s_coarse,
        'n_cycles': 2
    }

    coarse_results = []

    for lam in lam_values:
        print(f"\n  lam={lam:.2e}")
        net = InitNetwork(d=d, m=50, K=10, seed=42)
        net, hist = MiniBatchGD(
            big_trainX, big_trainY, big_trainy,
            big_valX,   big_valY,   big_valy,
            coarse_params, net, lam
        )
        best_val_acc = max(hist['val_acc'])
        final_val_acc = hist['val_acc'][-1]
        print(f"  lam={lam:.2e} | best val acc: {best_val_acc*100:.2f}% | final val acc: {final_val_acc*100:.2f}%")
        coarse_results.append((lam, best_val_acc, final_val_acc))

    print("\n-- Coarse search summary --")
    for lam, best, final in coarse_results:
        print(f"  lam={lam:.2e} | best={best*100:.2f}% | final={final*100:.2f}%")