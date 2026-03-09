"""
Fixed evaluation harness for SDF-GAN autoresearch.

This file is READ-ONLY — the agent must not modify it.
It contains data loading, loss functions, and evaluation metrics.

The ground truth metric is: valid_sharpe (monthly Sharpe ratio, higher is better).

Usage:
    from prepare import load_data, evaluate_all_splits, print_results
    from prepare import moment_loss, residual_loss, l1_penalty
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

UNK = -99.99
INDIVIDUAL_FEATURE_DIM = 46
MACRO_FEATURE_DIM = 178
N_TRAIN = 240
N_VALID = 60
N_TEST = 300

# Data paths relative to this file's directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "datasets")

CHAR_TRAIN = os.path.join(DATA_DIR, "char", "Char_train.npz")
CHAR_VALID = os.path.join(DATA_DIR, "char", "Char_valid.npz")
CHAR_TEST = os.path.join(DATA_DIR, "char", "Char_test.npz")
MACRO_TRAIN = os.path.join(DATA_DIR, "macro", "macro_train.npz")
MACRO_VALID = os.path.join(DATA_DIR, "macro", "macro_valid.npz")
MACRO_TEST = os.path.join(DATA_DIR, "macro", "macro_test.npz")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class AssetPricingDataset:
    """Loads and holds one split (train/valid/test) of asset pricing data."""

    def __init__(self, path_char, path_macro=None,
                 mean_macro=None, std_macro=None):
        tmp = np.load(path_char)
        data = tmp['data']
        self.R = data[:, :, 0]                         # (T, N)
        self.I = data[:, :, 1:]                        # (T, N, F)
        self.mask = (self.R != UNK)                    # (T, N)
        self.dates = list(tmp['date'])
        self.variables = list(tmp['variable'][1:])     # feature names (skip return)
        self.T, self.N, self.F = self.I.shape

        if path_macro is not None:
            tmp_m = np.load(path_macro)
            self.I_macro = tmp_m['data'].copy()
            self.F_macro = self.I_macro.shape[1]

            if mean_macro is None:
                self.mean_macro = self.I_macro.mean(axis=0)
                self.std_macro = self.I_macro.std(axis=0)
            else:
                self.mean_macro = mean_macro
                self.std_macro = std_macro
            self.I_macro = (self.I_macro - self.mean_macro) / self.std_macro
        else:
            self.I_macro = np.zeros((self.T, 0))
            self.F_macro = 0
            self.mean_macro = None
            self.std_macro = None

    def to_tensors(self, dev):
        return (
            torch.tensor(self.I_macro, dtype=torch.float32, device=dev),
            torch.tensor(self.I, dtype=torch.float32, device=dev),
            torch.tensor(self.R, dtype=torch.float32, device=dev),
            torch.tensor(self.mask, dtype=torch.bool, device=dev),
        )

    def loss_weight(self, dev):
        w = self.mask.sum(axis=0).astype(np.float32)
        return torch.tensor(w, device=dev)


def _check_data_exists():
    """Check that all required data files exist."""
    required = [CHAR_TRAIN, CHAR_VALID, CHAR_TEST,
                MACRO_TRAIN, MACRO_VALID, MACRO_TEST]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing data files:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        print(f"\nPlease place .npz files in: {DATA_DIR}/", file=sys.stderr)
        print("Expected structure:", file=sys.stderr)
        print("  datasets/char/Char_train.npz", file=sys.stderr)
        print("  datasets/char/Char_valid.npz", file=sys.stderr)
        print("  datasets/char/Char_test.npz", file=sys.stderr)
        print("  datasets/macro/macro_train.npz", file=sys.stderr)
        print("  datasets/macro/macro_valid.npz", file=sys.stderr)
        print("  datasets/macro/macro_test.npz", file=sys.stderr)
        sys.exit(1)


def load_data(device=None, weighted_loss=True):
    """Load all three splits. Returns a dict with tensors and metadata.

    Args:
        device: torch device (auto-selects cuda if available)
        weighted_loss: if True, compute per-stock loss weights

    Returns:
        dict with keys:
            train_tensors, valid_tensors, test_tensors: (I_macro, I_indiv, R, mask)
            lw_train, lw_valid, lw_test: loss weights (or None)
            variables: list of feature names
            device: the torch device used
    """
    _check_data_exists()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train = AssetPricingDataset(CHAR_TRAIN, MACRO_TRAIN)
    dl_valid = AssetPricingDataset(CHAR_VALID, MACRO_VALID,
                                   mean_macro=dl_train.mean_macro,
                                   std_macro=dl_train.std_macro)
    dl_test = AssetPricingDataset(CHAR_TEST, MACRO_TEST,
                                  mean_macro=dl_train.mean_macro,
                                  std_macro=dl_train.std_macro)

    print(f"Data loaded:")
    print(f"  Train: T={dl_train.T}, N={dl_train.N}, F_indiv={dl_train.F}, F_macro={dl_train.F_macro}")
    print(f"  Valid: T={dl_valid.T}, N={dl_valid.N}")
    print(f"  Test:  T={dl_test.T},  N={dl_test.N}")
    print(f"  Device: {device}")

    return {
        'train_tensors': dl_train.to_tensors(device),
        'valid_tensors': dl_valid.to_tensors(device),
        'test_tensors': dl_test.to_tensors(device),
        'lw_train': dl_train.loss_weight(device) if weighted_loss else None,
        'lw_valid': dl_valid.loss_weight(device) if weighted_loss else None,
        'lw_test': dl_test.loss_weight(device) if weighted_loss else None,
        'variables': dl_train.variables,
        'device': device,
    }


# ---------------------------------------------------------------------------
# Loss functions (importable by train.py, but ground truth lives here)
# ---------------------------------------------------------------------------


def moment_loss(R, mask, sdf, h, loss_weight=None):
    """Empirical moment condition loss.

    For unconditional: pass h = ones(1, T, N).
    For conditional:   pass h from MomentLayer (K, T, N).

    Loss = mean over K of [ mean over i of (time-avg of R*SDF*h)^2 ]
    """
    T_i = mask.float().sum(dim=0)                          # (N,)
    R_sdf = R * mask.float() * sdf                         # (T, N)
    emp_mean = (R_sdf.unsqueeze(0) * h).sum(dim=1) / T_i   # (K, N)
    sq = emp_mean.pow(2)

    if loss_weight is not None:
        w_norm = loss_weight / loss_weight.max()
        return (sq * w_norm.unsqueeze(0)).mean()
    return sq.mean()


def residual_loss(R, mask, w_flat):
    """Residual pricing error: mean(MSE(R - proj_w R)) / mean(MSE(R))."""
    R_flat = R[mask]
    N_i = mask.sum(dim=1)
    R_parts = torch.split(R_flat, N_i.tolist())
    w_parts = torch.split(w_flat, N_i.tolist())

    res_sq, r_sq = [], []
    for R_t, w_t in zip(R_parts, w_parts):
        coeff = (R_t * w_t).sum() / (w_t * w_t).sum()
        R_hat = coeff * w_t
        res_sq.append((R_t - R_hat).pow(2).mean())
        r_sq.append(R_t.pow(2).mean())
    return torch.stack(res_sq).mean() / torch.stack(r_sq).mean()


def l1_penalty(w_flat):
    """L1 penalty on stock weights for sparsity."""
    return w_flat.abs().mean()


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


def sharpe(r):
    """Monthly Sharpe ratio. r is a numpy array of monthly returns."""
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std())


@torch.no_grad()
def evaluate(model, data_tensors, lw=None, h0=None):
    """Evaluate loss and Sharpe on a single split (no gradients).

    Args:
        model: SDFGAN model (must have compute_weights_and_sdf method)
        data_tensors: (I_macro, I_indiv, R, mask)
        lw: loss weights (or None)
        h0: optional initial RNN hidden state (for chaining across splits)

    Returns:
        (metrics_dict, rnn_state)
    """
    model.eval()
    I_macro, I_indiv, R, mask = data_tensors
    T, N = R.shape

    w_flat, sdf, rnn_state = model.compute_weights_and_sdf(
        I_macro, I_indiv, R, mask, h0=h0)
    h_ones = torch.ones(1, T, N, device=R.device)

    loss = moment_loss(R, mask, sdf, h_ones, lw).item()
    res = residual_loss(R, mask, w_flat).item()
    ev = 1.0 - res  # explained variation

    portfolio = (1.0 - sdf[:, 0]).cpu().numpy()
    sr = sharpe(portfolio)

    model.train()
    return {'loss': loss, 'res_loss': res, 'ev': ev, 'sharpe': float(sr)}, rnn_state


def evaluate_all_splits(model, data):
    """Evaluate on train/valid/test with chained RNN states.

    Args:
        model: trained SDFGAN model
        data: dict from load_data()

    Returns:
        flat dict with all metrics
    """
    res_tr, h_tr = evaluate(model, data['train_tensors'], data['lw_train'])
    res_va, h_va = evaluate(model, data['valid_tensors'], data['lw_valid'], h0=h_tr)
    res_te, _ = evaluate(model, data['test_tensors'], data['lw_test'], h0=h_va)

    results = {}
    for split, res in [('train', res_tr), ('valid', res_va), ('test', res_te)]:
        for k, v in res.items():
            results[f'{split}_{k}'] = v
    return results


def print_results(results):
    """Print results in standardized, grep-friendly format."""
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print("---")
    print(f"valid_sharpe:     {results['valid_sharpe']:.6f}")
    print(f"test_sharpe:      {results['test_sharpe']:.6f}")
    print(f"train_sharpe:     {results['train_sharpe']:.6f}")
    print(f"valid_loss:       {results['valid_loss']:.6f}")
    print(f"test_loss:        {results['test_loss']:.6f}")
    print(f"train_loss:       {results['train_loss']:.6f}")
    print(f"valid_ev:         {results['valid_ev']:.6f}")
    print(f"test_ev:          {results['test_ev']:.6f}")
    print(f"train_ev:         {results['train_ev']:.6f}")
    print(f"peak_vram_mb:     {peak_mem:.1f}")


# ---------------------------------------------------------------------------
# Main: verify data is loadable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Verifying data setup...")
    _check_data_exists()
    print("All data files found. Ready to train.")
