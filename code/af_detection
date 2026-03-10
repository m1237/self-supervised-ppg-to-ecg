"""
AF Detection on MIMIC PERform AF-style segmented ECG data
=========================================================

This script implements the AF detection pipeline described in the task text.

Data format
--------------------

real_segments:      shape (N, T, 1)   # real ECG segments
generated_segments: shape (N, T, 1)   # generated ECG segments matched to same samples
labels:             shape (N,)        # binary labels: 1=AF, 0=non-AF
subject_ids:        shape (N,)        # subject id for each segment


---------------------
1. Segments are already 30-second ECG windows.
2. All segment arrays are aligned sample-wise:
      real_segments[i], generated_segments[i], labels[i], subject_ids[i]
   must correspond to the same segment/subject.
3. Subject-independent split is performed at the subject level.
4. The grid-search space from "Table 1".
"""

from __future__ import annotations

import os
import copy
import json
import math
import random
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic mode can slow training a little
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # General
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./af_detection_outputs"

    # Split
    train_subject_ratio: float = 0.68
    val_ratio_within_train: float = 0.10

    # Label meaning
    af_label: int = 1
    non_af_label: int = 0

    # Training
    batch_size: int = 128
    bidlstm_epochs: int = 70
    cnnlstm_epochs: int = 110
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # Segment
    n_features: int = 1  # ECG single-channel
    # time_window inferred from data

    # Model architecture from text
    lstm_hidden_size: int = 128
    fc_hidden_size: int = 256
    dropout: float = 0.5
    num_classes: int = 2

    # CNN + LSTM
    conv_kernel_size: int = 4
    pool_size: int = 2
    pool_stride: int = 2
    conv_filters: int = 128   # chosen to match downstream LSTM input size

    # Grid search
    grid_search: bool = True

    # Mixing ratios: proportion of generated ECG used in training,
    # while total training size remains constant
    generated_mix_ratios: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)

    # Class imbalance
    use_class_weights: bool = True

    # Early stopping
    early_stopping_patience: int = 12


CFG = Config()
os.makedirs(CFG.output_dir, exist_ok=True)


# ============================================================
# Dataset utilities
# ============================================================

class ECGSegmentDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        x: (N, T, C)
        y: (N,)
        """
        assert x.ndim == 3, f"Expected x shape (N,T,C), got {x.shape}"
        assert y.ndim == 1, f"Expected y shape (N,), got {y.shape}"
        assert len(x) == len(y), "x and y length mismatch"

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def minmax_normalize_per_segment(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Min-max normalization independently for each segment and channel.
    x: (N, T, C)
    """
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


def compute_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    freq = {c: n for c, n in zip(classes, counts)}
    total = len(y)
    weights = []
    for c in [0, 1]:
        count = freq.get(c, 1)
        weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ============================================================
# Subject-level split
# ============================================================

def subject_level_split(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    train_subject_ratio: float = 0.68,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns train_subjects, test_subjects

    Tries to preserve AF / non-AF distribution at the subject level.
    Assumes each subject has a unique class label across its segments.
    """
    unique_subjects = np.unique(subject_ids)

    subject_to_label = {}
    for sid in unique_subjects:
        sid_labels = np.unique(labels[subject_ids == sid])
        if len(sid_labels) != 1:
            raise ValueError(
                f"Subject {sid} has multiple labels across segments: {sid_labels}. "
                "Expected one subject-level AF/non-AF label."
            )
        subject_to_label[sid] = sid_labels[0]

    af_subjects = np.array([sid for sid in unique_subjects if subject_to_label[sid] == 1])
    non_af_subjects = np.array([sid for sid in unique_subjects if subject_to_label[sid] == 0])

    rng = np.random.RandomState(seed)
    rng.shuffle(af_subjects)
    rng.shuffle(non_af_subjects)

    n_af_train = max(1, int(round(len(af_subjects) * train_subject_ratio)))
    n_non_af_train = max(1, int(round(len(non_af_subjects) * train_subject_ratio)))

    train_subjects = np.concatenate([af_subjects[:n_af_train], non_af_subjects[:n_non_af_train]])
    test_subjects = np.concatenate([af_subjects[n_af_train:], non_af_subjects[n_non_af_train:]])

    rng.shuffle(train_subjects)
    rng.shuffle(test_subjects)

    return train_subjects, test_subjects


def train_val_split_by_subjects(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    train_subjects: np.ndarray,
    val_ratio_within_train: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split training subjects further into train/validation subjects.
    """
    train_subjects = np.asarray(train_subjects)

    subject_to_label = {}
    for sid in train_subjects:
        sid_labels = np.unique(labels[subject_ids == sid])
        if len(sid_labels) != 1:
            raise ValueError(f"Subject {sid} has inconsistent labels.")
        subject_to_label[sid] = sid_labels[0]

    af_subjects = np.array([sid for sid in train_subjects if subject_to_label[sid] == 1])
    non_af_subjects = np.array([sid for sid in train_subjects if subject_to_label[sid] == 0])

    rng = np.random.RandomState(seed)
    rng.shuffle(af_subjects)
    rng.shuffle(non_af_subjects)

    n_af_val = max(1, int(round(len(af_subjects) * val_ratio_within_train))) if len(af_subjects) > 1 else 0
    n_non_af_val = max(1, int(round(len(non_af_subjects) * val_ratio_within_train))) if len(non_af_subjects) > 1 else 0

    val_subjects = np.concatenate([af_subjects[:n_af_val], non_af_subjects[:n_non_af_val]])
    final_train_subjects = np.concatenate([af_subjects[n_af_val:], non_af_subjects[n_non_af_val:]])

    rng.shuffle(val_subjects)
    rng.shuffle(final_train_subjects)

    return final_train_subjects, val_subjects


def indices_from_subjects(subject_ids: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    return np.where(np.isin(subject_ids, subjects))[0]


# ============================================================
# Mixing real and generated ECG in training
# ============================================================

def build_mixed_training_data(
    real_x: np.ndarray,
    gen_x: np.ndarray,
    y: np.ndarray,
    generated_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maintain constant training size.
    Replace a proportion of real ECG segments with generated ECG segments
    for the same samples.

    generated_ratio = 0.0  -> all real
    generated_ratio = 1.0  -> all generated
    """
    assert real_x.shape == gen_x.shape
    assert len(real_x) == len(y)

    n = len(y)
    n_generated = int(round(n * generated_ratio))

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    gen_indices = indices[:n_generated]
    real_indices = indices[n_generated:]

    x_mixed = np.empty_like(real_x)
    x_mixed[real_indices] = real_x[real_indices]
    x_mixed[gen_indices] = gen_x[gen_indices]

    return x_mixed, y.copy()


# ============================================================
# Models
# ============================================================

class BidirectionalLSTMClassifier(nn.Module):
    """
    - two LSTM layers with 128 hidden units each
    - bidirectional
    - dropout 0.5
    - two fully connected layers with 256 units
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        fc_hidden_size: int = 256,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.out = nn.Linear(fc_hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x, _ = self.lstm1(x)  # (B, T, 2H)
        x, _ = self.lstm2(x)  # (B, T, 2H)

        # final time step representation
        x = x[:, -1, :]       # (B, 2H)
        x = self.dropout(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x


class CNNLSTMClassifier(nn.Module):
    """
    - 2 Conv1D layers
    - 2 max-pooling layers
    - dropout
    - then same LSTM/FC block as Bid-LSTM
    """

    def __init__(
        self,
        input_size: int = 1,
        conv_filters: int = 128,
        kernel_size: int = 4,
        pool_size: int = 2,
        hidden_size: int = 128,
        fc_hidden_size: int = 256,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=2)

        self.conv_dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=conv_filters,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.out = nn.Linear(fc_hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x: (B, T, C)
        x = x.transpose(1, 2)          # -> (B, C, T)

        x = self.relu(self.conv1(x))   # -> (B, F, T)
        x = self.pool1(x)              # -> (B, F, T/2)

        x = self.relu(self.conv2(x))   # -> (B, F, T/2)
        x = self.pool2(x)              # -> (B, F, T/4)

        x = self.conv_dropout(x)
        x = x.transpose(1, 2)          # -> (B, T_reduced, F)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]
        x = self.lstm_dropout(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x


# ============================================================
# Training / evaluation
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.best_state = None

    def step(self, value: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        """
        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == "max" and value > self.best_value:
            improved = True
        elif self.mode == "min" and value < self.best_value:
            improved = True

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience


def create_model(model_name: str, params: Dict[str, Any]) -> nn.Module:
    if model_name == "bidlstm":
        return BidirectionalLSTMClassifier(**params)
    elif model_name == "cnn_lstm":
        return CNNLSTMClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str,
) -> Dict[str, Any]:
    model.eval()

    running_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_prob": np.array(y_prob),
    }
    return metrics


def fit_model(
    model_name: str,
    model_params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Any]]:
    model = create_model(model_name, model_params).to(cfg.device)

    # loss
    if cfg.use_class_weights:
        train_labels = []
        for _, y in train_loader:
            train_labels.extend(y.numpy())
        class_weights = compute_class_weights(np.array(train_labels), cfg.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_params.get("learning_rate", cfg.learning_rate),
        weight_decay=model_params.get("weight_decay", cfg.weight_decay),
    )

    epochs = cfg.bidlstm_epochs if model_name == "bidlstm" else cfg.cnnlstm_epochs

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_metrics = evaluate_model(model, val_loader, criterion, cfg.device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        should_stop = stopper.step(val_metrics["f1"], model)

        print(
            f"[{model_name}] Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    best_val_metrics = evaluate_model(model, val_loader, criterion, cfg.device)
    return model, history, best_val_metrics


# ============================================================
# Grid search
# ============================================================

def get_grid_search_space(model_name: str, cfg: Config) -> List[Dict[str, Any]]:

    if model_name == "bidlstm":
        grid = {
            "input_size": [cfg.n_features],
            "hidden_size": [128],
            "fc_hidden_size": [256],
            "dropout": [0.2, 0.4, 0.5, 0.7],
            "num_classes": [2],
            "learning_rate": [1e-3, 2e-3, 3e-3],
            "weight_decay": [0.5, 0.9, 1, 3],
        }
    elif model_name == "cnn_lstm":
        grid = {
            "input_size": [cfg.n_features],
            "conv_filters": [64, 128],
            "kernel_size": [4],
            "pool_size": [2],
            "hidden_size": [128],
            "fc_hidden_size": [256],
            "dropout": [0.2, 0.4, 0.5, 0.7],
            "num_classes": [2],
            "learning_rate": [1e-3, 2e-3, 3e-3],
            "weight_decay": [0.5, 0.9, 1, 3],
        }
    else:
        raise ValueError(model_name)

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations


def run_grid_search(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, List[float]]]:
    search_space = get_grid_search_space(model_name, cfg) if cfg.grid_search else [get_grid_search_space(model_name, cfg)[0]]

    best_model = None
    best_params = None
    best_metrics = None
    best_history = None
    best_f1 = -1.0

    for i, params in enumerate(search_space, 1):
        print("=" * 80)
        print(f"Grid Search [{model_name}] {i}/{len(search_space)}")
        print(f"Params: {params}")

        # separate optimizer-only params from model init params
        model_init_params = params.copy()
        model_init_params.pop("learning_rate", None)
        model_init_params.pop("weight_decay", None)

        model, history, val_metrics = fit_model(
            model_name=model_name,
            model_params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model = model
            best_params = params
            best_metrics = val_metrics
            best_history = history

    print("=" * 80)
    print(f"Best [{model_name}] params: {best_params}")
    print(f"Best [{model_name}] validation F1: {best_metrics['f1']:.4f}")

    return best_model, best_params, best_metrics, best_history


# ============================================================
# Plotting
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, class_names=("Non-AF", "AF"), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.tight_layout()
    return fig


def save_json(obj: Dict[str, Any], filepath: str) -> None:
    serializable = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, np.generic):
            serializable[k] = v.item()
        else:
            serializable[k] = v
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


# ============================================================
# Experiment runner
# ============================================================

def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = ECGSegmentDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def experiment_single_model(
    model_name: str,
    real_segments: np.ndarray,
    generated_segments: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Runs:
    1. subject-independent split
    2. train/val split inside train subjects
    3. real/generated training mix experiments
    4. testing on real ECG
    5. best config tested on generated ECG
    """

    # Normalize both real and generated segments
    real_segments = minmax_normalize_per_segment(real_segments)
    generated_segments = minmax_normalize_per_segment(generated_segments)

    train_subjects, test_subjects = subject_level_split(
        subject_ids=subject_ids,
        labels=labels,
        train_subject_ratio=cfg.train_subject_ratio,
        seed=cfg.seed,
    )

    final_train_subjects, val_subjects = train_val_split_by_subjects(
        subject_ids=subject_ids,
        labels=labels,
        train_subjects=train_subjects,
        val_ratio_within_train=cfg.val_ratio_within_train,
        seed=cfg.seed,
    )

    train_idx = indices_from_subjects(subject_ids, final_train_subjects)
    val_idx = indices_from_subjects(subject_ids, val_subjects)
    test_idx = indices_from_subjects(subject_ids, test_subjects)

    print(f"\n{model_name.upper()} SUBJECT SPLIT")
    print(f"Train subjects: {len(final_train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Test subjects:  {len(test_subjects)}")
    print(f"Train samples:  {len(train_idx)}")
    print(f"Val samples:    {len(val_idx)}")
    print(f"Test samples:   {len(test_idx)}")

    results_by_mix = {}
    best_real_test_f1 = -1.0
    best_mix_ratio = None
    best_model = None
    best_params = None
    best_real_test_metrics = None

    # fixed validation and test sets use real ECG
    x_val_real = real_segments[val_idx]
    y_val = labels[val_idx]

    x_test_real = real_segments[test_idx]
    x_test_generated = generated_segments[test_idx]
    y_test = labels[test_idx]

    val_loader = make_loader(x_val_real, y_val, cfg.batch_size, shuffle=False)
    test_real_loader = make_loader(x_test_real, y_test, cfg.batch_size, shuffle=False)
    test_generated_loader = make_loader(x_test_generated, y_test, cfg.batch_size, shuffle=False)

    for mix_ratio in cfg.generated_mix_ratios:
        print("\n" + "#" * 100)
        print(f"{model_name.upper()} | GENERATED TRAINING RATIO = {mix_ratio:.2f}")
        print("#" * 100)

        x_train_real = real_segments[train_idx]
        x_train_generated = generated_segments[train_idx]
        y_train = labels[train_idx]

        x_train_mixed, y_train_mixed = build_mixed_training_data(
            real_x=x_train_real,
            gen_x=x_train_generated,
            y=y_train,
            generated_ratio=mix_ratio,
            seed=cfg.seed,
        )

        train_loader = make_loader(x_train_mixed, y_train_mixed, cfg.batch_size, shuffle=True)

        model, best_params_local, val_metrics, history = run_grid_search(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
        )

        criterion = nn.CrossEntropyLoss()
        real_test_metrics = evaluate_model(model, test_real_loader, criterion, cfg.device)

        results_by_mix[str(mix_ratio)] = {
            "mix_ratio_generated": mix_ratio,
            "best_params": best_params_local,
            "validation_metrics": {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            },
            "real_test_metrics": {
                "loss": real_test_metrics["loss"],
                "accuracy": real_test_metrics["accuracy"],
                "precision": real_test_metrics["precision"],
                "recall": real_test_metrics["recall"],
                "f1": real_test_metrics["f1"],
                "confusion_matrix": real_test_metrics["confusion_matrix"].tolist(),
            },
        }

        print("\nREAL TEST RESULTS")
        print(f"Accuracy : {real_test_metrics['accuracy']:.4f}")
        print(f"Precision: {real_test_metrics['precision']:.4f}")
        print(f"Recall   : {real_test_metrics['recall']:.4f}")
        print(f"F1-score : {real_test_metrics['f1']:.4f}")

        if real_test_metrics["f1"] > best_real_test_f1:
            best_real_test_f1 = real_test_metrics["f1"]
            best_mix_ratio = mix_ratio
            best_model = model
            best_params = best_params_local
            best_real_test_metrics = real_test_metrics

    # Evaluate best model on generated test ECG
    criterion = nn.CrossEntropyLoss()
    generated_test_metrics = evaluate_model(best_model, test_generated_loader, criterion, cfg.device)

    print("\n" + "=" * 100)
    print(f"BEST MIXING CONFIGURATION FOR {model_name.upper()}: generated ratio = {best_mix_ratio:.2f}")
    print("Best REAL TEST performance")
    print(f"Accuracy : {best_real_test_metrics['accuracy']:.4f}")
    print(f"Precision: {best_real_test_metrics['precision']:.4f}")
    print(f"Recall   : {best_real_test_metrics['recall']:.4f}")
    print(f"F1-score : {best_real_test_metrics['f1']:.4f}")

    print("\nGENERATED TEST RESULTS (train on best real/mixed setup, test on generated ECG)")
    print(f"Accuracy : {generated_test_metrics['accuracy']:.4f}")
    print(f"Precision: {generated_test_metrics['precision']:.4f}")
    print(f"Recall   : {generated_test_metrics['recall']:.4f}")
    print(f"F1-score : {generated_test_metrics['f1']:.4f}")

    # save confusion matrices
    fig1 = plot_confusion_matrix(
        best_real_test_metrics["confusion_matrix"],
        title=f"{model_name} - Best Real Test Confusion Matrix"
    )
    fig1.savefig(os.path.join(cfg.output_dir, f"{model_name}_best_real_test_confusion_matrix.png"), dpi=200)
    plt.close(fig1)

    fig2 = plot_confusion_matrix(
        generated_test_metrics["confusion_matrix"],
        title=f"{model_name} - Generated Test Confusion Matrix"
    )
    fig2.savefig(os.path.join(cfg.output_dir, f"{model_name}_generated_test_confusion_matrix.png"), dpi=200)
    plt.close(fig2)

    output = {
        "model_name": model_name,
        "train_subjects": train_subjects.tolist(),
        "test_subjects": test_subjects.tolist(),
        "final_train_subjects": final_train_subjects.tolist(),
        "val_subjects": val_subjects.tolist(),
        "results_by_mix": results_by_mix,
        "best_mix_ratio": best_mix_ratio,
        "best_params": best_params,
        "best_real_test_metrics": {
            "loss": best_real_test_metrics["loss"],
            "accuracy": best_real_test_metrics["accuracy"],
            "precision": best_real_test_metrics["precision"],
            "recall": best_real_test_metrics["recall"],
            "f1": best_real_test_metrics["f1"],
            "confusion_matrix": best_real_test_metrics["confusion_matrix"].tolist(),
        },
        "generated_test_metrics": {
            "loss": generated_test_metrics["loss"],
            "accuracy": generated_test_metrics["accuracy"],
            "precision": generated_test_metrics["precision"],
            "recall": generated_test_metrics["recall"],
            "f1": generated_test_metrics["f1"],
            "confusion_matrix": generated_test_metrics["confusion_matrix"].tolist(),
        },
    }

    save_json(output, os.path.join(cfg.output_dir, f"{model_name}_results.json"))
    return output


# ============================================================
# main
# ============================================================

def main():
    """

    Required arrays:
        real_segments      -> (N, T, 1)
        generated_segments -> (N, T, 1)
        labels             -> (N,)
        subject_ids        -> (N,)
    """

    # ------------------------------------------------------------------
    # dataset loading
    # ------------------------------------------------------------------

    dataset=["MIMIC-AFib"]
    window_size=4
    ecg_train_list = []
    ecg_generated_list = []

    DATA_PATH_REAL = "../../datasets/real/"
    DATA_PATH_GENERATED = "../../datasets/generated/"
    real_segments = np.load(DATA_PATH_REAL + dataset + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)
    generated_segments = np.load(DATA_PATH_GENERATED + dataset + f"/ecg_train_{window_size}sec.npy", allow_pickle=True).reshape(-1, 128*window_size)

    real_segments = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
    generated_segments = np.nan_to_num(np.concatenate(ecg_generated_list).astype("float32"))

    n_samples = 2000
    time_window = 30*130   # example only, depends on your sampling rate and 30 s length

    rng = np.random.RandomState(42)
    real_segments = rng.randn(n_samples, time_window, 1).astype(np.float32)
    generated_segments = real_segments + 0.05 * rng.randn(n_samples, time_window, 1).astype(np.float32)

    # Simulate 28 subjects as in the description
    n_subjects = 28
    subject_ids = np.repeat(np.arange(n_subjects), n_samples // n_subjects)
    subject_ids = np.concatenate([subject_ids, np.full(n_samples - len(subject_ids), n_subjects - 1)])
    subject_ids = subject_ids[:n_samples]

    # Subject-level labels: first 14 AF, next 14 non-AF (example only)
    subject_label_map = {sid: int(sid < 14) for sid in range(n_subjects)}
    labels = np.array([subject_label_map[sid] for sid in subject_ids], dtype=np.int64)

    # ------------------------------------------------------------------
    #
    # ------------------------------------------------------------------

    # basic checks
    assert real_segments.shape == generated_segments.shape
    assert len(real_segments) == len(labels) == len(subject_ids)
    assert real_segments.ndim == 3 and real_segments.shape[-1] == 1

    print("Data loaded successfully.")
    print(f"real_segments shape:      {real_segments.shape}")
    print(f"generated_segments shape: {generated_segments.shape}")
    print(f"labels shape:             {labels.shape}")
    print(f"subject_ids shape:        {subject_ids.shape}")

    # Run Bid-LSTM
    bidlstm_results = experiment_single_model(
        model_name="bidlstm",
        real_segments=real_segments,
        generated_segments=generated_segments,
        labels=labels,
        subject_ids=subject_ids,
        cfg=CFG,
    )

    # Run CNN + LSTM
    cnnlstm_results = experiment_single_model(
        model_name="cnn_lstm",
        real_segments=real_segments,
        generated_segments=generated_segments,
        labels=labels,
        subject_ids=subject_ids,
        cfg=CFG,
    )

    # Summary comparison
    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    for res in [bidlstm_results, cnnlstm_results]:
        print(f"\nModel: {res['model_name']}")
        print(f"Best training generated mix ratio: {res['best_mix_ratio']}")
        print("Best real-test metrics:")
        print(f"  Accuracy : {res['best_real_test_metrics']['accuracy']:.4f}")
        print(f"  Precision: {res['best_real_test_metrics']['precision']:.4f}")
        print(f"  Recall   : {res['best_real_test_metrics']['recall']:.4f}")
        print(f"  F1-score : {res['best_real_test_metrics']['f1']:.4f}")
        print("Generated-test metrics:")
        print(f"  Accuracy : {res['generated_test_metrics']['accuracy']:.4f}")
        print(f"  Precision: {res['generated_test_metrics']['precision']:.4f}")
        print(f"  Recall   : {res['generated_test_metrics']['recall']:.4f}")
        print(f"  F1-score : {res['generated_test_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()