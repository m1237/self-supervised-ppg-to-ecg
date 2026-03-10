"""
Evaluation metrics for ECG generation
=====================================

Implements the following metrics:

Fréchet Distance (FD)
Root Mean Squared Error (RMSE)
Pearson's Correlation Coefficient (rho)
Mean Absolute Error for Heart Rate (MAEHR)

Inputs expected
---------------
real_ecg_segments : np.ndarray of shape (N, T)
gen_ecg_segments  : np.ndarray of shape (N, T)
ppg_segments      : np.ndarray of shape (N, T)   # optional for MAEHR(P)
fs                : sampling frequency in Hz

Outputs
-------
- Per-segment metrics
- Aggregate summary statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from scipy import signal
from scipy.stats import pearsonr


# ============================================================
# Basic metrics
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def discrete_frechet_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Discrete Fréchet distance between two 1D signals.
    Signals are embedded as 2D curves: (time_index, amplitude).
    """

    P = np.column_stack([np.arange(len(y_true)), y_true])
    Q = np.column_stack([np.arange(len(y_pred)), y_pred])

    ca = np.full((len(P), len(Q)), -1.0, dtype=np.float64)

    def dist(a, b):
        return np.linalg.norm(a - b)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = dist(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), dist(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), dist(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                dist(P[i], Q[j]),
            )
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    return float(c(len(P) - 1, len(Q) - 1))


# ============================================================
# ECG peak detection (Hamilton-inspired practical version)
# ============================================================

def bandpass_filter_ecg(x: np.ndarray, fs: float, low: float = 5.0, high: float = 20.0, order: int = 2):
    nyq = fs / 2.0
    low = low / nyq
    high = min(high / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, x)


def detect_r_peaks_ecg(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Hamilton-style practical ECG R-peak detection.

    Steps:
    - bandpass filter
    - derivative
    - squaring
    - moving average integration
    - adaptive threshold peak detection
    - local peak refinement on original ECG
    """
    x_f = bandpass_filter_ecg(x, fs, low=5.0, high=20.0, order=2)

    dx = np.ediff1d(x_f, to_begin=0)
    squared = dx ** 2

    window = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    threshold = 0.35 * np.max(integrated) if np.max(integrated) > 0 else 0.0
    min_distance = max(1, int(0.25 * fs))

    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

    refined = []
    radius = max(1, int(0.08 * fs))
    for p in peaks:
        left = max(0, p - radius)
        right = min(len(x), p + radius + 1)
        local_peak = left + np.argmax(x[left:right])
        refined.append(local_peak)

    if len(refined) == 0:
        return np.array([], dtype=int)

    return np.array(sorted(set(refined)), dtype=int)


# ============================================================
# PPG peak detection (Elgendi-inspired practical version)
# ============================================================

def bandpass_filter_ppg(x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0, order: int = 3):
    nyq = fs / 2.0
    low = low / nyq
    high = min(high / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, x)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    return np.convolve(x, np.ones(window) / window, mode="same")


def detect_systolic_peaks_ppg(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Practical systolic peak detection.

    Steps:
    - bandpass filter
    - rectify / emphasize pulsatile structure
    - moving-average based envelope
    - peak detection with refractory period
    """
    x_f = bandpass_filter_ppg(x, fs, low=0.5, high=8.0, order=3)

    x_clip = np.maximum(x_f, 0)
    sq = x_clip ** 2

    w1 = max(1, int(0.111 * fs))
    w2 = max(1, int(0.667 * fs))

    ma_peak = moving_average(sq, w1)
    ma_beat = moving_average(sq, w2)

    threshold = ma_beat + 0.02 * np.mean(sq)
    wave = ma_peak > threshold

    candidate_regions = []
    in_region = False
    start = 0
    for i, flag in enumerate(wave):
        if flag and not in_region:
            start = i
            in_region = True
        elif not flag and in_region:
            candidate_regions.append((start, i))
            in_region = False
    if in_region:
        candidate_regions.append((start, len(wave) - 1))

    peaks = []
    for s, e in candidate_regions:
        if e - s < max(1, int(0.08 * fs)):
            continue
        local_peak = s + np.argmax(x_f[s:e])
        peaks.append(local_peak)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    peaks = np.array(sorted(set(peaks)), dtype=int)

    # enforce minimum physiological spacing
    min_distance = max(1, int(0.30 * fs))
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)

    return np.array(filtered, dtype=int)


# ============================================================
# Heart rate estimation
# ============================================================

def estimate_hr_from_peaks(peaks: np.ndarray, fs: float) -> float:
    """
    HR from average RR / PP interval:
        HR = 60 / interval_seconds
    """
    if len(peaks) < 2:
        return np.nan

    intervals_sec = np.diff(peaks) / fs
    intervals_sec = intervals_sec[intervals_sec > 0]

    if len(intervals_sec) == 0:
        return np.nan

    mean_interval = np.mean(intervals_sec)
    if mean_interval <= 0:
        return np.nan

    return float(60.0 / mean_interval)


def estimate_hr_ecg(ecg: np.ndarray, fs: float) -> float:
    r_peaks = detect_r_peaks_ecg(ecg, fs)
    return estimate_hr_from_peaks(r_peaks, fs)


def estimate_hr_ppg(ppg: np.ndarray, fs: float) -> float:
    p_peaks = detect_systolic_peaks_ppg(ppg, fs)
    return estimate_hr_from_peaks(p_peaks, fs)


def mae_hr(hr_gt: np.ndarray, hr_est: np.ndarray) -> float:
    valid = ~np.isnan(hr_gt) & ~np.isnan(hr_est)
    if np.sum(valid) == 0:
        return np.nan
    return float(np.mean(np.abs(hr_gt[valid] - hr_est[valid])))


# ============================================================
# Segment-level evaluation
# ============================================================

def evaluate_single_segment(
    real_ecg: np.ndarray,
    gen_ecg: np.ndarray,
    fs: float,
    ppg: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    fd = discrete_frechet_distance(real_ecg, gen_ecg)
    seg_rmse = rmse(real_ecg, gen_ecg)
    rho = pearson_rho(real_ecg, gen_ecg)

    hr_gt = estimate_hr_ecg(real_ecg, fs)
    hr_gen = estimate_hr_ecg(gen_ecg, fs)
    maehr_gen = np.abs(hr_gt - hr_gen) if (not np.isnan(hr_gt) and not np.isnan(hr_gen)) else np.nan

    result = {
        "FD": fd,
        "RMSE_mV": seg_rmse,
        "rho": rho,
        "HR_GT_bpm": hr_gt,
        "HR_genECG_bpm": hr_gen,
        "MAEHR_Eprime_bpm": maehr_gen,
    }

    if ppg is not None:
        hr_ppg = estimate_hr_ppg(ppg, fs)
        maehr_ppg = np.abs(hr_gt - hr_ppg) if (not np.isnan(hr_gt) and not np.isnan(hr_ppg)) else np.nan
        result["HR_PPG_bpm"] = hr_ppg
        result["MAEHR_P_bpm"] = maehr_ppg

    return result


# ============================================================
# Dataset-level evaluation
# ============================================================

@dataclass
class EvaluationSummary:
    FD_mean: float
    FD_std: float
    RMSE_mean: float
    RMSE_std: float
    rho_mean: float
    rho_std: float
    MAEHR_Eprime_mean: float
    MAEHR_Eprime_std: float
    MAEHR_P_mean: Optional[float] = None
    MAEHR_P_std: Optional[float] = None
    valid_HR_segments: Optional[int] = None


def summarize_array(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), np.nan
    return float(np.mean(x)), float(np.std(x, ddof=1))


def evaluate_dataset(
    real_ecg_segments: np.ndarray,
    gen_ecg_segments: np.ndarray,
    fs: float,
    ppg_segments: Optional[np.ndarray] = None,
    segment_ids: Optional[np.ndarray] = None,
    subject_ids: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, EvaluationSummary]:
    """
    Evaluate the whole dataset.

    Parameters
    ----------
    real_ecg_segments : (N, T)
    gen_ecg_segments  : (N, T)
    ppg_segments      : (N, T), optional
    fs                : sampling rate

    Returns
    -------
    per_segment_df, summary
    """
    assert real_ecg_segments.shape == gen_ecg_segments.shape
    assert real_ecg_segments.ndim == 2

    N = len(real_ecg_segments)

    if ppg_segments is not None:
        assert ppg_segments.shape == real_ecg_segments.shape

    rows = []
    for i in range(N):
        real_ecg = real_ecg_segments[i]
        gen_ecg = gen_ecg_segments[i]
        ppg = ppg_segments[i] if ppg_segments is not None else None

        metrics = evaluate_single_segment(real_ecg, gen_ecg, fs, ppg)

        row = {"index": i}
        if segment_ids is not None:
            row["segment_id"] = segment_ids[i]
        if subject_ids is not None:
            row["subject_id"] = subject_ids[i]
        if labels is not None:
            row["label"] = labels[i]

        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    fd_mean, fd_std = summarize_array(df["FD"].values)
    rmse_mean, rmse_std = summarize_array(df["RMSE_mV"].values)
    rho_mean, rho_std = summarize_array(df["rho"].values)
    mae_e_mean, mae_e_std = summarize_array(df["MAEHR_Eprime_bpm"].values)

    mae_p_mean = None
    mae_p_std = None
    if "MAEHR_P_bpm" in df.columns:
        mae_p_mean, mae_p_std = summarize_array(df["MAEHR_P_bpm"].values)

    valid_hr_segments = int(np.sum(~np.isnan(df["MAEHR_Eprime_bpm"].values)))

    summary = EvaluationSummary(
        FD_mean=fd_mean,
        FD_std=fd_std,
        RMSE_mean=rmse_mean,
        RMSE_std=rmse_std,
        rho_mean=rho_mean,
        rho_std=rho_std,
        MAEHR_Eprime_mean=mae_e_mean,
        MAEHR_Eprime_std=mae_e_std,
        MAEHR_P_mean=mae_p_mean,
        MAEHR_P_std=mae_p_std,
        valid_HR_segments=valid_hr_segments,
    )

    return df, summary


# ============================================================
# Class-wise summary utility
# ============================================================

def summarize_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    metrics = ["FD", "RMSE_mV", "rho", "MAEHR_Eprime_bpm"]
    if "MAEHR_P_bpm" in df.columns:
        metrics.append("MAEHR_P_bpm")

    rows = []
    for group_name, g in df.groupby(group_col):
        for m in metrics:
            vals = g[m].dropna().values
            rows.append({
                "group": group_name,
                "metric": m,
                "mean": np.mean(vals) if len(vals) else np.nan,
                "std": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                "median": np.median(vals) if len(vals) else np.nan,
                "n": len(vals),
            })
    return pd.DataFrame(rows)


# ============================================================
# Example usage
# ============================================================

def create_synthetic_signals(N: int = 20, T: int = 3900, fs: float = 130.0):
    """
    Synthetic ECG/PPG-like signals for demonstration.
    Replace with real signals in practice.
    """
    t = np.arange(T) / fs

    real_ecg_segments = []
    gen_ecg_segments = []
    ppg_segments = []
    labels = []

    for i in range(N):
        hr = np.random.uniform(60, 100)
        rr = 60.0 / hr

        ecg = 0.01 * np.sin(2 * np.pi * 0.3 * t)

        beat_times = np.arange(0.5, t[-1], rr)
        for bt in beat_times:
            c = int(bt * fs)
            if 3 <= c < T - 4:
                ecg[c - 3:c + 4] += np.array([0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1])

        ecg += 0.01 * np.random.randn(T)
        gen = ecg + 0.02 * np.random.randn(T)

        ppg = 0.02 * np.sin(2 * np.pi * 0.2 * t)
        for bt in beat_times + 0.15:
            c = int(bt * fs)
            if 4 <= c < T - 5:
                ppg[c - 4:c + 5] += np.array([0.05, 0.12, 0.25, 0.45, 0.6, 0.45, 0.25, 0.12, 0.05])

        ppg += 0.01 * np.random.randn(T)

        real_ecg_segments.append(ecg.astype(np.float32))
        gen_ecg_segments.append(gen.astype(np.float32))
        ppg_segments.append(ppg.astype(np.float32))
        labels.append(np.random.randint(0, 2))

    return (
        np.asarray(real_ecg_segments),
        np.asarray(gen_ecg_segments),
        np.asarray(ppg_segments),
        np.asarray(labels),
    )


def main():
    fs = 130.0

    # Replace with your actual arrays
    real_ecg_segments, gen_ecg_segments, ppg_segments, labels = create_synthetic_signals(
        N=24, T=int(30 * fs), fs=fs
    )
    subject_ids = np.arange(len(real_ecg_segments))
    segment_ids = np.arange(len(real_ecg_segments))

    per_segment_df, summary = evaluate_dataset(
        real_ecg_segments=real_ecg_segments,
        gen_ecg_segments=gen_ecg_segments,
        ppg_segments=ppg_segments,
        fs=fs,
        segment_ids=segment_ids,
        subject_ids=subject_ids,
        labels=labels,
    )

    print("\nPer-segment results:")
    print(per_segment_df.head())

    print("\nSummary:")
    print(summary)

    print("\nClass-wise summary:")
    class_summary = summarize_by_group(per_segment_df, "label")
    print(class_summary)

    # Optional: save
    per_segment_df.to_csv("ecg_generation_evaluation_per_segment.csv", index=False)
    class_summary.to_csv("ecg_generation_evaluation_by_label.csv", index=False)


if __name__ == "__main__":
    main()