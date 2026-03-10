"""
Baseline fluctuation analysis for real and generated ECG
=======================================================


Inputs:
----------------
real_ecg_segments      : np.ndarray, shape (N, T)
generated_ecg_segments : np.ndarray, shape (N, T)
labels                 : np.ndarray, shape (N,)   # 1=AF, 0=non-AF
subject_ids            : np.ndarray, shape (N,)   # participant IDs
fs                     : int or float             # sampling frequency in Hz

Optional:
---------
segment_ids            : np.ndarray, shape (N,)   # if you want external IDs

Outputs:
--------
- Summary tables for baseline metrics
- Summary tables for PSD band powers
- Representative figures for one AF and one non-AF sample
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal, interpolate
from scipy.stats import pearsonr
from scipy.integrate import simpson


# ============================================================
# Utility
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(pearsonr(x, y)[0])


def discrete_frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Discrete Fréchet distance between two 1D curves represented as point sequences.
    Each curve is converted to 2D points: (time_index, amplitude).

    P, Q: shape (T,)
    """
    P2 = np.column_stack([np.arange(len(P)), P])
    Q2 = np.column_stack([np.arange(len(Q)), Q])

    ca = np.full((len(P2), len(Q2)), -1.0, dtype=np.float64)

    def euclidean(a, b):
        return np.linalg.norm(a - b)

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = euclidean(P2[0], Q2[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), euclidean(P2[i], Q2[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), euclidean(P2[0], Q2[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)),
                euclidean(P2[i], Q2[j]),
            )
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    return float(_c(len(P2) - 1, len(Q2) - 1))


# ============================================================
# Pan-Tompkins-inspired QRS detection
# ============================================================

def bandpass_filter_ecg(ecg: np.ndarray, fs: float, lowcut: float = 5.0, highcut: float = 15.0, order: int = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, ecg)


def detect_qrs_pan_tompkins(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Simplified Pan-Tompkins-inspired QRS detector:
    1) bandpass filter
    2) derivative
    3) squaring
    4) moving window integration
    5) peak detection

    Returns:
        r_peaks: array of detected R-peak sample indices
    """
    ecg_f = bandpass_filter_ecg(ecg, fs, lowcut=5.0, highcut=15.0, order=2)

    # derivative
    diff = np.ediff1d(ecg_f, to_begin=0)

    # squaring
    squared = diff ** 2

    # moving window integration (~150 ms)
    win_size = max(1, int(0.150 * fs))
    mwa = np.convolve(squared, np.ones(win_size) / win_size, mode="same")

    # adaptive threshold
    threshold = 0.35 * np.max(mwa) if np.max(mwa) > 0 else 0.0

    # refractory period ~200 ms
    distance = max(1, int(0.200 * fs))

    peaks, _ = signal.find_peaks(mwa, height=threshold, distance=distance)

    # refine peak positions on original ECG around each integrated peak
    search_radius = max(1, int(0.080 * fs))
    r_peaks = []
    for p in peaks:
        left = max(0, p - search_radius)
        right = min(len(ecg), p + search_radius + 1)
        local_r = left + np.argmax(ecg[left:right])
        r_peaks.append(local_r)

    if len(r_peaks) == 0:
        return np.array([], dtype=int)

    r_peaks = np.array(sorted(set(r_peaks)), dtype=int)
    return r_peaks


# ============================================================
# Baseline extraction by QRS removal + interpolation
# ============================================================

def qrs_mask_from_rpeaks(length: int, r_peaks: np.ndarray, fs: float,
                         pre_ms: float = 80.0, post_ms: float = 120.0) -> np.ndarray:
    """
    Mask QRS regions around detected R-peaks.
    """
    mask = np.zeros(length, dtype=bool)
    pre = int((pre_ms / 1000.0) * fs)
    post = int((post_ms / 1000.0) * fs)

    for r in r_peaks:
        start = max(0, r - pre)
        end = min(length, r + post + 1)
        mask[start:end] = True

    return mask


def interpolate_non_qrs_baseline(ecg: np.ndarray, qrs_mask: np.ndarray) -> np.ndarray:
    """
    Keep non-QRS samples, interpolate across QRS-removed regions.
    """
    x = np.arange(len(ecg))
    keep = ~qrs_mask

    if np.sum(keep) < 2:
        # fallback if too much removed
        return np.full_like(ecg, fill_value=np.mean(ecg), dtype=float)

    baseline = np.interp(x, x[keep], ecg[keep])
    return baseline.astype(np.float64)


def extract_baseline(ecg: np.ndarray, fs: float,
                     pre_ms: float = 80.0,
                     post_ms: float = 120.0):
    """
    Returns:
        baseline        : interpolated baseline after QRS removal
        r_peaks         : detected QRS centers
        qrs_mask        : boolean mask of removed QRS samples
    """
    r_peaks = detect_qrs_pan_tompkins(ecg, fs)
    qrs_mask = qrs_mask_from_rpeaks(len(ecg), r_peaks, fs, pre_ms=pre_ms, post_ms=post_ms)
    baseline = interpolate_non_qrs_baseline(ecg, qrs_mask)
    return baseline, r_peaks, qrs_mask


# ============================================================
# PSD analysis
# ============================================================

def welch_psd(x: np.ndarray, fs: float,
              nperseg: int | None = None,
              noverlap: int | None = None):
    if nperseg is None:
        nperseg = min(len(x), int(4 * fs))  # ~4-second window by default
    if noverlap is None:
        noverlap = nperseg // 2

    f, pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return f, pxx


def band_power(f: np.ndarray, pxx: np.ndarray, fmin: float, fmax: float) -> float:
    idx = (f >= fmin) & (f <= fmax)
    if np.sum(idx) < 2:
        return np.nan
    return float(simpson(pxx[idx], x=f[idx]))


# ============================================================
# Per-segment analysis
# ============================================================

def analyze_single_pair(real_ecg: np.ndarray, gen_ecg: np.ndarray, fs: float) -> dict:
    """
    Performs both baseline and spectral analyses for one real-generated ECG pair.
    """
    real_baseline, real_rpeaks, real_mask = extract_baseline(real_ecg, fs)
    gen_baseline, gen_rpeaks, gen_mask = extract_baseline(gen_ecg, fs)

    # 8.5.1 baseline comparison
    baseline_rmse = rmse(real_baseline, gen_baseline)
    baseline_fd = discrete_frechet_distance(real_baseline, gen_baseline)
    baseline_rho = safe_pearsonr(real_baseline, gen_baseline)

    # 8.5.2 PSD analysis on extracted baseline dynamics
    f_real, pxx_real = welch_psd(real_baseline, fs)
    f_gen, pxx_gen = welch_psd(gen_baseline, fs)

    # same frequency grid expected from same settings, but keep robust
    if len(f_real) != len(f_gen) or not np.allclose(f_real, f_gen):
        # interpolate generated PSD onto real frequency grid if needed
        interp_fun = interpolate.interp1d(
            f_gen, pxx_gen, bounds_error=False, fill_value="extrapolate"
        )
        pxx_gen_aligned = interp_fun(f_real)
        f_use = f_real
        pxx_real_use = pxx_real
        pxx_gen_use = pxx_gen_aligned
    else:
        f_use = f_real
        pxx_real_use = pxx_real
        pxx_gen_use = pxx_gen

    spectral = {
        "real_power_0.05_1": band_power(f_use, pxx_real_use, 0.05, 1.0),
        "gen_power_0.05_1": band_power(f_use, pxx_gen_use, 0.05, 1.0),
        "real_power_3_9": band_power(f_use, pxx_real_use, 3.0, 9.0),
        "gen_power_3_9": band_power(f_use, pxx_gen_use, 3.0, 9.0),
        "real_power_0.05_9": band_power(f_use, pxx_real_use, 0.05, 9.0),
        "gen_power_0.05_9": band_power(f_use, pxx_gen_use, 0.05, 9.0),
    }

    return {
        "real_baseline": real_baseline,
        "gen_baseline": gen_baseline,
        "real_rpeaks": real_rpeaks,
        "gen_rpeaks": gen_rpeaks,
        "real_qrs_mask": real_mask,
        "gen_qrs_mask": gen_mask,
        "baseline_rmse_mV": baseline_rmse,
        "baseline_fd": baseline_fd,
        "baseline_rho": baseline_rho,
        "freqs": f_use,
        "real_psd": pxx_real_use,
        "gen_psd": pxx_gen_use,
        **spectral,
    }


# ============================================================
# Dataset-level analysis
# ============================================================

def summarize_metrics(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    metrics = [
        "baseline_rmse_mV",
        "baseline_fd",
        "baseline_rho",
        "real_power_0.05_1",
        "gen_power_0.05_1",
        "real_power_3_9",
        "gen_power_3_9",
        "real_power_0.05_9",
        "gen_power_0.05_9",
    ]

    rows = []
    for m in metrics:
        vals = df[m].dropna().values
        rows.append({
            "group": group_name,
            "metric": m,
            "mean": np.mean(vals) if len(vals) else np.nan,
            "std": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "median": np.median(vals) if len(vals) else np.nan,
            "n": len(vals),
        })
    return pd.DataFrame(rows)


def run_baseline_analysis(
    real_ecg_segments: np.ndarray,
    generated_ecg_segments: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    fs: float,
    output_dir: str = "./baseline_analysis_outputs",
):
    """
    real_ecg_segments      : (N, T)
    generated_ecg_segments : (N, T)
    labels                 : (N,) 1=AF, 0=non-AF
    subject_ids            : (N,)
    """

    ensure_dir(output_dir)

    assert real_ecg_segments.shape == generated_ecg_segments.shape
    assert real_ecg_segments.ndim == 2
    assert len(real_ecg_segments) == len(labels) == len(subject_ids)

    all_results = []
    stored_examples = {}

    for i in range(len(real_ecg_segments)):
        real_ecg = real_ecg_segments[i]
        gen_ecg = generated_ecg_segments[i]
        label = int(labels[i])
        subject_id = subject_ids[i]

        result = analyze_single_pair(real_ecg, gen_ecg, fs)

        row = {
            "index": i,
            "subject_id": subject_id,
            "label": label,
            "class_name": "AF" if label == 1 else "non-AF",
            "baseline_rmse_mV": result["baseline_rmse_mV"],
            "baseline_fd": result["baseline_fd"],
            "baseline_rho": result["baseline_rho"],
            "real_power_0.05_1": result["real_power_0.05_1"],
            "gen_power_0.05_1": result["gen_power_0.05_1"],
            "real_power_3_9": result["real_power_3_9"],
            "gen_power_3_9": result["gen_power_3_9"],
            "real_power_0.05_9": result["real_power_0.05_9"],
            "gen_power_0.05_9": result["gen_power_0.05_9"],
        }
        all_results.append(row)

        # store first example from each class for visualization
        if label == 1 and "AF" not in stored_examples:
            stored_examples["AF"] = {
                "raw_real": real_ecg,
                "raw_gen": gen_ecg,
                **result,
                "subject_id": subject_id,
                "index": i,
            }
        elif label == 0 and "non-AF" not in stored_examples:
            stored_examples["non-AF"] = {
                "raw_real": real_ecg,
                "raw_gen": gen_ecg,
                **result,
                "subject_id": subject_id,
                "index": i,
            }

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "baseline_analysis_per_segment.csv"), index=False)

    # Summaries
    summary_all = summarize_metrics(results_df, "All")
    summary_af = summarize_metrics(results_df[results_df["label"] == 1], "AF")
    summary_nonaf = summarize_metrics(results_df[results_df["label"] == 0], "non-AF")

    summary_df = pd.concat([summary_all, summary_af, summary_nonaf], axis=0, ignore_index=True)
    summary_df.to_csv(os.path.join(output_dir, "baseline_analysis_summary.csv"), index=False)

    print("\nPer-segment results saved to:")
    print(os.path.join(output_dir, "baseline_analysis_per_segment.csv"))

    print("\nSummary results saved to:")
    print(os.path.join(output_dir, "baseline_analysis_summary.csv"))

    print("\n===== Summary: All samples =====")
    print(summary_all.to_string(index=False))

    print("\n===== Summary: AF =====")
    print(summary_af.to_string(index=False))

    print("\n===== Summary: non-AF =====")
    print(summary_nonaf.to_string(index=False))

    # Representative plots
    if "AF" in stored_examples:
        plot_representative_example(
            stored_examples["AF"],
            fs=fs,
            class_name="AF",
            save_path=os.path.join(output_dir, "representative_AF.png"),
        )

    if "non-AF" in stored_examples:
        plot_representative_example(
            stored_examples["non-AF"],
            fs=fs,
            class_name="non-AF",
            save_path=os.path.join(output_dir, "representative_nonAF.png"),
        )

    return results_df, summary_df, stored_examples


# ============================================================
# Plotting
# ============================================================

def plot_representative_example(example: dict, fs: float, class_name: str, save_path: str):
    raw_real = example["raw_real"]
    raw_gen = example["raw_gen"]
    baseline_real = example["real_baseline"]
    baseline_gen = example["gen_baseline"]
    r_real = example["real_rpeaks"]
    r_gen = example["gen_rpeaks"]
    qrs_mask_real = example["real_qrs_mask"]
    qrs_mask_gen = example["gen_qrs_mask"]
    freqs = example["freqs"]
    psd_real = example["real_psd"]
    psd_gen = example["gen_psd"]

    t = np.arange(len(raw_real)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    # Raw ECG + detected R-peaks
    axes[0].plot(t, raw_real, label="Real ECG", linewidth=1.0)
    axes[0].plot(t, raw_gen, label="Generated ECG", linewidth=1.0, alpha=0.8)
    if len(r_real):
        axes[0].scatter(r_real / fs, raw_real[r_real], s=15, label="Real R-peaks")
    if len(r_gen):
        axes[0].scatter(r_gen / fs, raw_gen[r_gen], s=15, label="Generated R-peaks")
    axes[0].set_title(f"{class_name}: raw ECG and detected R-peaks")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Baseline after QRS removal + interpolation
    axes[1].plot(t, baseline_real, label="Real baseline", linewidth=1.5)
    axes[1].plot(t, baseline_gen, label="Generated baseline", linewidth=1.5, alpha=0.85)
    axes[1].set_title(
        f"{class_name}: interpolated baseline after QRS removal\n"
        f"RMSE={example['baseline_rmse_mV']:.4f} mV, "
        f"FD={example['baseline_fd']:.4f}, "
        f"ρ={example['baseline_rho']:.4f}"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    # PSD comparison
    band_idx = (freqs >= 0.05) & (freqs <= 9.0)
    axes[2].plot(freqs[band_idx], psd_real[band_idx], label="Real baseline PSD", linewidth=1.5)
    axes[2].plot(freqs[band_idx], psd_gen[band_idx], label="Generated baseline PSD", linewidth=1.5, alpha=0.85)
    axes[2].axvspan(0.05, 1.0, alpha=0.15, label="0.05–1 Hz")
    axes[2].axvspan(3.0, 9.0, alpha=0.10, label="3–9 Hz")
    axes[2].set_title(
        f"{class_name}: Welch PSD of baseline dynamics\n"
        f"Real/Gen power 0.05–1 Hz = {example['real_power_0.05_1']:.4e} / {example['gen_power_0.05_1']:.4e} | "
        f"Real/Gen power 3–9 Hz = {example['real_power_3_9']:.4e} / {example['gen_power_3_9']:.4e}"
    )
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("PSD")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved representative plot: {save_path}")


# ============================================================
# Example main
# ============================================================

def main():
    """
    R

    Required:
        real_ecg_segments      : shape (N, T)
        generated_ecg_segments : shape (N, T)
        labels                 : shape (N,)
        subject_ids            : shape (N,)
        fs                     : sampling rate
    """
    np.random.seed(42)


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
    
    fs = 100.0
    N = 20
    T = int(30 * fs)  # 30-second segments

    time = np.arange(T) / fs


    labels = []
    subject_ids = []

    for i in range(N):
        label = 1 if i < N // 2 else 0  # first half AF, second half non-AF
        subject_id = i

        # basic generated ECG-like signal + baseline fluctuation
        baseline = 0.05 * np.sin(2 * np.pi * 0.25 * time) + 0.02 * np.sin(2 * np.pi * 0.8 * time)
        if label == 1:
            baseline += 0.015 * np.sin(2 * np.pi * 5.0 * time)  # AF-like higher-frequency activity

        ecg = baseline.copy()

        # crude heartbeat injection
        rr = np.random.uniform(0.55, 0.9) if label == 1 else np.random.uniform(0.75, 0.95)
        peak_times = np.arange(0.5, 30.0, rr)
        for pt in peak_times:
            center = int(pt * fs)
            if 2 <= center < T - 3:
                ecg[center-2:center+3] += np.array([0.15, 0.5, 1.0, 0.5, 0.15])

        ecg += 0.01 * np.random.randn(T)

        gen_ecg = ecg + 0.01 * np.random.randn(T)
        gen_ecg += 0.005 * np.sin(2 * np.pi * 0.12 * time)

        real_segments.append(ecg)
        generated_segments.append(gen_ecg)
        labels.append(label)
        subject_ids.append(subject_id)

    real_ecg_segments = np.asarray(real_ecg_segments, dtype=np.float64)
    generated_ecg_segments = np.asarray(generated_ecg_segments, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    subject_ids = np.asarray(subject_ids)

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------
    results_df, summary_df, examples = run_baseline_analysis(
        real_ecg_segments=real_ecg_segments,
        generated_ecg_segments=generated_ecg_segments,
        labels=labels,
        subject_ids=subject_ids,
        fs=fs,
        output_dir="./baseline_analysis_outputs",
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()