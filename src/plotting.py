from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_spectrogram(
    out_path: str | Path,
    freqs_hz: np.ndarray,
    times_sec: np.ndarray,
    mag: np.ndarray,
    *,
    title: str,
    log_freq: bool = True,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert magnitude to dB-like scale for visualization.
    eps = 1e-12
    db = 20.0 * np.log10(np.maximum(mag, eps))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    mesh = ax.pcolormesh(times_sec, freqs_hz, db, shading="auto", cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if log_freq:
        ax.set_yscale("log")
        ax.set_ylim(max(1.0, float(freqs_hz[1]) if freqs_hz.size > 1 else 1.0), float(freqs_hz.max()))
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_spectrogram_comparison(
    out_path: str | Path,
    freqs_hz: np.ndarray,
    times_sec: np.ndarray,
    mag_before: np.ndarray,
    mag_after: np.ndarray,
    *,
    log_freq: bool = True,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    eps = 1e-12
    db_b = 20.0 * np.log10(np.maximum(mag_before, eps))
    db_a = 20.0 * np.log10(np.maximum(mag_after, eps))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, sharex=True, sharey=True)
    m0 = axes[0].pcolormesh(times_sec, freqs_hz, db_b, shading="auto", cmap="gray")
    axes[0].set_title("Before")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    m1 = axes[1].pcolormesh(times_sec, freqs_hz, db_a, shading="auto", cmap="gray")
    axes[1].set_title("After")
    axes[1].set_xlabel("Time (s)")
    if log_freq:
        for ax in axes:
            ax.set_yscale("log")
            ax.set_ylim(max(1.0, float(freqs_hz[1]) if freqs_hz.size > 1 else 1.0), float(freqs_hz.max()))
    fig.colorbar(m1, ax=axes.ravel().tolist(), label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

