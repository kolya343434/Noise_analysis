from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def _db_mag(mag: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def _log_ylim(freqs_hz: np.ndarray) -> tuple[float, float]:
    f_min = max(1.0, float(freqs_hz[1]) if freqs_hz.size > 1 else 1.0)
    f_max = float(freqs_hz.max()) if freqs_hz.size else 1.0
    return f_min, f_max


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

    db = _db_mag(mag)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    mesh = ax.pcolormesh(times_sec, freqs_hz, db, shading="auto", cmap="cividis")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if log_freq:
        ax.set_yscale("log")
        ax.set_ylim(*_log_ylim(freqs_hz))
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

    db_b = _db_mag(mag_before)
    db_a = _db_mag(mag_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, sharex=True, sharey=True)
    axes[0].pcolormesh(times_sec, freqs_hz, db_b, shading="auto", cmap="cividis")
    axes[0].set_title("Before")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    m1 = axes[1].pcolormesh(times_sec, freqs_hz, db_a, shading="auto", cmap="cividis")
    axes[1].set_title("After")
    axes[1].set_xlabel("Time (s)")
    if log_freq:
        for ax in axes:
            ax.set_yscale("log")
            ax.set_ylim(*_log_ylim(freqs_hz))
    fig.colorbar(m1, ax=axes.ravel().tolist(), label="Magnitude (dB)", shrink=0.85)
    fig.subplots_adjust(wspace=0.08, right=0.92)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_overview_figure(
    out_path: str | Path,
    signal_before: np.ndarray,
    signal_after: np.ndarray,
    sr: int,
    freqs_hz: np.ndarray,
    times_sec: np.ndarray,
    mag_before: np.ndarray,
    mag_after: np.ndarray,
    *,
    peak_time: float | None = None,
    peak_band: tuple[float, float] | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_before = np.arange(signal_before.size) / sr
    t_after = np.arange(signal_after.size) / sr
    db_b = _db_mag(mag_before)
    db_a = _db_mag(mag_after)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=170)
    fig.patch.set_facecolor("#f7fafc")

    axes[0, 0].plot(t_before, signal_before, color="#1f4e79", linewidth=0.8)
    axes[0, 0].set_title("Исходная запись")
    axes[0, 0].set_xlabel("Время, с")
    axes[0, 0].set_ylabel("Амплитуда")
    axes[0, 0].grid(alpha=0.15)

    axes[0, 1].plot(t_after, signal_after, color="#2a7f62", linewidth=0.8)
    axes[0, 1].set_title("После шумоподавления")
    axes[0, 1].set_xlabel("Время, с")
    axes[0, 1].set_ylabel("Амплитуда")
    axes[0, 1].grid(alpha=0.15)

    m0 = axes[1, 0].pcolormesh(times_sec, freqs_hz, db_b, shading="auto", cmap="cividis")
    axes[1, 0].set_title("Спектрограмма исходной записи")
    axes[1, 0].set_xlabel("Время, с")
    axes[1, 0].set_ylabel("Частота, Гц")

    m1 = axes[1, 1].pcolormesh(times_sec, freqs_hz, db_a, shading="auto", cmap="cividis")
    axes[1, 1].set_title("Спектрограмма после обработки")
    axes[1, 1].set_xlabel("Время, с")
    axes[1, 1].set_ylabel("Частота, Гц")

    for ax in axes[1, :]:
        ax.set_yscale("log")
        ax.set_ylim(*_log_ylim(freqs_hz))

    if peak_time is not None:
        axes[1, 1].axvline(peak_time, color="#ff3b30", linewidth=1.3, linestyle="--")
    if peak_time is not None and peak_band is not None:
        axes[1, 1].add_patch(
            Rectangle(
                (peak_time - 0.05, peak_band[0]),
                0.1,
                max(peak_band[1] - peak_band[0], 1.0),
                linewidth=1.6,
                edgecolor="#ff3b30",
                facecolor="none",
            )
        )

    fig.colorbar(m0, ax=[axes[1, 0], axes[1, 1]], label="Magnitude, dB", shrink=0.82)
    fig.subplots_adjust(wspace=0.18, hspace=0.28, right=0.92)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_energy_map(
    out_path: str | Path,
    energy_map: np.ndarray,
    time_edges: np.ndarray,
    freq_edges: np.ndarray,
    *,
    max_cell: dict[str, float] | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=170)
    mesh = ax.pcolormesh(time_edges, freq_edges, energy_map, shading="auto", cmap="cividis")
    ax.set_title("Карта энергии E(t, f) для восстановленного сигнала")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    ax.set_ylim(float(freq_edges[0]), min(float(freq_edges[-1]), 5000.0))
    fig.colorbar(mesh, ax=ax, label="Энергия")

    if max_cell is not None:
        rect = Rectangle(
            (max_cell["t0_sec"], max_cell["f0_hz"]),
            max_cell["t1_sec"] - max_cell["t0_sec"],
            max_cell["f1_hz"] - max_cell["f0_hz"],
            linewidth=2.0,
            edgecolor="#ff3b30",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            max_cell["t0_sec"],
            max_cell["f1_hz"] + 60.0,
            "Максимум",
            color="#ff3b30",
            fontsize=10,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

