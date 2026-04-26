from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, stft, istft, wiener, windows


WindowName = Literal["hann"]


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 2048
    hop_length: int = 512
    window: WindowName = "hann"


@dataclass(frozen=True)
class NoiseEstimationConfig:
    method: Literal["median_per_frequency"] = "median_per_frequency"
    scale: float = 1.0


@dataclass(frozen=True)
class SavgolConfig:
    enabled: bool = True
    window_length: int = 11
    polyorder: int = 2


@dataclass(frozen=True)
class DenoiseConfig:
    method: Literal["spectral_subtraction", "wiener_time", "lowpass", "savgol_time"] = "spectral_subtraction"
    floor_ratio: float = 0.02
    savgol: SavgolConfig = SavgolConfig()
    lowpass_hz: float = 5000.0


@dataclass(frozen=True)
class EnergySearchConfig:
    dt_sec: float = 0.1
    df_hz: float = 50.0


def load_audio_mono(path: str | Path, *, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """
    Loads audio file to mono float32 in [-1, 1], returns (y, sr).
    Supports WAV; for MP3 you need libs/ffmpeg available to libsndfile.
    """
    y, sr = sf.read(str(path), always_2d=True)
    y = y.astype(np.float32)
    y_mono = y.mean(axis=1)
    if target_sr is not None and target_sr != sr:
        # Simple resample via FFT method (scipy.signal.resample_poly would be better, but keep deps minimal).
        # Use linear interpolation as a robust fallback.
        x_old = np.linspace(0.0, 1.0, num=len(y_mono), endpoint=False)
        n_new = int(round(len(y_mono) * (target_sr / sr)))
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        y_mono = np.interp(x_new, x_old, y_mono).astype(np.float32)
        sr = int(target_sr)
    return y_mono, int(sr)


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), y.astype(np.float32), sr)


def _get_window(name: WindowName, n_fft: int) -> np.ndarray:
    if name == "hann":
        return windows.hann(n_fft, sym=False)
    raise ValueError(f"Unsupported window: {name}")


def stft_mag_phase(y: np.ndarray, sr: int, cfg: STFTConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win = _get_window(cfg.window, cfg.n_fft)
    f, t, z = stft(
        y,
        fs=sr,
        window=win,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(z).astype(np.float64)
    phase = np.angle(z).astype(np.float64)
    return f.astype(np.float64), t.astype(np.float64), mag, phase


def istft_from_mag_phase(mag: np.ndarray, phase: np.ndarray, sr: int, cfg: STFTConfig) -> np.ndarray:
    win = _get_window(cfg.window, cfg.n_fft)
    z = mag * np.exp(1j * phase)
    _, y = istft(
        z,
        fs=sr,
        window=win,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        input_onesided=True,
        boundary=True,
    )
    y = np.asarray(y, dtype=np.float32)
    # Safety clip
    return np.clip(y, -1.0, 1.0)


def estimate_noise_profile(mag: np.ndarray, cfg: NoiseEstimationConfig) -> np.ndarray:
    if cfg.method != "median_per_frequency":
        raise ValueError(f"Unsupported noise estimation: {cfg.method}")
    prof = np.median(mag, axis=1)
    return prof * float(cfg.scale)


def spectral_subtraction(
    mag: np.ndarray,
    noise_prof: np.ndarray,
    *,
    floor_ratio: float,
    savgol: SavgolConfig,
) -> np.ndarray:
    if mag.ndim != 2:
        raise ValueError("mag must be 2D [freq, time].")
    if noise_prof.shape[0] != mag.shape[0]:
        raise ValueError("noise_prof must match mag frequency bins.")

    floor_ratio = float(floor_ratio)
    if floor_ratio < 0:
        raise ValueError("floor_ratio must be >= 0.")

    noise_2d = noise_prof[:, None]
    out = mag - noise_2d
    floor = floor_ratio * np.maximum(mag, 1e-12)
    out = np.maximum(out, floor)

    if savgol.enabled:
        wl = int(savgol.window_length)
        po = int(savgol.polyorder)
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, po + 2 + ((po + 2) % 2 == 0))
        # Smooth along time for each frequency bin.
        out = savgol_filter(out, window_length=wl, polyorder=po, axis=1, mode="interp")
        out = np.maximum(out, 0.0)

    return out


def denoise_time_domain(y: np.ndarray, sr: int, cfg: DenoiseConfig) -> np.ndarray:
    if cfg.method == "wiener_time":
        # Simple Wiener filter on waveform.
        out = wiener(y.astype(np.float64)).astype(np.float32)
        return np.clip(out, -1.0, 1.0)
    if cfg.method == "lowpass":
        cutoff = float(cfg.lowpass_hz)
        if cutoff <= 0 or cutoff >= sr / 2:
            raise ValueError("lowpass_hz must be in (0, sr/2).")
        b, a = butter(6, cutoff / (sr / 2), btype="low")
        out = filtfilt(b, a, y.astype(np.float64)).astype(np.float32)
        return np.clip(out, -1.0, 1.0)
    if cfg.method == "savgol_time":
        wl = int(cfg.savgol.window_length)
        po = int(cfg.savgol.polyorder)
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, po + 2 + ((po + 2) % 2 == 0))
        out = savgol_filter(y.astype(np.float64), window_length=wl, polyorder=po, mode="interp").astype(np.float32)
        return np.clip(out, -1.0, 1.0)
    raise ValueError(f"Unsupported time-domain method: {cfg.method}")


def psnr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    mse = float(np.mean((ref - test) ** 2))
    if mse <= 0:
        return float("inf")
    peak = 1.0
    return float(10.0 * math.log10((peak * peak) / mse))


def energy_peaks(
    mag: np.ndarray,
    freqs_hz: np.ndarray,
    times_sec: np.ndarray,
    cfg: EnergySearchConfig,
    *,
    top_k: int = 5,
) -> dict[str, object]:
    """
    Finds time moments with maximum energy in neighborhood:
    - time step dt_sec (aggregates into windows of length dt_sec)
    - frequency neighborhood aggregated into bands of width df_hz
    """
    dt = float(cfg.dt_sec)
    df = float(cfg.df_hz)

    if mag.size == 0:
        return {"dt_sec": dt, "df_hz": df, "peaks": []}

    # Aggregate frequency bins into df bands (40-50 Hz recommended by task; we use df_hz from config).
    f_max = float(freqs_hz.max())
    n_bands = max(1, int(math.ceil(f_max / df)))
    band_edges = np.arange(n_bands + 1, dtype=np.float64) * df

    band_energy = np.zeros((n_bands, mag.shape[1]), dtype=np.float64)  # [band, frame]
    for bi in range(n_bands):
        f0, f1 = band_edges[bi], band_edges[bi + 1]
        idx = np.where((freqs_hz >= f0) & (freqs_hz < f1))[0]
        if idx.size:
            band_energy[bi, :] = np.sum(mag[idx, :] ** 2, axis=0)

    # Aggregate into time windows of dt_sec.
    t_end = float(times_sec.max()) if times_sec.size else 0.0
    n_win = max(1, int(math.ceil((t_end + 1e-9) / dt)))
    win_edges = np.arange(n_win + 1, dtype=np.float64) * dt

    win_energy = np.zeros(n_win, dtype=np.float64)
    win_band_energy = np.zeros((n_bands, n_win), dtype=np.float64)
    for wi in range(n_win):
        t0, t1 = win_edges[wi], win_edges[wi + 1]
        tidx = np.where((times_sec >= t0) & (times_sec < t1))[0]
        if tidx.size:
            win_band_energy[:, wi] = np.sum(band_energy[:, tidx], axis=1)
            win_energy[wi] = float(np.sum(win_band_energy[:, wi]))

    # Peaks over windows (neighborhood step is dt_sec).
    peaks_wi, _ = find_peaks(win_energy, distance=1)
    if peaks_wi.size == 0:
        peaks_wi = np.array([int(np.argmax(win_energy))], dtype=int)

    order = np.argsort(-win_energy[peaks_wi])[: int(top_k)]
    peaks_wi = peaks_wi[order]

    peaks: list[dict[str, float]] = []
    for wi in peaks_wi:
        b = int(np.argmax(win_band_energy[:, wi]))
        peaks.append(
            {
                "time_sec": float((win_edges[wi] + win_edges[wi + 1]) / 2.0),
                "energy": float(win_energy[wi]),
                "dominant_band_hz_start": float(band_edges[b]),
                "dominant_band_hz_end": float(band_edges[b + 1]),
            }
        )

    peaks = sorted(peaks, key=lambda d: d["time_sec"])
    return {"dt_sec": dt, "df_hz": df, "peaks": peaks}


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_demo_wav(path: str | Path, *, sr: int = 22050, dur_sec: float = 3.0, seed: int = 42) -> None:
    """
    Synthetic "instrument-like" demo: sum of harmonics + envelope + additive noise.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * dur_sec)
    t = np.arange(n, dtype=np.float64) / sr

    f0 = 440.0
    y = (
        0.55 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Simple pluck envelope.
    env = np.exp(-3.0 * t) * (1.0 - np.exp(-50.0 * t))
    y = y * env
    # Additive background noise.
    y = y + 0.03 * rng.normal(size=n)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    save_wav(path, y, sr)
