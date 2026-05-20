from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from audio_lab9 import (
    aggregate_energy_map,
    db20,
    DenoiseConfig,
    EnergySearchConfig,
    frame_signal_rms,
    inspect_audio,
    NoiseEstimationConfig,
    rms,
    STFTConfig,
    denoise_time_domain,
    energy_peaks,
    estimate_noise_profile_from_quiet_frames,
    generate_demo_wav,
    istft_from_mag_phase,
    load_audio_mono,
    load_config,
    psnr_db,
    save_wav,
    spectral_subtraction,
    stft_mag_phase,
    summarize_top_energy_cells,
)
from plotting import save_energy_map, save_overview_figure, save_spectrogram, save_spectrogram_comparison


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_cfg_path = repo_root / "config" / "lab9_config.json"

    def rel(path: Path) -> str:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")

    parser = argparse.ArgumentParser(description="Lab 9: Audio noise analysis (variant 14).")
    parser.add_argument("--input", type=str, default=None, help="Input audio file (.wav or .mp3).")
    parser.add_argument("--config", type=str, default=str(default_cfg_path), help="Path to config JSON.")
    parser.add_argument("--demo", action="store_true", help="Generate demo wav + all outputs into assets/ and outputs/.")
    parser.add_argument("--json-out", type=str, default=None, help="Write report JSON to a file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = int(cfg.get("variant", 14))
    if variant != 14:
        raise SystemExit("This repo is prepared for variant 14 (config can be edited if needed).")

    sample_rate = int(cfg.get("sample_rate", 22050))
    stft_cfg = STFTConfig(**cfg["stft"])
    noise_cfg = NoiseEstimationConfig(**cfg["noise_estimation"])
    den_cfg_dict = cfg["denoise"].copy()
    savgol_dict = den_cfg_dict.pop("savgol", {})
    denoise_cfg = DenoiseConfig(**{**den_cfg_dict, "savgol": DenoiseConfig().savgol.__class__(**savgol_dict)})
    energy_cfg = EnergySearchConfig(**cfg["energy_search"])

    assets_dir = repo_root / "assets"
    out_dir = repo_root / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        demo_in = assets_dir / "demo_input.wav"
        generate_demo_wav(demo_in, sr=sample_rate)
        input_path = demo_in
    else:
        input_path = Path(args.input) if args.input else (repo_root.parent / "piano.mp3")
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")

    source_meta = inspect_audio(input_path)
    y, sr = load_audio_mono(input_path, target_sr=sample_rate)
    mono_wav_path = assets_dir / "piano_mono_22050.wav"
    save_wav(mono_wav_path, y, sr)

    repo_input_copy = assets_dir / input_path.name
    if not args.demo and input_path.suffix.lower() != ".wav":
        shutil.copy2(input_path, repo_input_copy)

    freqs, times, mag, phase = stft_mag_phase(y, sr, stft_cfg)
    noise_prof, quiet_idx = estimate_noise_profile_from_quiet_frames(mag, noise_cfg, quiet_fraction=0.15)

    if denoise_cfg.method == "spectral_subtraction":
        mag_denoised = spectral_subtraction(
            mag,
            noise_prof,
            floor_ratio=float(denoise_cfg.floor_ratio),
            savgol=denoise_cfg.savgol,
        )
        y_denoised = istft_from_mag_phase(mag_denoised, phase, sr, stft_cfg)
    else:
        y_denoised = denoise_time_domain(y, sr, denoise_cfg)
        _, _, mag_denoised, _ = stft_mag_phase(y_denoised, sr, stft_cfg)

    y_denoised = (
        y_denoised[: y.shape[0]]
        if y_denoised.shape[0] >= y.shape[0]
        else np.pad(y_denoised, (0, y.shape[0] - y_denoised.shape[0]))
    )
    denoised_wav_path = assets_dir / "piano_denoised.wav"
    save_wav(denoised_wav_path, y_denoised, sr)

    frame_len = stft_cfg.n_fft
    hop_len = stft_cfg.hop_length
    frame_rms_before = frame_signal_rms(y, frame_len, hop_len)
    frame_rms_after = frame_signal_rms(y_denoised, frame_len, hop_len)
    aligned_quiet_idx = quiet_idx[quiet_idx < min(frame_rms_before.size, frame_rms_after.size)]
    quiet_rms_before = float(np.median(frame_rms_before[aligned_quiet_idx])) if aligned_quiet_idx.size else rms(y)
    quiet_rms_after = float(np.median(frame_rms_after[aligned_quiet_idx])) if aligned_quiet_idx.size else rms(y_denoised)
    def robust_spectral_floor_db(profile: np.ndarray) -> float:
        positive = profile[profile > 1e-10]
        return float(np.median(db20(positive))) if positive.size else float(np.median(db20(profile)))

    spectral_floor_before_db = robust_spectral_floor_db(noise_prof)

    after_noise_prof, _ = estimate_noise_profile_from_quiet_frames(mag_denoised, noise_cfg, quiet_fraction=0.15)
    spectral_floor_after_db = robust_spectral_floor_db(after_noise_prof)

    energy_map_data = aggregate_energy_map(mag_denoised, freqs, times, energy_cfg)
    energy_map = np.asarray(energy_map_data["energy"])
    freq_edges = np.asarray(energy_map_data["freq_edges"])
    time_edges = np.asarray(energy_map_data["time_edges"])
    top_cells = summarize_top_energy_cells(energy_map, freq_edges, time_edges, top_k=10)
    global_max = top_cells[0] if top_cells else None

    # Save outputs
    save_spectrogram(assets_dir / "spectrogram_before.png", freqs, times, mag, title="Спектрограмма до обработки", log_freq=True)
    save_spectrogram(assets_dir / "spectrogram_after.png", freqs, times, mag_denoised, title="Спектрограмма после обработки", log_freq=True)
    save_spectrogram_comparison(assets_dir / "spectrogram_compare.png", freqs, times, mag, mag_denoised, log_freq=True)
    save_overview_figure(
        assets_dir / "analysis_overview.png",
        y,
        y_denoised,
        sr,
        freqs,
        times,
        mag,
        mag_denoised,
        peak_time=((global_max["t0_sec"] + global_max["t1_sec"]) / 2.0) if global_max else None,
        peak_band=(global_max["f0_hz"], global_max["f1_hz"]) if global_max else None,
    )
    save_energy_map(assets_dir / "energy_map.png", energy_map, time_edges, freq_edges, max_cell=global_max)

    save_wav(out_dir / "denoised.wav", y_denoised, sr)
    if args.demo:
        save_wav(assets_dir / "demo_denoised.wav", y_denoised, sr)

    peaks = energy_peaks(mag_denoised, freqs, times, energy_cfg, top_k=10)

    report = {
        "variant": 14,
        "input": {
            "path": str(input_path),
            "sample_rate_after_hz": sr,
            "samples_after": int(y.shape[0]),
            "duration_sec": float(y.shape[0] / sr),
            "source": source_meta.__dict__,
            "repo_copy": rel(repo_input_copy) if repo_input_copy.exists() else None,
            "converted_wav": rel(mono_wav_path),
        },
        "config": cfg,
        "noise_profile": {
            "method": noise_cfg.method,
            "scale": noise_cfg.scale,
            "quiet_frame_fraction": 0.15,
            "quiet_frame_count": int(aligned_quiet_idx.size),
            "quiet_rms_before": quiet_rms_before,
            "quiet_rms_after": quiet_rms_after,
            "median_spectral_floor_before_db": spectral_floor_before_db,
            "median_spectral_floor_after_db": spectral_floor_after_db,
        },
        "denoise": {"method": denoise_cfg.method, "floor_ratio": denoise_cfg.floor_ratio, "savgol": denoise_cfg.savgol.__dict__},
        "energy_peaks": peaks,
        "top_energy_cells": top_cells,
        "global_maximum": global_max,
        "quality": {
            "psnr_db_input_vs_denoised": psnr_db(y, y_denoised),
            "rms_before": rms(y),
            "rms_after": rms(y_denoised),
        },
        "outputs": {
            "denoised_wav": rel(denoised_wav_path),
            "spectrogram_before_png": rel(assets_dir / "spectrogram_before.png"),
            "spectrogram_after_png": rel(assets_dir / "spectrogram_after.png"),
            "spectrogram_compare_png": rel(assets_dir / "spectrogram_compare.png"),
            "analysis_overview_png": rel(assets_dir / "analysis_overview.png"),
            "energy_map_png": rel(assets_dir / "energy_map.png"),
        },
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    if args.demo:
        (assets_dir / "demo_report.json").write_text(text, encoding="utf-8")
    else:
        (assets_dir / "piano_report.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
