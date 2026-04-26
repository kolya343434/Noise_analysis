from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_lab9 import (
    DenoiseConfig,
    EnergySearchConfig,
    NoiseEstimationConfig,
    STFTConfig,
    denoise_time_domain,
    energy_peaks,
    estimate_noise_profile,
    generate_demo_wav,
    istft_from_mag_phase,
    load_audio_mono,
    load_config,
    psnr_db,
    save_wav,
    spectral_subtraction,
    stft_mag_phase,
)
from plotting import save_spectrogram, save_spectrogram_comparison


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_cfg_path = repo_root / "config" / "lab9_config.json"

    parser = argparse.ArgumentParser(description="Lab 9: Audio noise analysis (variant 14).")
    parser.add_argument("--input", type=str, default=None, help="Input audio file (.wav recommended).")
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

    if args.demo:
        demo_in = assets_dir / "demo_input.wav"
        generate_demo_wav(demo_in, sr=sample_rate)
        input_path = demo_in
    else:
        if not args.input:
            raise SystemExit("Provide --input PATH or use --demo.")
        input_path = Path(args.input)

    y, sr = load_audio_mono(input_path, target_sr=sample_rate)

    freqs, times, mag, phase = stft_mag_phase(y, sr, stft_cfg)
    noise_prof = estimate_noise_profile(mag, noise_cfg)

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

    # Save outputs
    save_spectrogram(assets_dir / "spectrogram_before.png", freqs, times, mag, title="Spectrogram (before)", log_freq=True)
    save_spectrogram(assets_dir / "spectrogram_after.png", freqs, times, mag_denoised, title="Spectrogram (after)", log_freq=True)
    save_spectrogram_comparison(assets_dir / "spectrogram_compare.png", freqs, times, mag, mag_denoised, log_freq=True)

    save_wav(out_dir / "denoised.wav", y_denoised, sr)
    if args.demo:
        save_wav(assets_dir / "demo_denoised.wav", y_denoised, sr)

    peaks = energy_peaks(mag_denoised, freqs, times, energy_cfg, top_k=6)

    report = {
        "variant": 14,
        "input": {"path": str(input_path), "sr": sr, "samples": int(y.shape[0]), "duration_sec": float(y.shape[0] / sr)},
        "config": cfg,
        "noise_profile": {"method": noise_cfg.method, "scale": noise_cfg.scale},
        "denoise": {"method": denoise_cfg.method, "floor_ratio": denoise_cfg.floor_ratio, "savgol": denoise_cfg.savgol.__dict__},
        "energy_peaks": peaks,
        "quality": {"psnr_db_input_vs_denoised": psnr_db(y, y_denoised)},
        "outputs": {
            "denoised_wav": str(out_dir / "denoised.wav"),
            "spectrogram_before_png": str(assets_dir / "spectrogram_before.png"),
            "spectrogram_after_png": str(assets_dir / "spectrogram_after.png"),
            "spectrogram_compare_png": str(assets_dir / "spectrogram_compare.png"),
        },
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    if args.demo:
        (assets_dir / "demo_report.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
