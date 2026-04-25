from __future__ import annotations

import argparse
import json
from pathlib import Path

from noise_analysis import (
    DenoiseConfig,
    NoiseConfig,
    analyze,
    apply_noise,
    basic_stats,
    denoise,
    estimate_noise_sigma_gaussian,
    generate_clean_demo_image,
    load_grayscale,
    save_grayscale,
)


def _load_variant14(repo_root: Path) -> tuple[NoiseConfig, DenoiseConfig]:
    data = json.loads((repo_root / "config" / "variant14.json").read_text(encoding="utf-8"))
    noise = data["noise"]
    den = data["denoise"]
    return (
        NoiseConfig(
            type=str(noise["type"]),  # type: ignore[arg-type]
            sigma=float(noise.get("sigma", 14.0)),
            salt_pepper_prob=float(noise.get("salt_pepper_prob", 0.02)),
        ),
        DenoiseConfig(
            method=str(den["method"]),  # type: ignore[arg-type]
            kernel_size=int(den.get("kernel_size", 3)),
            blur_radius=float(den.get("blur_radius", 1.0)),
        ),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Lab 9: Noise analysis (variant 14).")
    parser.add_argument("--image", type=str, help="Path to input image (grayscale or RGB).")
    parser.add_argument("--variant", type=int, default=14, help="Variant number (default: 14).")
    parser.add_argument("--demo", action="store_true", help="Generate demo images, analyze, and write assets.")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising step.")
    parser.add_argument("--json-out", type=str, default=None, help="Write analysis JSON to a file.")
    args = parser.parse_args()

    if args.variant != 14:
        raise SystemExit("Only variant 14 is preconfigured in this repo.")

    noise_cfg, denoise_cfg = _load_variant14(repo_root)

    if args.demo:
        assets = repo_root / "assets"
        clean = generate_clean_demo_image()
        noisy = apply_noise(clean, noise_cfg)
        denoised = None if args.no_denoise else denoise(noisy, denoise_cfg)

        save_grayscale(clean, assets / "clean.png")
        save_grayscale(noisy, assets / "noisy.png")
        if denoised is not None:
            save_grayscale(denoised, assets / "denoised.png")

        report = {
            "variant": 14,
            "noise_config": noise_cfg.__dict__,
            "denoise_config": denoise_cfg.__dict__,
            "analysis": analyze(clean_u8=clean, noisy_u8=noisy, denoised_u8=denoised),
        }
        (assets / "demo_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if not args.image:
        raise SystemExit("Provide --image PATH or use --demo.")

    # For a user image we don't have a clean reference; still compute stats + sigma estimate.
    img = load_grayscale(args.image)
    denoised = None if args.no_denoise else denoise(img, denoise_cfg)
    report = {
        "variant": 14,
        "denoise_config": denoise_cfg.__dict__,
        "analysis_no_reference": {"input": basic_stats(img), "sigma_est_input": estimate_noise_sigma_gaussian(img)},
    }
    # Lightweight: print only denoised stats too.
    if denoised is not None:
        report["analysis_no_reference"]["denoised"] = basic_stats(denoised)  # type: ignore[index]
        report["analysis_no_reference"]["sigma_est_denoised"] = estimate_noise_sigma_gaussian(denoised)  # type: ignore[index]

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
