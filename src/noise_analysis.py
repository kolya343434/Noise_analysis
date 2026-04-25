from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter


NoiseType = Literal["gaussian", "salt_pepper"]
DenoiseMethod = Literal["median", "gaussian_blur"]


@dataclass(frozen=True)
class NoiseConfig:
    type: NoiseType = "gaussian"
    sigma: float = 14.0
    salt_pepper_prob: float = 0.02


@dataclass(frozen=True)
class DenoiseConfig:
    method: DenoiseMethod = "median"
    kernel_size: int = 3
    blur_radius: float = 1.0


def load_grayscale(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def save_grayscale(image_u8: np.ndarray, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u8.astype(np.uint8), mode="L").save(p)


def generate_clean_demo_image(size: int = 256) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    gradient = np.clip((x / (size - 1)) * 255, 0, 255).astype(np.float32)
    circle = (((x - size / 2) ** 2 + (y - size / 2) ** 2) < (size / 4) ** 2).astype(np.float32) * 70.0
    img = np.clip(gradient + circle, 0, 255)
    return img.astype(np.uint8)


def add_gaussian_noise(image_u8: np.ndarray, sigma: float, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=image_u8.shape)
    out = np.clip(image_u8.astype(np.float64) + noise, 0, 255)
    return out.astype(np.uint8)


def add_salt_pepper_noise(image_u8: np.ndarray, prob: float, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = float(prob)
    if p < 0 or p > 1:
        raise ValueError("salt_pepper_prob must be in [0..1].")
    r = rng.random(image_u8.shape)
    out = image_u8.copy()
    out[r < (p / 2)] = 0
    out[r > 1 - (p / 2)] = 255
    return out


def apply_noise(image_u8: np.ndarray, cfg: NoiseConfig) -> np.ndarray:
    if cfg.type == "gaussian":
        return add_gaussian_noise(image_u8, cfg.sigma)
    if cfg.type == "salt_pepper":
        return add_salt_pepper_noise(image_u8, cfg.salt_pepper_prob)
    raise ValueError(f"Unsupported noise type: {cfg.type}")


def denoise(image_u8: np.ndarray, cfg: DenoiseConfig) -> np.ndarray:
    img = Image.fromarray(image_u8, mode="L")
    if cfg.method == "median":
        k = int(cfg.kernel_size)
        if k < 3 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3.")
        out = img.filter(ImageFilter.MedianFilter(size=k))
        return np.asarray(out, dtype=np.uint8)
    if cfg.method == "gaussian_blur":
        out = img.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_radius)))
        return np.asarray(out, dtype=np.uint8)
    raise ValueError(f"Unsupported denoise method: {cfg.method}")


def mse(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    diff = a_u8.astype(np.float64) - b_u8.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(ref_u8: np.ndarray, test_u8: np.ndarray, *, peak: float = 255.0) -> float:
    v = mse(ref_u8, test_u8)
    if v == 0:
        return float("inf")
    return float(10.0 * math.log10((peak * peak) / v))


def basic_stats(image_u8: np.ndarray) -> dict[str, float]:
    x = image_u8.astype(np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "var": float(x.var(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def estimate_noise_sigma_gaussian(image_u8: np.ndarray) -> float:
    """
    Simple robust estimate for (roughly) additive Gaussian noise:
    sigma ≈ 1.4826 * MAD(residual), where residual = image - median_filter(image).
    """
    base = denoise(image_u8, DenoiseConfig(method="median", kernel_size=3))
    residual = image_u8.astype(np.int16) - base.astype(np.int16)
    mad = float(np.median(np.abs(residual)))
    return float(1.4826 * mad)


def analyze(
    *,
    clean_u8: np.ndarray,
    noisy_u8: np.ndarray,
    denoised_u8: np.ndarray | None = None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "clean": basic_stats(clean_u8),
        "noisy": basic_stats(noisy_u8),
        "sigma_est_noisy": estimate_noise_sigma_gaussian(noisy_u8),
        "mse_clean_noisy": mse(clean_u8, noisy_u8),
        "psnr_clean_noisy": psnr(clean_u8, noisy_u8),
    }
    if denoised_u8 is not None:
        out["denoised"] = basic_stats(denoised_u8)
        out["sigma_est_denoised"] = estimate_noise_sigma_gaussian(denoised_u8)
        out["mse_clean_denoised"] = mse(clean_u8, denoised_u8)
        out["psnr_clean_denoised"] = psnr(clean_u8, denoised_u8)
    return out
