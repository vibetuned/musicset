"""Scan-like augmentation of rendered pages, boxes included.

    from musicset.augment import Augmenter
    aug = Augmenter("mixed")
    out = aug(image, boxes, rng=seed)   # out.image (H, W, 3) uint8, out.boxes, out.keep

Presets follow the physical process (ink, paper, scanner, sensor):

* ``engraved``: IMSLP-style plate engravings scanned to black and white;
  ink spread, slight skew, blur, binarization. Calibrated on IMSLP scans
  (Schumann op. 68, Schmitt, Pischna: staff lines 0.14-0.33 of a staff space,
  median 0.25); with varied render profiles it measures median 0.23
  (``scripts/calibrate_augment.py``);
* ``aged``: old prints in greyscale or colour; faded ink, show-through,
  tinted and stained paper, bent pages, uneven light, sensor noise, JPEG;
* ``photo``: phone pictures; perspective, page bend, strong light gradients;
* ``mixed``: one of the above per sample (or none, 10% of the time).

Run augmentation on full-resolution pages, before any downscaling to a
model's input size: thin strokes are only a pixel wide even there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from musicset.augment.ops import (
    Binarize,
    Blur,
    Border,
    Downscale,
    Illumination,
    InkFade,
    InkSpread,
    Jpeg,
    Noise,
    Paper,
    Perspective,
    Rotate,
    Sample,
    ShowThrough,
    Speckle,
    Warp,
)

PRESETS: dict[str, list[Any]] = {
    "clean": [],
    "engraved": [
        # spread 0/1/2 px a side with p 0.16/0.68/0.16: the spread of real editions
        InkSpread((1, 1), p=0.8),
        InkSpread((1, 1), p=0.2),
        Rotate(0.8, p=0.6),
        Speckle(p=0.4),
        Blur((0.4, 1.0), p=0.6),
        Downscale((0.6, 0.85), p=0.2),
        Binarize((0.5, 0.72), p=0.85),
        Jpeg((60, 90), p=0.2),
    ],
    "aged": [
        InkSpread((0, 1), p=0.7),
        InkFade(p=0.4),
        ShowThrough(p=0.5),
        Paper(p=0.95),
        Rotate(1.5, p=0.7),
        Warp(p=0.4),
        Border(p=0.15),
        Blur((0.4, 1.2), p=0.6),
        Downscale(p=0.3),
        Illumination(p=0.6),
        Speckle(p=0.4),
        Noise(p=0.6),
        Jpeg(p=0.5),
    ],
    "photo": [
        InkSpread((0, 1), p=0.5),
        Paper((0.01, 0.03), (0.0, 0.05), p=0.8),
        Perspective(0.03, p=0.9),
        Warp((0.003, 0.012), p=0.6),
        Blur((0.6, 1.6), p=0.7),
        Illumination((0.15, 0.4), p=0.9),
        Noise((0.02, 0.05), p=0.8),
        Jpeg((40, 80), p=0.8),
    ],
}
MIXED = {"clean": 0.1, "engraved": 0.4, "aged": 0.35, "photo": 0.15}


@dataclass
class Augmented:
    image: np.ndarray
    """(H, W, 3) uint8."""
    boxes: np.ndarray
    """(N, 4) float32 x1, y1, x2, y2 in pixels, clipped to the image."""
    keep: np.ndarray
    """(N,) bool: the box still covers at least a pixel of the image."""
    preset: str
    log: list[str] = field(default_factory=list)


class Augmenter:
    def __init__(self, preset: str = "mixed", strength: float = 1.0) -> None:
        if preset != "mixed" and preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}: {[*PRESETS, 'mixed']}")
        self.preset = preset
        self.strength = strength

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray | list | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> Augmented:
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        name = self.preset
        if name == "mixed":
            names, weights = zip(*MIXED.items(), strict=True)
            name = str(rng.choice(names, p=np.array(weights) / sum(weights)))

        img = np.asarray(image)
        grey = img.mean(axis=2) if img.ndim == 3 else img
        boxes_arr = np.asarray(boxes if boxes is not None else [], dtype=np.float32).reshape(-1, 4)
        sample = Sample((grey / 255.0).astype(np.float32), boxes_arr.copy())
        for op in PRESETS[name]:
            if rng.random() < op.p:
                op(sample, rng, self.strength)

        out = np.clip(sample.image, 0, 1)
        if out.ndim == 2:
            out = np.repeat(out[..., None], 3, axis=2)
        h, w = out.shape[:2]
        b = sample.boxes
        if len(b):
            b = np.stack(
                [b[:, 0].clip(0, w), b[:, 1].clip(0, h), b[:, 2].clip(0, w), b[:, 3].clip(0, h)],
                axis=1,
            )
        keep = (
            ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1) if len(b) else np.zeros(0, bool)
        )
        return Augmented((out * 255).round().astype(np.uint8), b, keep, name, sample.log)


__all__ = ["MIXED", "PRESETS", "Augmented", "Augmenter"]
