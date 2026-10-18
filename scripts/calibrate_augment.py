"""Stroke weight of augmented renders against real scans.

Staff line thickness / staff space, measured on rendered pages after each
preset's ink and scanner steps (geometric steps skipped: the measurement needs
horizontal staves), next to the same measurement on reference scans.

    uv run python scripts/calibrate_augment.py [RENDERED_IMGS] [SCANS_DIR]
    (defaults: examples/build/imgs, examples/imslp)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from musicset.augment import PRESETS
from musicset.augment.ops import Perspective, Rotate, Sample, Warp
from musicset.scans import load_scan, pdf_page_count, staff_profile

GEOMETRIC = (Rotate, Perspective, Warp)


def summary(ratios: list[float]) -> str:
    r = np.array([v for v in ratios if np.isfinite(v)])
    if not len(r):
        return "no staves found"
    return (
        f"median {np.median(r):.3f}  p10-p90 {np.percentile(r, 10):.3f}-{np.percentile(r, 90):.3f}"
    )


def ratio(image) -> float:
    profile = staff_profile(image)
    return profile.ratio if profile else float("nan")


def main() -> None:
    renders = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/build/imgs")
    scans = Path(sys.argv[2] if len(sys.argv) > 2 else "examples/imslp")

    reference = []
    for pdf in sorted(scans.glob("*.pdf")):
        n = pdf_page_count(pdf)
        reference += [ratio(load_scan(pdf, p)) for p in range(2, n, max(1, n // 8))]
    print(f"{'scans':9} {summary(reference)}  ({len(reference)} pages from {scans})")

    pages = sorted(renders.glob("*.png"))[:12]
    images = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255 for p in pages]
    print(f"{'clean':9} {summary([ratio(g) for g in images])}")
    for name, ops in PRESETS.items():
        if not ops:
            continue
        ratios = []
        for i, grey in enumerate(images):
            for seed in range(3):
                rng = np.random.default_rng(1000 * i + seed)
                s = Sample(grey.copy(), np.zeros((0, 4), np.float32))
                for op in ops:
                    if rng.random() < op.p and not isinstance(op, GEOMETRIC):
                        op(s, rng, 1.0)
                ratios.append(ratio(s.image))
        print(f"{name:9} {summary(ratios)}")


if __name__ == "__main__":
    main()
