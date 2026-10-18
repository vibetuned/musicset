"""Real scans: reading PDF pages and measuring engraving weight.

The target of augmentation is what scanned editions look like, so their
measurable properties are kept here: IMSLP plate engravings (Schumann op. 68,
Schmitt, Pischna) draw staff lines at 0.15-0.31 of a staff space, binarized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    return len(pdfium.PdfDocument(str(path)))


def pdf_page(path: Path, page: int = 0, dpi: int = 300) -> Image.Image:
    """One page (0-based) of a PDF as an RGB image."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    return doc[page].render(scale=dpi / 72).to_pil().convert("RGB")


def load_scan(path: Path, page: int = 0, dpi: int = 300) -> Image.Image:
    """A PDF page or an image file."""
    if path.suffix.lower() == ".pdf":
        return pdf_page(path, page, dpi)
    return Image.open(path).convert("RGB")


@dataclass(frozen=True, slots=True)
class StaffProfile:
    line: float
    """Staff line thickness, pixels."""
    space: float
    """Distance between staff lines, pixels."""

    @property
    def ratio(self) -> float:
        """Line thickness / staff space: 0.056 in default Verovio renders."""
        return self.line / self.space


def skew_angle(grey: np.ndarray, max_degrees: float = 2.0, step: float = 0.1) -> float:
    """Page skew in degrees: the rotation that makes staff lines horizontal,
    found where the row darkness profile is sharpest (on a downscaled copy)."""
    h, w = grey.shape
    f = min(1.0, 800 / w)
    small = Image.fromarray((grey * 255).astype(np.uint8)).resize(
        (max(1, round(w * f)), max(1, round(h * f)))
    )
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_degrees, max_degrees + step / 2, step):
        rotated = np.asarray(small.rotate(float(angle), Image.Resampling.BILINEAR, fillcolor=255))
        score = float(np.var((rotated < 128).mean(axis=1)))
        if score > best_score:
            best, best_score = round(float(angle), 2), score
    return best


def staff_profile(image: Image.Image | np.ndarray, deskew: bool = True) -> StaffProfile | None:
    """Staff line weight from the rows that are dark across most of the page.

    Skewed pages (scans, augmented renders) are straightened first; None when
    no staff lines are found.
    """
    arr = np.asarray(image, dtype=np.float32)
    grey = arr.mean(axis=2) if arr.ndim == 3 else arr
    if grey.max() > 1.5:
        grey = grey / 255.0
    if deskew:
        angle = skew_angle(grey)
        if angle:
            pil = Image.fromarray((grey * 255).astype(np.uint8))
            grey = np.asarray(pil.rotate(angle, Image.Resampling.BILINEAR, fillcolor=255)) / 255.0
    rows = np.nonzero((grey < 0.5).mean(axis=1) > 0.35)[0]
    if len(rows) < 5:
        return None
    lines = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    centers = np.array([line.mean() for line in lines])
    gaps = np.diff(centers)
    gaps = gaps[(gaps > 4) & (gaps < 80)]
    if not len(gaps):
        return None
    return StaffProfile(float(np.median([len(line) for line in lines])), float(np.median(gaps)))
