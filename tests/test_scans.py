import numpy as np
import pytest
from PIL import Image

from musicset.scans import load_scan, skew_angle, staff_profile


def _staves(thickness: int = 3, space: int = 18) -> Image.Image:
    a = np.full((800, 600), 255, np.uint8)
    for top in (100, 400):
        for k in range(5):
            y = top + k * space
            a[y : y + thickness, 40:560] = 0
    return Image.fromarray(a)


def test_staff_profile_measures_line_weight():
    profile = staff_profile(_staves(3, 18))
    assert profile is not None
    assert profile.line == 3 and profile.space == 18
    assert profile.ratio == pytest.approx(3 / 18)


def test_skewed_pages_are_straightened_first():
    skewed = _staves(3, 18).rotate(1.2, Image.Resampling.BILINEAR, fillcolor=255)
    assert skew_angle(np.asarray(skewed, np.float32) / 255) == pytest.approx(-1.2, abs=0.11)
    profile = staff_profile(skewed)
    assert profile is not None and profile.ratio == pytest.approx(3 / 18, abs=0.03)
    assert staff_profile(np.full((100, 100), 255, np.uint8)) is None


def test_pdf_pages_load(tmp_path):
    _staves().convert("RGB").save(tmp_path / "scan.pdf")
    image = load_scan(tmp_path / "scan.pdf", 0, dpi=72)
    assert image.size == (600, 800)
