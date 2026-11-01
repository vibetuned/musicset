"""MNIST-like isolated symbol dataset for fingerings (1-5), ornaments,
articulations, accidentals, dynamics and pedal marks.

Two sources feed the dataset, alone or combined:

* **Synthetic symbols** (:func:`generate_symbol_samples`): SMuFL glyphs from
  every music font Verovio ships (Bravura, Leipzig, Gootville, Leland,
  Petaluma) plus stylus-drawn stroke templates for fingerings 1-5 and common
  score markings, varied with affine jitter, elastic stroke deformation,
  thickness/pressure modulation and scan/tablet noise.
* **Score crops** (:func:`extract_dataset_symbols`): bounding-box crops pulled
  off rendered pages of a built ``musicset`` dataset (``pages/`` + ``imgs/``,
  or ``annotations/`` + ``imgs/``), with ``fing`` elements split into
  ``fing1``..``fing5`` by their engraved text.

Outputs both a single compressed ``dataset.npz`` archive (``x_train``,
``y_train``, ``x_val``, ``y_val``, ``x_test``, ``y_test``, ``classes``) and an
``ImageFolder`` directory tree (``train/<class>/*.png``, ``val/``, ``test/``)
with ``labels.json``.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from svgpathtools import parse_path

from musicset.annotate.geometry import BBox
from musicset.annotate.schema import PageAnnotation
from musicset.annotate.smufl import available_fonts, verovio_data_dir
from musicset.pipeline import PAGE_SUFFIX, page_file_stem

SYMBOL_GROUPS = ("fingering", "ornament", "articulation", "accidental", "dynamic", "pedal")
PRESETS = ("mixed", "tablet", "score", "clean")


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    name: str
    group: str
    codes: tuple[str, ...]
    """SMuFL codepoints representing this class across Verovio's fonts."""
    scale_range: tuple[float, float] = (0.58, 0.82)
    """Fraction of the target canvas the symbol's bounding box spans."""
    handwritten: bool = False
    """Whether a stylus stroke template is also available for tablet synthesis."""


SYMBOLS: tuple[SymbolSpec, ...] = (
    # Fingerings 1-5 (timeSig, tuplet, figured bass, light/narrow SMuFL digits + stylus strokes)
    SymbolSpec(
        "fing1",
        "fingering",
        ("E081", "E881", "EA51", "F4CD", "F441", "F45E", "F507"),
        scale_range=(0.55, 0.82),
        handwritten=True,
    ),
    SymbolSpec(
        "fing2",
        "fingering",
        ("E082", "E882", "EA52", "F4CE", "F442", "F45F", "F508"),
        scale_range=(0.55, 0.82),
        handwritten=True,
    ),
    SymbolSpec(
        "fing3",
        "fingering",
        ("E083", "E883", "EA54", "F4CF", "F443", "F460", "F509"),
        scale_range=(0.55, 0.82),
        handwritten=True,
    ),
    SymbolSpec(
        "fing4",
        "fingering",
        ("E084", "E884", "EA55", "F4D0", "F444", "F461", "F50A"),
        scale_range=(0.55, 0.82),
        handwritten=True,
    ),
    SymbolSpec(
        "fing5",
        "fingering",
        ("E085", "E885", "EA57", "F4D1", "F445", "F462", "F50B"),
        scale_range=(0.55, 0.82),
        handwritten=True,
    ),
    # Ornaments
    SymbolSpec("trill", "ornament", ("E566",), scale_range=(0.58, 0.84), handwritten=True),
    SymbolSpec(
        "mordent",
        "ornament",
        ("E56D", "E56C", "E56E"),
        scale_range=(0.60, 0.85),
        handwritten=True,
    ),
    SymbolSpec(
        "turn",
        "ornament",
        ("E567", "E568", "E569"),
        scale_range=(0.60, 0.85),
        handwritten=True,
    ),
    SymbolSpec(
        "fermata",
        "ornament",
        ("E4C0", "E4C1"),
        scale_range=(0.58, 0.84),
        handwritten=True,
    ),
    SymbolSpec("arpeg", "ornament", ("EAA9",), scale_range=(0.65, 0.88), handwritten=True),
    # Articulations
    SymbolSpec(
        "articStaccatoAbove",
        "articulation",
        ("E4A2", "E4A3"),
        scale_range=(0.22, 0.52),
        handwritten=True,
    ),
    SymbolSpec(
        "articTenutoAbove",
        "articulation",
        ("E4A4", "E4A5"),
        scale_range=(0.50, 0.78),
        handwritten=True,
    ),
    SymbolSpec(
        "articAccentAbove",
        "articulation",
        ("E4A0", "E4A1"),
        scale_range=(0.55, 0.82),
        handwritten=True,
    ),
    SymbolSpec(
        "articStaccatissimoAbove",
        "articulation",
        ("E4A6", "E4A7", "E4A8", "E4A9", "E4AA", "E4AB"),
        scale_range=(0.45, 0.76),
        handwritten=True,
    ),
    SymbolSpec(
        "articMarcatoAbove",
        "articulation",
        ("E4AC", "E4AD"),
        scale_range=(0.52, 0.80),
        handwritten=True,
    ),
    # Accidentals
    SymbolSpec(
        "accidentalFlat",
        "accidental",
        ("E260", "F427"),
        scale_range=(0.60, 0.86),
        handwritten=True,
    ),
    SymbolSpec(
        "accidentalNatural",
        "accidental",
        ("E261", "F428"),
        scale_range=(0.60, 0.86),
        handwritten=True,
    ),
    SymbolSpec(
        "accidentalSharp",
        "accidental",
        ("E262", "F429"),
        scale_range=(0.60, 0.86),
        handwritten=True,
    ),
    SymbolSpec(
        "accidentalDoubleSharp",
        "accidental",
        ("E263",),
        scale_range=(0.52, 0.78),
        handwritten=True,
    ),
    SymbolSpec(
        "accidentalDoubleFlat",
        "accidental",
        ("E264", "F4A1"),
        scale_range=(0.62, 0.86),
        handwritten=True,
    ),
    # Dynamics
    SymbolSpec("dynamicPiano", "dynamic", ("E520", "F46B"), scale_range=(0.56, 0.82)),
    SymbolSpec("dynamicMezzo", "dynamic", ("E521", "F46C"), scale_range=(0.54, 0.80)),
    SymbolSpec("dynamicForte", "dynamic", ("E522", "F46D"), scale_range=(0.60, 0.86)),
    SymbolSpec("dynamicSforzando", "dynamic", ("E524", "E536", "F46F"), scale_range=(0.54, 0.80)),
    SymbolSpec("dynamicZ", "dynamic", ("E525", "F470"), scale_range=(0.52, 0.78)),
    SymbolSpec("dynamicPP", "dynamic", ("E52B",), scale_range=(0.64, 0.88)),
    SymbolSpec("dynamicMP", "dynamic", ("E52C",), scale_range=(0.64, 0.88)),
    SymbolSpec("dynamicMF", "dynamic", ("E52D",), scale_range=(0.64, 0.88)),
    SymbolSpec("dynamicFF", "dynamic", ("E52F",), scale_range=(0.64, 0.88)),
    SymbolSpec("dynamicFFF", "dynamic", ("E530",), scale_range=(0.68, 0.90)),
    SymbolSpec("dynamicForzando", "dynamic", ("E535",), scale_range=(0.62, 0.86)),
    SymbolSpec("dynamicSforzato", "dynamic", ("E539",), scale_range=(0.64, 0.88)),
    # Pedal
    SymbolSpec(
        "pedal",
        "pedal",
        ("E650", "E655", "F434"),
        scale_range=(0.60, 0.86),
        handwritten=True,
    ),
)

SYMBOL_BY_NAME: dict[str, SymbolSpec] = {s.name: s for s in SYMBOLS}

# Element classes in PageAnnotation that map directly to a symbol class
_ELEMENT_SYMBOLS = frozenset(("trill", "mordent", "turn", "fermata", "arpeg", "pedal"))

# SMuFL codepoint -> symbol class for page glyph extraction
_GLYPH_TO_SYMBOL: dict[str, str] = {
    code: spec.name
    for spec in SYMBOLS
    if spec.group in ("articulation", "accidental", "dynamic", "ornament", "pedal")
    for code in spec.codes
}


def symbol_specs(groups: Sequence[str] = SYMBOL_GROUPS) -> tuple[SymbolSpec, ...]:
    """Symbol specs belonging to ``groups`` in canonical order."""
    unknown = set(groups) - set(SYMBOL_GROUPS)
    if unknown:
        raise ValueError(f"unknown symbol group(s) {sorted(unknown)}: {SYMBOL_GROUPS}")
    wanted = set(groups)
    return tuple(s for s in SYMBOLS if s.group in wanted)


# =============================================================================
# SMuFL vector outline extraction & rendering
# =============================================================================


@cache
def _glyph_Contours(font: str, code: str) -> tuple[tuple[np.ndarray, int], ...] | None:
    """Closed polygon contours ``(pts_Nx2, winding_sign)`` of a SMuFL glyph."""
    xml_path = verovio_data_dir() / font / f"{code}.xml"
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    path_el = root.find("path")
    if path_el is None or not path_el.get("d"):
        return None
    flip_y = -1.0 if "scale(1,-1)" in path_el.get("transform", "").replace(" ", "") else 1.0
    parsed = parse_path(path_el.get("d", ""))
    contours: list[tuple[np.ndarray, int]] = []
    for sp in parsed.continuous_subpaths():
        if len(sp) == 0:
            continue
        pts: list[tuple[float, float]] = []
        for seg in sp:
            length = seg.length()
            steps = max(4, min(28, int(length / 10.0) + 2))
            for t in np.linspace(0.0, 1.0, steps, endpoint=False):
                z = seg.point(float(t))
                pts.append((float(z.real), float(z.imag) * flip_y))
        if len(pts) >= 3:
            arr = np.array(pts, dtype=np.float64)
            x, y = arr[:, 0], arr[:, 1]
            area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            contours.append((arr, 1 if area >= 0 else -1))
    return tuple(contours) if contours else None


@cache
def _available_font_codes(code: str) -> tuple[tuple[str, str], ...]:
    """All ``(font, code)`` pairs available in Verovio for ``code``."""
    return tuple(
        (font, code) for font in available_fonts() if _glyph_Contours(font, code) is not None
    )


def _arpeggio_contours(font: str, segments: int, arrow: bool) -> list[tuple[np.ndarray, int]]:
    """Vertical arpeggio built the way Verovio engraves ``<g class="arpeg">``:
    ``EAA9`` repeated horizontally and rotated -90 degrees."""
    base = _glyph_Contours(font, "EAA9") or _glyph_Contours("Bravura", "EAA9")
    assert base is not None
    all_pts = np.vstack([pts for pts, _ in base])
    step = max(120.0, float(all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.82)
    tiled: list[tuple[np.ndarray, int]] = []
    for k in range(segments):
        for pts, sign in base:
            shifted = pts + np.array([k * step, 0.0])
            # rotate -90 degrees: (x, y) -> (y, -x), which preserves orientation
            rotated = np.column_stack([shifted[:, 1], -shifted[:, 0]])
            tiled.append((rotated, sign))
    if arrow:
        cap = _glyph_Contours(font, "EAAD") or _glyph_Contours("Bravura", "EAAD")
        if cap is not None:
            for pts, sign in cap:
                shifted = pts + np.array([segments * step, 0.0])
                rotated = np.column_stack([shifted[:, 1], -shifted[:, 0]])
                tiled.append((rotated, sign))
    return tiled


def _rasterize_contours(
    contours: Sequence[tuple[np.ndarray, int]],
    canvas: int,
    target_fraction: float,
    rng: np.random.Generator,
    *,
    jitter: float = 1.0,
) -> np.ndarray:
    """Rasterize signed contours to a ``(canvas, canvas)`` float32 image (1.0=paper, 0.0=ink)."""
    all_pts = np.vstack([pts for pts, _ in contours])
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    centered = [(pts - [cx, cy], sign) for pts, sign in contours]

    # Affine perturbation in contour space before rasterization
    if jitter > 0:
        angle = float(rng.uniform(-0.18, 0.18) * jitter)
        sx = float(rng.uniform(1.0 - 0.14 * jitter, 1.0 + 0.14 * jitter))
        sy = float(rng.uniform(1.0 - 0.14 * jitter, 1.0 + 0.14 * jitter))
        shear = float(rng.uniform(-0.14, 0.14) * jitter)
        c, s = math.cos(angle), math.sin(angle)
        mat = np.array([[c, -s], [s, c]], dtype=np.float64) @ np.array(
            [[sx, shear], [0.0, sy]], dtype=np.float64
        )
        centered = [(pts @ mat.T, sign) for pts, sign in centered]

    trans_pts = np.vstack([pts for pts, _ in centered])
    txmin, tymin = trans_pts.min(axis=0)
    txmax, tymax = trans_pts.max(axis=0)
    span = max(1e-3, float(txmax - txmin), float(tymax - tymin))
    scale = (canvas * target_fraction) / span
    tcx, tcy = 0.5 * (txmin + txmax), 0.5 * (tymin + tymax)

    max_shift = canvas * 0.08 * jitter
    dx = float(rng.uniform(-max_shift, max_shift)) if max_shift > 0 else 0.0
    dy = float(rng.uniform(-max_shift, max_shift)) if max_shift > 0 else 0.0

    winding = np.zeros((canvas, canvas), dtype=np.int32)
    for pts, sign in centered:
        mapped = (pts - [tcx, tcy]) * scale + (canvas * 0.5) + [dx, dy]
        mask = Image.new("L", (canvas, canvas), 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in mapped.tolist()], fill=1)
        winding += sign * np.asarray(mask, dtype=np.int32)
    return 1.0 - (winding != 0).astype(np.float32)


# =============================================================================
# Stylus / handwritten stroke templates (for tablet input diversity)
# =============================================================================


def _bezier(
    p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], n: int = 16
) -> list[tuple[float, float]]:
    ts = np.linspace(0.0, 1.0, n)
    return [
        (
            float((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]),
            float((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]),
        )
        for t in ts
    ]


def _handwritten_strokes(name: str, rng: np.random.Generator) -> list[list[tuple[float, float]]]:
    """Normalized strokes in ``[-1, 1] x [-1, 1]`` for stylus-drawn symbols."""
    j = lambda s=0.06: float(rng.normal(0.0, s))  # noqa: E731
    if name == "fing1":
        style = int(rng.integers(3))
        top = (0.04 + j(), -0.82 + j())
        bot = (0.0 + j(), 0.82 + j())
        mid = (0.02 + j(0.04), 0.0 + j(0.04))
        strokes = [_bezier(top, mid, bot)]
        if style >= 1:  # top serif
            strokes.append([(-0.34 + j(), -0.46 + j()), top])
        if style == 2:  # base bar
            strokes.append([(-0.36 + j(), 0.82 + j()), (0.36 + j(), 0.82 + j())])
        return strokes
    if name == "fing2":
        p0 = (-0.52 + j(), -0.45 + j())
        p1 = (0.05 + j(), -0.98 + j())
        p2 = (0.55 + j(), -0.38 + j())
        p3 = (-0.55 + j(), 0.78 + j())
        p4 = (0.58 + j(), 0.78 + j())
        return [_bezier(p0, p1, p2, 14) + _bezier(p2, (0.25 + j(), 0.15 + j()), p3, 14) + [p4]]
    if name == "fing3":
        if rng.random() < 0.35:  # flat-top 3
            top_l, top_r = (-0.48 + j(), -0.78 + j()), (0.48 + j(), -0.78 + j())
            waist = (-0.08 + j(), -0.05 + j())
            bot = _bezier(waist, (0.72 + j(), 0.15 + j()), (0.45 + j(), 0.65 + j()), 12) + _bezier(
                (0.45 + j(), 0.65 + j()), (-0.05 + j(), 0.95 + j()), (-0.55 + j(), 0.58 + j()), 12
            )
            return [[top_l, top_r, waist], bot]
        waist = (-0.08 + j(), -0.02 + j())
        upper = _bezier(
            (-0.48 + j(), -0.58 + j()), (0.05 + j(), -0.96 + j()), (0.48 + j(), -0.45 + j()), 12
        ) + _bezier((0.48 + j(), -0.45 + j()), (0.45 + j(), -0.08 + j()), waist, 10)
        lower = _bezier(waist, (0.62 + j(), 0.08 + j()), (0.50 + j(), 0.58 + j()), 12) + _bezier(
            (0.50 + j(), 0.58 + j()), (0.0 + j(), 0.96 + j()), (-0.52 + j(), 0.55 + j()), 12
        )
        return [upper + lower]
    if name == "fing4":
        if rng.random() < 0.5:  # open-top 4
            return [
                [(-0.38 + j(), -0.76 + j()), (-0.48 + j(), 0.18 + j()), (0.56 + j(), 0.18 + j())],
                [(0.18 + j(), -0.78 + j()), (0.14 + j(), 0.82 + j())],
            ]
        return [
            [(0.20 + j(), -0.80 + j()), (-0.54 + j(), 0.20 + j()), (0.56 + j(), 0.20 + j())],
            [(0.20 + j(), -0.80 + j()), (0.18 + j(), 0.82 + j())],
        ]
    if name == "fing5":
        top_r = (0.48 + j(), -0.78 + j())
        top_l = (-0.42 + j(), -0.78 + j())
        corner = (-0.46 + j(), -0.06 + j())
        belly = _bezier(corner, (0.62 + j(), -0.22 + j()), (0.52 + j(), 0.48 + j()), 14) + _bezier(
            (0.52 + j(), 0.48 + j()), (0.05 + j(), 0.94 + j()), (-0.52 + j(), 0.56 + j()), 14
        )
        return [[top_r, top_l, corner, *belly]]
    if name == "trill":
        # 't' stem + crossbar + 'r' arch
        t_stem = _bezier(
            (-0.42 + j(), -0.78 + j()), (-0.45 + j(), 0.10 + j()), (-0.25 + j(), 0.65 + j()), 14
        )
        t_bar = [(-0.68 + j(), -0.28 + j()), (-0.12 + j(), -0.28 + j())]
        r_stem = [(-0.02 + j(), -0.18 + j()), (-0.05 + j(), 0.65 + j())]
        r_arch = _bezier(
            (-0.02 + j(), 0.10 + j()), (0.22 + j(), -0.32 + j()), (0.58 + j(), -0.12 + j()), 12
        )
        return [t_stem, t_bar, r_stem, r_arch]
    if name == "mordent":
        pts = [
            (-0.78 + j(), 0.18 + j()),
            (-0.38 + j(), -0.28 + j()),
            (-0.05 + j(), 0.22 + j()),
            (0.35 + j(), -0.28 + j()),
            (0.75 + j(), 0.18 + j()),
        ]
        strokes = [pts]
        if rng.random() < 0.65:  # vertical bar through mordent
            strokes.append([(0.0 + j(), -0.68 + j()), (0.0 + j(), 0.68 + j())])
        return strokes
    if name == "turn":
        s_left = _bezier(
            (-0.72 + j(), -0.05 + j()), (-0.42 + j(), -0.68 + j()), (0.0 + j(), 0.0 + j()), 14
        )
        s_right = _bezier(
            (0.0 + j(), 0.0 + j()), (0.42 + j(), 0.68 + j()), (0.72 + j(), 0.05 + j()), 14
        )
        return [s_left + s_right]
    if name == "fermata":
        flip = -1.0 if rng.random() < 0.25 else 1.0
        arch = _bezier(
            (-0.78 + j(), 0.32 * flip + j()),
            (0.0 + j(), -0.85 * flip + j()),
            (0.78 + j(), 0.32 * flip + j()),
            20,
        )
        dot = _bezier((-0.06, 0.18 * flip), (0.0, 0.12 * flip), (0.06, 0.18 * flip), 6)
        return [arch, dot]
    if name == "arpeg":
        waves = int(rng.integers(3, 6))
        ts = np.linspace(-0.82, 0.82, waves * 10)
        amp = float(rng.uniform(0.14, 0.24))
        pts = [(float(amp * math.sin(waves * math.pi * (y + 0.82) / 1.64)), float(y)) for y in ts]
        return [pts]
    if name == "articStaccatoAbove":
        r = float(rng.uniform(0.08, 0.16))
        angles = np.linspace(0, 2 * math.pi, 14)
        return [[(float(r * math.cos(a)), float(r * math.sin(a))) for a in angles]]
    if name == "articTenutoAbove":
        return [[(-0.62 + j(), 0.0 + j(0.04)), (0.62 + j(), 0.0 + j(0.04))]]
    if name == "articAccentAbove":
        tip = (0.62 + j(), 0.0 + j())
        return [[(-0.58 + j(), -0.38 + j()), tip, (-0.58 + j(), 0.38 + j())]]
    if name == "articStaccatissimoAbove":
        return [
            [
                (-0.10 + j(0.03), -0.52 + j()),
                (0.0 + j(0.03), 0.52 + j()),
                (0.10 + j(0.03), -0.52 + j()),
            ]
        ]
    if name == "articMarcatoAbove":
        flip = -1.0 if rng.random() < 0.25 else 1.0
        return [
            [
                (-0.48 + j(), 0.42 * flip + j()),
                (0.0 + j(), -0.45 * flip + j()),
                (0.48 + j(), 0.42 * flip + j()),
            ]
        ]
    if name == "accidentalFlat":
        stem = [(-0.28 + j(), -0.85 + j()), (-0.28 + j(), 0.72 + j())]
        loop = _bezier(
            (-0.28 + j(), 0.05 + j()), (0.48 + j(), -0.12 + j()), (0.35 + j(), 0.42 + j()), 12
        ) + _bezier(
            (0.35 + j(), 0.42 + j()), (0.12 + j(), 0.75 + j()), (-0.28 + j(), 0.72 + j()), 10
        )
        return [stem, loop]
    if name == "accidentalNatural":
        return [
            [(-0.24 + j(), -0.82 + j()), (-0.24 + j(), 0.42 + j())],
            [(0.24 + j(), -0.42 + j()), (0.24 + j(), 0.82 + j())],
            [(-0.28 + j(), -0.22 + j()), (0.28 + j(), -0.38 + j())],
            [(-0.28 + j(), 0.38 + j()), (0.28 + j(), 0.22 + j())],
        ]
    if name == "accidentalSharp":
        return [
            [(-0.22 + j(), -0.78 + j()), (-0.22 + j(), 0.82 + j())],
            [(0.22 + j(), -0.82 + j()), (0.22 + j(), 0.78 + j())],
            [(-0.52 + j(), -0.18 + j()), (0.52 + j(), -0.36 + j())],
            [(-0.52 + j(), 0.36 + j()), (0.52 + j(), 0.18 + j())],
        ]
    if name == "accidentalDoubleSharp":
        return [
            [(-0.52 + j(), -0.52 + j()), (0.52 + j(), 0.52 + j())],
            [(-0.52 + j(), 0.52 + j()), (0.52 + j(), -0.52 + j())],
        ]
    if name == "accidentalDoubleFlat":
        strokes: list[list[tuple[float, float]]] = []
        for ox in (-0.32, 0.18):
            stem = [(ox - 0.14 + j(0.03), -0.82 + j()), (ox - 0.14 + j(0.03), 0.70 + j())]
            loop = _bezier((ox - 0.14, 0.05), (ox + 0.30, -0.08), (ox + 0.22, 0.40), 10) + _bezier(
                (ox + 0.22, 0.40), (ox + 0.06, 0.70), (ox - 0.14, 0.70), 8
            )
            strokes.extend([stem, loop])
        return strokes
    # pedal star '*'
    return [
        [(-0.55 + j(), 0.0 + j()), (0.55 + j(), 0.0 + j())],
        [(0.0 + j(), -0.55 + j()), (0.0 + j(), 0.55 + j())],
        [(-0.40 + j(), -0.40 + j()), (0.40 + j(), 0.40 + j())],
        [(-0.40 + j(), 0.40 + j()), (0.40 + j(), -0.40 + j())],
    ]


def _render_strokes(
    strokes: Sequence[Sequence[tuple[float, float]]],
    canvas: int,
    target_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render stylus strokes onto a ``(canvas, canvas)`` float32 canvas (1.0=paper, 0.0=ink)."""
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    scale = canvas * 0.5 * target_fraction
    cx = canvas * 0.5 + float(rng.uniform(-canvas * 0.06, canvas * 0.06))
    cy = canvas * 0.5 + float(rng.uniform(-canvas * 0.06, canvas * 0.06))
    width = max(2, round(canvas * float(rng.uniform(0.055, 0.115))))
    for stroke in strokes:
        pts = [(cx + x * scale, cy + y * scale) for x, y in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=width, joint="curve")
            r = width * 0.45
            for px, py in (pts[0], pts[-1]):
                draw.ellipse((px - r, py - r, px + r, py + r), fill=0)
    return np.asarray(img, dtype=np.float32) / 255.0


# =============================================================================
# Pixel-space tablet & scan augmentations
# =============================================================================


def _elastic_deform(
    image: np.ndarray, rng: np.random.Generator, alpha: float, sigma: float
) -> np.ndarray:
    """Simard-style smooth elastic displacement field for handwritten/tablet feel."""
    h, w = image.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32), sigma) * alpha
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    coords = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    return ndimage.map_coordinates(image, coords, order=1, mode="constant", cval=1.0).astype(
        np.float32
    )


def _augment_symbol(
    canvas_img: np.ndarray,
    preset: str,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply tablet or score-scan augmentations on the supersampled float32 canvas."""
    if preset == "clean" or strength <= 0:
        return canvas_img
    mode = preset
    if mode == "mixed":
        mode = str(rng.choice(("tablet", "score", "clean"), p=(0.50, 0.40, 0.10)))
    if mode == "clean":
        return canvas_img

    img = canvas_img.copy()
    h, w = img.shape

    # 1. Morphological stroke weight change (thicker or thinner ink)
    if rng.random() < 0.65:
        radius = max(1, round(float(rng.uniform(1.0, 2.2)) * strength))
        if rng.random() < 0.75:  # thicken ink
            img = ndimage.minimum_filter(img, size=2 * radius + 1)
        else:  # thin ink slightly (only if enough ink remains)
            thinned = ndimage.maximum_filter(img, size=3)
            if (thinned < 0.5).sum() >= max(8, (img < 0.5).sum() * 0.25):
                img = thinned

    # 2. Elastic deformation (stronger on tablet preset, milder on score preset)
    elastic_p = 0.85 if mode == "tablet" else 0.40
    if rng.random() < elastic_p:
        alpha = (18.0 if mode == "tablet" else 9.0) * strength * (h / 96.0)
        sigma = (9.0 if mode == "tablet" else 11.0) * (h / 96.0)
        img = _elastic_deform(img, rng, alpha=alpha, sigma=sigma)

    # 3. Occasional background staff line on score crops (so classifiers ignore staff lines)
    if mode == "score" and rng.random() < 0.35:
        line_y = int(rng.integers(int(h * 0.18), int(h * 0.82)))
        line_thick = max(1, round(float(rng.uniform(1.0, 2.5)) * (h / 96.0)))
        line_ink = float(rng.uniform(0.15, 0.55))
        y0 = max(0, line_y - line_thick // 2)
        y1 = min(h, y0 + line_thick)
        img[y0:y1, :] = np.minimum(img[y0:y1, :], line_ink)

    # 4. Uneven stylus pressure / ink fade
    if rng.random() < 0.45:
        small = rng.uniform(0.0, 1.0, (4, 4)).astype(np.float32)
        field = np.asarray(
            Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        fade = float(rng.uniform(0.10, 0.35)) * strength
        img = 1.0 - (1.0 - img) * (1.0 - fade * field)

    # 5. Optical blur
    if rng.random() < 0.55:
        sigma = float(rng.uniform(0.4, 1.3)) * strength * (h / 96.0)
        img = ndimage.gaussian_filter(img, sigma=sigma)

    # 6. Sensor noise or scanner thresholding
    if mode == "score" and rng.random() < 0.45:
        if rng.random() < 0.5:
            thresh = float(rng.uniform(0.48, 0.68))
            img = (img > thresh).astype(np.float32)
        else:
            noise = rng.normal(0.0, 0.03 * strength, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)
    elif mode == "tablet" and rng.random() < 0.25:
        noise = rng.normal(0.0, 0.018 * strength, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return np.clip(img, 0.0, 1.0)


def render_symbol(
    symbol: str | SymbolSpec,
    *,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Render one ``(size, size)`` uint8 grayscale image of ``symbol``.

    Default polarity is black ink on white paper (``0`` = ink, ``255`` = paper),
    matching ``musicset`` page images; ``invert=True`` flips to white ink on
    black background (classic MNIST polarity).
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}: {PRESETS}")
    spec = SYMBOL_BY_NAME[symbol] if isinstance(symbol, str) else symbol
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    canvas = max(96, size * 3)
    lo, hi = spec.scale_range
    jitter = 0.25 if preset == "clean" else min(1.5, max(0.2, strength))
    fraction = float(gen.uniform(lo, hi))

    use_strokes = (
        spec.handwritten
        and preset != "clean"
        and gen.random() < (0.55 if preset == "tablet" else 0.35 if preset == "mixed" else 0.15)
    )
    if use_strokes:
        strokes = _handwritten_strokes(spec.name, gen)
        raw = _render_strokes(strokes, canvas, fraction, gen)
    elif spec.name == "arpeg":
        font = str(gen.choice(available_fonts()))
        segs = int(gen.integers(3, 6))
        arrow = bool(gen.random() < 0.35)
        contours = _arpeggio_contours(font, segs, arrow)
        raw = _rasterize_contours(contours, canvas, fraction, gen, jitter=jitter)
    else:
        candidates = [pair for code in spec.codes for pair in _available_font_codes(code)]
        font, code = candidates[int(gen.integers(len(candidates)))]
        contours_opt = _glyph_Contours(font, code)
        assert contours_opt is not None
        raw = _rasterize_contours(contours_opt, canvas, fraction, gen, jitter=jitter)

    augmented = _augment_symbol(raw, preset=preset, strength=strength, rng=gen)
    pil = Image.fromarray((np.clip(augmented, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    resized = np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    return (255 - resized) if invert else resized


def generate_symbol_samples(
    samples_per_class: int,
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Synthesize ``samples_per_class`` images for each class in ``groups``.

    Returns ``(images, labels, class_names)`` with ``images`` of shape
    ``(N, size, size)`` uint8 and ``labels`` of shape ``(N,)`` int64.
    """
    specs = symbol_specs(groups)
    class_names = tuple(s.name for s in specs)
    total = len(specs) * samples_per_class
    images = np.empty((total, size, size), dtype=np.uint8)
    labels = np.empty(total, dtype=np.int64)
    idx = 0
    for cls_idx, spec in enumerate(specs):
        for k in range(samples_per_class):
            rng = np.random.default_rng([seed, cls_idx, k])
            images[idx] = render_symbol(
                spec, size=size, preset=preset, strength=strength, invert=invert, rng=rng
            )
            labels[idx] = cls_idx
            idx += 1
    return images, labels, class_names


# =============================================================================
# Score page crop extraction
# =============================================================================


def _in_key_sig(page: PageAnnotation, element_idx: int | None) -> bool:
    return any(
        "keySig" in el.classes or "keyAccid" in el.classes for el in page.ancestors(element_idx)
    )


def _crop_box_to_patch(
    image: np.ndarray,
    bbox: BBox | Sequence[float],
    size: int,
    *,
    margin: float = 0.18,
    min_span_px: float = 14.0,
    invert: bool = False,
) -> np.ndarray | None:
    """Crop ``bbox`` (pixels) from a greyscale ``image``, pad to square, resize to ``size``."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x1, x2 = max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))
    y1, y2 = max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2.0 or bh < 2.0:
        return None

    span = max(min_span_px, max(bw, bh) * (1.0 + 2.0 * margin))
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half = 0.5 * span
    ix0, iy0 = math.floor(cx - half), math.floor(cy - half)
    ix1, iy1 = math.ceil(cx + half), math.ceil(cy + half)
    side = max(1, ix1 - ix0, iy1 - iy0)
    ix1, iy1 = ix0 + side, iy0 + side

    # Fill out-of-bounds margins with the local paper tone (median of border pixels)
    sx0, sy0 = max(0, ix0), max(0, iy0)
    sx1, sy1 = min(w, ix1), min(h, iy1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    sub = image[sy0:sy1, sx0:sx1]
    bg = int(np.percentile(sub, 85)) if sub.size else 255
    square = np.full((side, side), bg, dtype=np.uint8)
    square[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = sub

    patch = np.asarray(
        Image.fromarray(square, mode="L").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    return (255 - patch) if invert else patch


def page_symbol_boxes(
    page: PageAnnotation,
    image_size: tuple[int, int],
    *,
    allowed: set[str] | None = None,
) -> list[tuple[str, BBox]]:
    """``(class_name, pixel_bbox)`` pairs of every target symbol on a canonical page."""
    width, height = image_size
    sx, sy = width / page.width, height / page.height
    out: list[tuple[str, BBox]] = []

    def to_px(b: BBox) -> BBox | None:
        x1, y1 = max(0.0, min(float(width), b[0] * sx)), max(0.0, min(float(height), b[1] * sy))
        x2, y2 = max(0.0, min(float(width), b[2] * sx)), max(0.0, min(float(height), b[3] * sy))
        return (x1, y1, x2, y2) if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0 else None

    # Elements: fingerings 1-5 and structural ornaments/pedal
    # Track elements covered by element boxes so we don't double-emit their glyph
    covered_elements: set[int] = set()
    for idx, el in enumerate(page.elements):
        if el.bbox is None or el.synthetic or not el.classes:
            continue
        cls: str | None = None
        if "fing" in el.classes and el.text is not None:
            digit = el.text.strip()
            if digit in ("1", "2", "3", "4", "5"):
                cls = f"fing{digit}"
        elif el.classes[0] in _ELEMENT_SYMBOLS:
            cls = el.classes[0]
        if cls is not None and (allowed is None or cls in allowed):
            px_box = to_px(el.bbox)
            if px_box is not None:
                out.append((cls, px_box))
                covered_elements.add(idx)

    # Glyphs: articulations, accidentals (outside key signatures), dynamics, and
    # any ornament/pedal glyph not already emitted via its enclosing element
    for glyph in page.glyphs:
        if glyph.bbox is None or glyph.element in covered_elements:
            continue
        cls = _GLYPH_TO_SYMBOL.get(glyph.code)
        if cls is None or (allowed is not None and cls not in allowed):
            continue
        if cls.startswith("accidental") and _in_key_sig(page, glyph.element):
            continue
        px_box = to_px(glyph.bbox)
        if px_box is not None:
            out.append((cls, px_box))

    return out


def extract_page_symbols(
    page: PageAnnotation,
    image: Image.Image | np.ndarray | Path,
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    invert: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Extract ``(class_name, patch)`` crops from one rendered page."""
    if isinstance(image, Path):
        with Image.open(image) as im:
            grey = np.asarray(im.convert("L"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        grey = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        grey = arr.mean(axis=2).round().astype(np.uint8) if arr.ndim == 3 else arr.astype(np.uint8)

    h, w = grey.shape[:2]
    allowed = {s.name for s in symbol_specs(groups)}
    crops: list[tuple[str, np.ndarray]] = []
    for cls, bbox in page_symbol_boxes(page, (w, h), allowed=allowed):
        patch = _crop_box_to_patch(grey, bbox, size, invert=invert)
        if patch is not None:
            crops.append((cls, patch))
    return crops


def _fingering_id_map(pages_dir: Path) -> dict[str, str]:
    """Element id -> ``fing1``..``fing5`` from canonical ``pages/*.json.gz``."""
    mapping: dict[str, str] = {}
    if not pages_dir.is_dir():
        return mapping
    for path in sorted(pages_dir.glob(f"*{PAGE_SUFFIX}")):
        page = PageAnnotation.load(path)
        for el in page.elements:
            if "fing" in el.classes and el.id and el.text:
                digit = el.text.strip()
                if digit in ("1", "2", "3", "4", "5"):
                    mapping[el.id] = f"fing{digit}"
    return mapping


def extract_dataset_symbols(
    dataset: Path | Sequence[Path],
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    invert: bool = False,
    max_per_class: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract symbol crops across one or more built ``musicset`` dataset directories.

    Reads canonical ``pages/*.json.gz`` when present, falling back to
    ``annotations/*.json`` (for datasets produced by ``musicset augment``).
    """
    roots = [dataset] if isinstance(dataset, Path) else list(dataset)
    specs = symbol_specs(groups)
    class_names = tuple(s.name for s in specs)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    allowed = set(class_to_idx)

    by_class: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}
    for root in roots:
        pages_dir = root / "pages"
        imgs_dir = root / "imgs"
        ann_dir = root / "annotations"
        page_files = sorted(pages_dir.glob(f"*{PAGE_SUFFIX}")) if pages_dir.is_dir() else []
        if page_files:
            for page_path in page_files:
                stem = page_file_stem(page_path)
                img_path = imgs_dir / f"{stem}.png"
                if not img_path.exists():
                    continue
                page = PageAnnotation.load(page_path)
                for cls, patch in extract_page_symbols(
                    page, img_path, groups=groups, size=size, invert=invert
                ):
                    by_class[class_to_idx[cls]].append(patch)
        elif ann_dir.is_dir():
            fing_ids = _fingering_id_map(pages_dir)
            for ann_path in sorted(ann_dir.glob("*.json")):
                img_path = imgs_dir / f"{ann_path.stem}.png"
                if not img_path.exists():
                    continue
                record = json.loads(ann_path.read_text(encoding="utf-8"))
                with Image.open(img_path) as im:
                    grey = np.asarray(im.convert("L"), dtype=np.uint8)
                for ann in record.get("annotations", []):
                    cls = ann.get("class", "")
                    if cls == "fing":
                        text = str(ann.get("text", "")).strip()
                        if text in ("1", "2", "3", "4", "5"):
                            cls = f"fing{text}"
                        elif ann.get("id") in fing_ids:
                            cls = fing_ids[ann["id"]]
                    if cls not in allowed:
                        continue
                    patch = _crop_box_to_patch(grey, ann["bbox"], size, invert=invert)
                    if patch is not None:
                        by_class[class_to_idx[cls]].append(patch)

    rng = np.random.default_rng(seed)
    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for cls_idx, patches in by_class.items():
        if not patches:
            continue
        if max_per_class > 0 and len(patches) > max_per_class:
            order = rng.permutation(len(patches))[:max_per_class]
            patches = [patches[int(i)] for i in order]
        images_list.extend(patches)
        labels_list.extend([cls_idx] * len(patches))

    if not images_list:
        return (
            np.empty((0, size, size), dtype=np.uint8),
            np.empty(0, dtype=np.int64),
            class_names,
        )
    return (
        np.stack(images_list, axis=0),
        np.array(labels_list, dtype=np.int64),
        class_names,
    )


# =============================================================================
# Dataset assembly, splits, saving and loading
# =============================================================================


@dataclass(frozen=True, slots=True)
class SymbolDatasetReport:
    out: Path
    classes: tuple[str, ...]
    counts: dict[str, int]
    splits: dict[str, int]


def _stratified_split(
    images: np.ndarray,
    labels: np.ndarray,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split ``(images, labels)`` into ``train``, ``val``, ``test`` per class."""
    r_train, r_val, r_test = ratios
    total_r = r_train + r_val + r_test
    if total_r <= 0 or min(ratios) < 0:
        raise ValueError(f"invalid split ratios {ratios}")
    r_train, r_val, r_test = r_train / total_r, r_val / total_r, r_test / total_r

    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for cls_idx in np.unique(labels):
        indices = np.flatnonzero(labels == cls_idx)
        rng.shuffle(indices)
        n = len(indices)
        n_test = round(n * r_test) if r_test > 0 else 0
        n_val = round(n * r_val) if r_val > 0 else 0
        if n >= 3 and r_val > 0 and r_test > 0:
            n_test = max(1, min(n - 2, n_test))
            n_val = max(1, min(n - 1 - n_test, n_val))
        n_train = max(0, n - n_val - n_test)
        buckets["train"].extend(indices[:n_train].tolist())
        buckets["val"].extend(indices[n_train : n_train + n_val].tolist())
        buckets["test"].extend(indices[n_train + n_val :].tolist())

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, idxs in buckets.items():
        arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(arr)
        out[split] = (images[arr], labels[arr])
    return out


def build_symbol_dataset(
    out: Path,
    *,
    samples_per_class: int = 200,
    from_dataset: Path | Sequence[Path] | None = None,
    max_crops_per_class: int = 0,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    write_png: bool = True,
    seed: int = 0,
) -> SymbolDatasetReport:
    """Build a complete MNIST-like symbol dataset into ``out``.

    Writes:
    * ``out/dataset.npz`` (``x_train``, ``y_train``, ``x_val``, ``y_val``,
      ``x_test``, ``y_test``, ``classes``);
    * ``out/labels.json`` (class list, index map, group map, per-class and
      per-split counts);
    * ``out/<split>/<class>/<index>.png`` when ``write_png=True``.
    """
    if samples_per_class < 0:
        raise ValueError("samples_per_class must be >= 0")
    if samples_per_class == 0 and not from_dataset:
        raise ValueError("need samples_per_class > 0 or from_dataset")

    specs = symbol_specs(groups)
    class_names = tuple(s.name for s in specs)
    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []

    if samples_per_class > 0:
        sx, sy, _ = generate_symbol_samples(
            samples_per_class,
            groups=groups,
            size=size,
            preset=preset,
            strength=strength,
            invert=invert,
            seed=seed,
        )
        parts_x.append(sx)
        parts_y.append(sy)

    if from_dataset:
        cx, cy, _ = extract_dataset_symbols(
            from_dataset,
            groups=groups,
            size=size,
            invert=invert,
            max_per_class=max_crops_per_class,
            seed=seed,
        )
        if len(cx):
            parts_x.append(cx)
            parts_y.append(cy)

    images = (
        np.concatenate(parts_x, axis=0) if parts_x else np.empty((0, size, size), dtype=np.uint8)
    )
    labels = np.concatenate(parts_y, axis=0) if parts_y else np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    splits = _stratified_split(images, labels, split, rng)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz",
        x_train=splits["train"][0],
        y_train=splits["train"][1],
        x_val=splits["val"][0],
        y_val=splits["val"][1],
        x_test=splits["test"][0],
        y_test=splits["test"][1],
        classes=np.array(class_names),
    )

    if write_png:
        for split_name, (xs, ys) in splits.items():
            if not len(xs):
                continue
            split_dir = out / split_name
            for cls_name in class_names:
                (split_dir / cls_name).mkdir(parents=True, exist_ok=True)
            counters: Counter[int] = Counter()
            for img, cls_idx in zip(xs, ys, strict=True):
                c = int(cls_idx)
                k = counters[c]
                counters[c] = k + 1
                Image.fromarray(img, mode="L").save(
                    split_dir / class_names[c] / f"{k:06d}.png", compress_level=1
                )

    counts = {name: int((labels == i).sum()) for i, name in enumerate(class_names)}
    split_counts = {k: len(v[1]) for k, v in splits.items()}
    by_group: dict[str, list[str]] = {}
    for s in specs:
        by_group.setdefault(s.group, []).append(s.name)

    meta: dict[str, Any] = {
        "classes": list(class_names),
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "groups": by_group,
        "size": size,
        "invert": invert,
        "preset": preset,
        "counts": counts,
        "splits": split_counts,
    }
    (out / "labels.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return SymbolDatasetReport(out=out, classes=class_names, counts=counts, splits=split_counts)


def load_symbol_dataset(
    path: Path, split: str = "train"
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load ``(images, labels, classes)`` for ``split`` (``train``, ``val``, ``test``)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}: ('train', 'val', 'test')")
    npz_path = path / "dataset.npz" if path.is_dir() else path
    with np.load(npz_path) as data:
        return (
            data[f"x_{split}"],
            data[f"y_{split}"],
            tuple(str(c) for c in data["classes"].tolist()),
        )


__all__ = [
    "PRESETS",
    "SYMBOLS",
    "SYMBOL_BY_NAME",
    "SYMBOL_GROUPS",
    "SymbolDatasetReport",
    "SymbolSpec",
    "build_symbol_dataset",
    "extract_dataset_symbols",
    "extract_page_symbols",
    "generate_symbol_samples",
    "load_symbol_dataset",
    "page_symbol_boxes",
    "render_symbol",
    "symbol_specs",
]
