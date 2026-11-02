"""MNIST-like isolated symbol dataset for fingerings (1-5), ornaments,
articulations, accidentals, dynamics and pedal marks.

Two sources feed the dataset, alone or combined:

* **Synthetic symbols** (:func:`generate_symbol_samples`): SMuFL glyphs from
  every music font Verovio ships (Bravura, Leipzig, Gootville, Leland,
  Petaluma) plus stylus-drawn stroke templates for fingerings 1-5 and common
  score markings, varied with affine jitter, elastic stroke deformation,
  thickness/pressure modulation and scan/tablet noise.
* **HOMUS pen strokes** (:mod:`musicset.homus`): 15,200 symbols drawn on a
  tablet by 100 writers, rasterised like the stroke templates; writers never
  straddle splits.
* **Score crops** (:func:`extract_dataset_symbols`): bounding-box crops pulled
  off rendered pages of a built ``musicset`` dataset (``pages/`` + ``imgs/``,
  or ``annotations/`` + ``imgs/``), with ``fing`` elements split into
  ``fing1``..``fing5`` by their engraved text.

Outputs both a single compressed ``dataset.npz`` archive (``x_train``,
``y_train``, ``x_val``, ``y_val``, ``x_test``, ``y_test``, ``classes``) and an
``ImageFolder`` directory tree (``train/<class>/*.png``, ``val/``, ``test/``)
with ``labels.json``.

Every sample carries a source code (``s_train``, ``s_val``, ``s_test``:
``SOURCE_FONT`` glyph render, ``SOURCE_STROKE`` stylus template, ``SOURCE_CROP``
score crop). Score crops are split by score so one score never straddles
splits, patches with too little ink are dropped, and a byte-identical image
is kept in one split only, since identical engravings recur across scores.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from svgpathtools import parse_path

from musicset.annotate.geometry import BBox, union
from musicset.annotate.schema import Element, Glyph, PageAnnotation
from musicset.annotate.smufl import available_fonts, verovio_data_dir
from musicset.context import (
    CONTEXT_DIM,
    CONTEXT_FEATURES,
    PageGeometry,
    SymbolContext,
    context_vector,
)
from musicset.emnist import (
    EMNIST_SIGNS,
    EMNIST_SYMBOLS,
    compose_sign,
    compose_word,
    load_emnist,
)
from musicset.fornes import fornes_writer_ids, load_fornes
from musicset.homus import (
    Stroke,
    StrokeSample,
    compose_homus_codas,
    load_homus,
    stroke_fraction,
)
from musicset.parallel import map_pool
from musicset.pipeline import PAGE_SUFFIX, page_file_stem
from musicset.strokes import load_stroke_contributions

SYMBOL_GROUPS = (
    "fingering",
    "ornament",
    "articulation",
    "accidental",
    "dynamic",
    "pedal",
    "note",
    "rest",
    "clef",
    "barline",
    "navigation",
)
PRESETS = ("mixed", "tablet", "score", "clean")

EDITOR_PEN_WIDTH = 0.05
"""Pen width, as a fraction of the canvas, that val/test pen samples are rendered with:
the width the score editor sends (``granita.DEFAULT_PEN_WIDTH``; it scored best on
held-out writers). Training renders span 0.05-0.13."""

SOURCE_FONT, SOURCE_STROKE, SOURCE_CROP, SOURCE_HAND, SOURCE_EMNIST, SOURCE_FORNES = range(6)
SOURCE_NAMES = ("font", "stroke", "crop", "hand", "emnist", "fornes")
"""Per-sample provenance, stored as ``s_<split>`` in ``dataset.npz``: SMuFL glyph
render, stylus stroke template, score crop, real pen strokes (HOMUS or an editor),
real handwritten digits and letters (EMNIST) for fingerings and dynamics, scanned
handwritten clefs and accidentals (Fornés)."""


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
    compose: tuple[str, str] | None = None
    """Time signature drawn from stacked SMuFL digits, ``(numerator, denominator)``."""
    text: tuple[str, ...] = ()
    """Strings that also render the class in Verovio's text font (``D.S.``, ``Fine``)."""


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
    # Notes: precomposed SMuFL notes, stem up and stem down merged (pitch is not the
    # classifier's job: it comes from where the symbol sits on the staff)
    SymbolSpec("noteWhole", "note", ("E1D2", "E0A2"), scale_range=(0.26, 0.46)),
    SymbolSpec("noteHalf", "note", ("E1D3", "E1D4"), scale_range=(0.62, 0.88)),
    SymbolSpec("noteQuarter", "note", ("E1D5", "E1D6"), scale_range=(0.62, 0.88)),
    SymbolSpec("note8th", "note", ("E1D7", "E1D8"), scale_range=(0.62, 0.88)),
    SymbolSpec("note16th", "note", ("E1D9", "E1DA"), scale_range=(0.62, 0.88)),
    SymbolSpec("note32nd", "note", ("E1DB", "E1DC"), scale_range=(0.64, 0.90)),
    SymbolSpec("note64th", "note", ("E1DD", "E1DE"), scale_range=(0.66, 0.90)),
    SymbolSpec("augmentationDot", "note", ("E1E7",), scale_range=(0.10, 0.24)),
    # Rests: whole and half rests share one shape and differ only by staff position
    SymbolSpec("restWholeHalf", "rest", ("E4E3", "E4E4"), scale_range=(0.24, 0.44)),
    SymbolSpec("restQuarter", "rest", ("E4E5",), scale_range=(0.52, 0.80)),
    SymbolSpec("rest8th", "rest", ("E4E6",), scale_range=(0.42, 0.68)),
    SymbolSpec("rest16th", "rest", ("E4E7",), scale_range=(0.50, 0.76)),
    SymbolSpec("rest32nd", "rest", ("E4E8",), scale_range=(0.56, 0.82)),
    SymbolSpec("rest64th", "rest", ("E4E9",), scale_range=(0.62, 0.88)),
    # Clefs
    SymbolSpec("gClef", "clef", ("E050",), scale_range=(0.70, 0.90)),
    SymbolSpec("fClef", "clef", ("E062",), scale_range=(0.52, 0.78)),
    SymbolSpec("cClef", "clef", ("E05C",), scale_range=(0.58, 0.84)),
    # Barline: drawn as a bar, Verovio engraves it as a line rather than a glyph
    SymbolSpec("barlineSingle", "barline", (), scale_range=(0.58, 0.86)),
    # Navigation marks: D.S. and D.C. as SMuFL glyphs or set in Verovio's text font,
    # Fine in the text font only, the segno and coda signs
    SymbolSpec(
        "dalSegno", "navigation", ("E045",), scale_range=(0.62, 0.88), text=("D.S.", "D. S.")
    ),
    SymbolSpec("daCapo", "navigation", ("E046",), scale_range=(0.62, 0.88), text=("D.C.", "D. C.")),
    SymbolSpec("fine", "navigation", (), scale_range=(0.58, 0.86), text=("Fine",)),
    SymbolSpec("segno", "navigation", ("E047",), scale_range=(0.62, 0.88)),
    SymbolSpec("coda", "navigation", ("E048",), scale_range=(0.60, 0.86)),
)

SYMBOL_BY_NAME: dict[str, SymbolSpec] = {s.name: s for s in SYMBOLS}


@dataclass(frozen=True, slots=True)
class SymbolProfile:
    """The class set one consumer of the dataset needs."""

    name: str
    classes: tuple[str, ...]
    """In catalog order."""
    handwriting: bool
    """Whether stylus stroke templates, HOMUS and EMNIST belong in its builds."""


BATTUTA_DROPPED: dict[str, str] = {
    "accidentalDoubleSharp": "too rare to be worth a pen gesture",
    "accidentalDoubleFlat": "too rare to be worth a pen gesture",
    "dynamicMezzo": "only written inside mp / mf",
    "dynamicZ": "only written inside sfz / fz",
    "pedal": "not drawn by hand in the editor",
    "note32nd": "too short for pen input",
    "note64th": "too short for pen input",
    "restWholeHalf": "the editor infers a rest's duration from the measure",
    "rest8th": "the editor infers a rest's duration from the measure",
    "rest16th": "the editor infers a rest's duration from the measure",
    "rest32nd": "the editor infers a rest's duration from the measure",
    "rest64th": "the editor infers a rest's duration from the measure",
    "barlineSingle": "the editor draws barlines itself",
}
"""Catalog classes the Battuta editor's stylus model leaves out, and why. ``restQuarter``
stays as the one rest gesture."""

SYMBOL_PROFILES: dict[str, SymbolProfile] = {
    "kakigori": SymbolProfile("kakigori", tuple(s.name for s in SYMBOLS), handwriting=False),
    "battuta": SymbolProfile(
        "battuta",
        tuple(s.name for s in SYMBOLS if s.name not in BATTUTA_DROPPED),
        handwriting=True,
    ),
}
"""``kakigori``: every class, engraved only (fonts and score crops), for resolving the
OMR detector's merged classes. ``battuta``: the stylus set of the Battuta editor, with
real handwriting (HOMUS, EMNIST, contributions)."""

# Element classes in PageAnnotation that map directly to a symbol class
_ELEMENT_SYMBOLS = frozenset(("trill", "mordent", "turn", "fermata", "arpeg", "pedal"))

# Stacked time signatures by (numerator, denominator)
_TIMESIG_BY_COMPOSE: dict[tuple[str, str], str] = {
    spec.compose: spec.name for spec in SYMBOLS if spec.compose is not None
}

# Standalone note durations (MEI @dur -> symbol class)
_NOTE_BY_DUR: dict[str, str] = {
    "1": "noteWhole",
    "2": "noteHalf",
    "4": "noteQuarter",
    "8": "note8th",
    "16": "note16th",
    "32": "note32nd",
    "64": "note64th",
}
_NOTE_BODY_CLASSES = frozenset(("notehead", "stem", "flag"))
_SKIP_NOTE_ANCESTORS = frozenset(("beam", "chord", "graceGrp"))
_DIGIT_CODES = frozenset(f"E08{d}" for d in range(10))

# SMuFL codepoint -> symbol class for page glyph extraction
_GLYPH_TO_SYMBOL: dict[str, str] = {
    code: spec.name
    for spec in SYMBOLS
    if spec.group
    in (
        "articulation",
        "accidental",
        "dynamic",
        "ornament",
        "pedal",
        "note",
        "rest",
        "clef",
        "timesig",
        "navigation",
    )
    for code in spec.codes
}


def symbol_specs(
    groups: Sequence[str] = SYMBOL_GROUPS, classes: Sequence[str] | None = None
) -> tuple[SymbolSpec, ...]:
    """Symbol specs belonging to ``groups`` (and to ``classes`` when given) in canonical
    order."""
    unknown = set(groups) - set(SYMBOL_GROUPS)
    if unknown:
        raise ValueError(f"unknown symbol group(s) {sorted(unknown)}: {SYMBOL_GROUPS}")
    if classes is not None:
        bad = set(classes) - set(SYMBOL_BY_NAME)
        if bad:
            raise ValueError(f"unknown symbol class(es) {sorted(bad)}")
    wanted = set(groups)
    keep = None if classes is None else set(classes)
    return tuple(s for s in SYMBOLS if s.group in wanted and (keep is None or s.name in keep))


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


def _timesig_contours(font: str, top: str, bottom: str) -> list[tuple[np.ndarray, int]]:
    """Time signature as SMuFL digits (``E080``..``E089``) laid out in two stacked rows."""

    def row(digits: str) -> list[tuple[np.ndarray, int]]:
        pieces: list[tuple[np.ndarray, int]] = []
        x = 0.0
        for ch in digits:
            code = f"E08{int(ch)}"
            base = _glyph_Contours(font, code) or _glyph_Contours("Bravura", code)
            assert base is not None
            xs = np.vstack([pts for pts, _ in base])[:, 0]
            pieces.extend((pts - [float(xs.min()) - x, 0.0], sign) for pts, sign in base)
            x += float(xs.max() - xs.min()) * 1.08
        return pieces

    def bounds(contours: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
        pts = np.vstack([p for p, _ in contours])
        return pts.min(axis=0), pts.max(axis=0)

    top_c, bottom_c = row(top), row(bottom)
    (tx0, ty0), (tx1, ty1) = bounds(top_c)
    (bx0, by0), (bx1, _) = bounds(bottom_c)
    gap = 0.06 * float(ty1 - ty0)
    # y grows downwards: the top row ends just above 0, the bottom row starts just below
    out = [(pts - [0.5 * (tx0 + tx1), float(ty1) + gap / 2], sign) for pts, sign in top_c]
    out += [(pts - [0.5 * (bx0 + bx1), float(by0) - gap / 2], sign) for pts, sign in bottom_c]
    return out


def _barline_contours() -> list[tuple[np.ndarray, int]]:
    """A single barline: a bar 8% as wide as it is tall."""
    pts = np.array([[0.0, 0.0], [80.0, 0.0], [80.0, 1000.0], [0.0, 1000.0]], dtype=np.float64)
    return [(pts, 1)]


@lru_cache(maxsize=1)
def _text_faces() -> tuple[bytes, ...]:
    """Verovio's text font, Liberation (regular, bold, italic, bold italic), decoded from
    the WOFF2 data its ``Liberation.css`` embeds: the face page crops of ``D.C.`` or
    ``Fine`` were set in."""
    css = (verovio_data_dir() / "Liberation.css").read_text()
    faces = tuple(base64.b64decode(m) for m in re.findall(r"base64,([A-Za-z0-9+/=]+)", css))
    if not faces:
        raise FileNotFoundError(f"no embedded font in {verovio_data_dir() / 'Liberation.css'}")
    return faces


@lru_cache(maxsize=8)
def _text_font(face: int, size: int = 96) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(io.BytesIO(_text_faces()[face]), size)


def _rasterize_text(
    text: str, canvas: int, target_fraction: float, rng: np.random.Generator, *, jitter: float
) -> np.ndarray:
    """``text`` in a random Liberation face on a ``(canvas, canvas)`` float32 image
    (1.0=paper, 0.0=ink), its longer side spanning ``target_fraction``, with the same
    rotation, scale and shear jitter as :func:`_rasterize_contours`."""
    font = _text_font(int(rng.integers(len(_text_faces()))))
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("L", (right - left + 24, bottom - top + 24), 255)
    ImageDraw.Draw(img).text((12 - left, 12 - top), text, font=font, fill=0)
    if jitter > 0:
        angle = math.degrees(float(rng.uniform(-0.18, 0.18) * jitter))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)
        shear = float(rng.uniform(-0.14, 0.14) * jitter)
        w, h = img.size
        extra = round(abs(shear) * h)
        img = img.transform(
            (w + extra, h),
            Image.Transform.AFFINE,
            (1.0, shear, -extra if shear > 0 else 0.0, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=255,
        )
    ink = np.asarray(img, dtype=np.uint8)
    rows = np.flatnonzero((ink < 128).any(axis=1))
    cols = np.flatnonzero((ink < 128).any(axis=0))
    if not len(rows):
        return np.ones((canvas, canvas), dtype=np.float32)
    crop = img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
    sx = sy = 1.0
    if jitter > 0:
        sx = float(rng.uniform(1.0 - 0.14 * jitter, 1.0 + 0.14 * jitter))
        sy = float(rng.uniform(1.0 - 0.14 * jitter, 1.0 + 0.14 * jitter))
    scale = canvas * target_fraction / max(crop.width * sx, crop.height * sy)
    tw = max(1, min(canvas, round(crop.width * sx * scale)))
    th = max(1, min(canvas, round(crop.height * sy * scale)))
    glyph = crop.resize((tw, th), Image.Resampling.BILINEAR)
    page = Image.new("L", (canvas, canvas), 255)
    shift = canvas * 0.04 * jitter
    x = round((canvas - tw) / 2 + float(rng.uniform(-shift, shift)))
    y = round((canvas - th) / 2 + float(rng.uniform(-shift, shift)))
    page.paste(glyph, (max(0, min(canvas - tw, x)), max(0, min(canvas - th, y))))
    return np.asarray(page, dtype=np.float32) / 255.0


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


def _segment_widths(pts: Sequence[tuple[float, float]], width: int) -> list[int]:
    """Pressure-like pen widths per segment: wide where the pen moves slowly, narrow where
    fast. Mirrors ``granita.infer.predictor._segment_widths``; keep the two in sync."""
    lengths = [math.dist(a, b) for a, b in pairwise(pts)]
    ref = sorted(lengths)[len(lengths) // 2] if lengths else 0.0
    if ref <= 1e-9:
        return [width] * len(lengths)
    return [max(1, round(width * min(1.3, max(0.55, 1.3 - 0.5 * (ln / ref))))) for ln in lengths]


def _render_strokes(
    strokes: Sequence[Sequence[tuple[float, float]]],
    canvas: int,
    target_fraction: float,
    rng: np.random.Generator,
    *,
    width_frac: float | None = None,
    jitter: bool = True,
    brush: bool = False,
) -> np.ndarray:
    """Render stylus strokes (normalised to ``[-1, 1]``) onto a ``(canvas, canvas)``
    float32 canvas (1.0=paper, 0.0=ink).

    The pen is ``width_frac`` canvas widths wide (random 0.055-0.115 when
    unset) with round caps; ``jitter`` shifts the centre by up to 6%. ``brush``
    imitates a pressure brush: each segment's width follows the pen speed and
    the drawing gets a soft edge, exactly like ``granita.render_strokes(brush=True)``.
    """
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    scale = canvas * 0.5 * target_fraction
    shift = canvas * 0.06
    cx = canvas * 0.5 + (float(rng.uniform(-shift, shift)) if jitter else 0.0)
    cy = canvas * 0.5 + (float(rng.uniform(-shift, shift)) if jitter else 0.0)
    frac = float(rng.uniform(0.055, 0.115)) if width_frac is None else width_frac
    width = max(2, round(canvas * frac))
    for stroke in strokes:
        pts = [(cx + x * scale, cy + y * scale) for x, y in stroke]
        if not pts:
            continue
        if brush and len(pts) >= 2:
            widths = _segment_widths(pts, width)
            for (a, b), w in zip(pairwise(pts), widths, strict=True):
                draw.line([a, b], fill=0, width=w)
                rr = w * 0.5
                draw.ellipse((b[0] - rr, b[1] - rr, b[0] + rr, b[1] + rr), fill=0)
            r = widths[0] * 0.5
        else:
            if len(pts) >= 2:
                draw.line(pts, fill=0, width=width, joint="curve")
            r = width * 0.45
        for px, py in (pts[0], pts[-1]):
            draw.ellipse((px - r, py - r, px + r, py + r), fill=0)
    if brush:
        img = img.filter(ImageFilter.GaussianBlur(radius=canvas * 0.012))
    return np.asarray(img, dtype=np.float32) / 255.0


def normalize_strokes(
    strokes: Sequence[Sequence[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Centre raw pen strokes (any units) and scale them, aspect kept, into ``[-1, 1]``."""
    points = [(float(x), float(y)) for stroke in strokes for x, y in stroke]
    if not points:
        raise ValueError("no stroke points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    scale = 1.0 / half if half > 1e-9 else 0.0
    return [
        [((float(x) - cx) * scale, (float(y) - cy) * scale) for x, y in stroke]
        for stroke in strokes
        if len(stroke)
    ]


def render_strokes_sample(
    strokes: Sequence[Sequence[tuple[float, float]]],
    *,
    size: int = 32,
    fraction: float = 0.7,
    width: float = EDITOR_PEN_WIDTH,
    preset: str = "clean",
    strength: float = 1.0,
    invert: bool = False,
    brush: bool = False,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Rasterise raw pen strokes (an editor's or HOMUS's) like a stylus template.

    The strokes are centred with their longer side spanning ``fraction`` of the
    canvas and drawn with a pen ``width`` canvas widths wide (``EDITOR_PEN_WIDTH``
    is what the editor sends), or a pressure-like ``brush``.
    ``preset="clean"`` is deterministic, the other presets add the tablet/scan
    augmentations and centre jitter.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}: {PRESETS}")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    canvas = max(96, size * 3)
    raw = _render_strokes(
        normalize_strokes(strokes),
        canvas,
        fraction,
        gen,
        width_frac=width,
        jitter=preset != "clean",
        brush=brush,
    )
    augmented = _augment_symbol(raw, preset=preset, strength=strength, rng=gen)
    pil = Image.fromarray((np.clip(augmented, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    resized = np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    return (255 - resized) if invert else resized


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


def render_symbol_with_source(
    symbol: str | SymbolSpec,
    *,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    rng: np.random.Generator | int | None = None,
    stroke_templates: bool = True,
) -> tuple[np.ndarray, int]:
    """:func:`render_symbol` plus the sample's source: ``SOURCE_STROKE`` when a
    stylus stroke template was drawn, else ``SOURCE_FONT``. ``stroke_templates=False``
    never draws one (an engraved-only build).
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
        stroke_templates
        and spec.handwritten
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
    elif spec.compose is not None:
        font = str(gen.choice(available_fonts()))
        contours = _timesig_contours(font, *spec.compose)
        raw = _rasterize_contours(contours, canvas, fraction, gen, jitter=jitter)
    elif spec.name == "barlineSingle":
        raw = _rasterize_contours(_barline_contours(), canvas, fraction, gen, jitter=jitter)
    elif spec.text and (not spec.codes or gen.random() < 0.6):
        text = spec.text[int(gen.integers(len(spec.text)))]
        raw = _rasterize_text(text, canvas, fraction, gen, jitter=jitter)
    else:
        candidates = [pair for code in spec.codes for pair in _available_font_codes(code)]
        font, code = candidates[int(gen.integers(len(candidates)))]
        contours_opt = _glyph_Contours(font, code)
        assert contours_opt is not None
        raw = _rasterize_contours(contours_opt, canvas, fraction, gen, jitter=jitter)

    augmented = _augment_symbol(raw, preset=preset, strength=strength, rng=gen)
    pil = Image.fromarray((np.clip(augmented, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    resized = np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    image = (255 - resized) if invert else resized
    return image, (SOURCE_STROKE if use_strokes else SOURCE_FONT)


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
    image, _ = render_symbol_with_source(
        symbol, size=size, preset=preset, strength=strength, invert=invert, rng=rng
    )
    return image


def ink_stroke_width(ink: np.ndarray) -> float:
    """Mean stroke width, in pixels, of a black-on-white ink image: twice the ink's area
    over its boundary (2 for any stroke one or two pixels wide)."""
    m = ink < 128
    if not m.any():
        return 0.0
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return 2.0 * float(m.sum()) / max(1, int((m & ~inner).sum()))


def render_ink_sample(
    ink: np.ndarray,
    *,
    size: int = 32,
    fraction: float = 0.7,
    preset: str = "clean",
    strength: float = 1.0,
    invert: bool = False,
    rng: np.random.Generator | int | None = None,
    pen: float | None = None,
) -> np.ndarray:
    """Put a black-on-white ink image (a handwritten glyph or word) on the symbol canvas
    like a stylus template: centred, aspect kept, its longer side spanning ``fraction``
    of the canvas; presets other than ``clean`` add the tablet/scan augmentations.
    ``pen`` (a fraction of the canvas, like ``EDITOR_PEN_WIDTH``) thickens strokes
    thinner than that pen to its width, for scans whose fine lines would fade at 32 px."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}: {PRESETS}")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    canvas = max(96, size * 3)
    h, w = ink.shape
    scale = canvas * fraction / max(1, h, w)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    glyph = Image.fromarray(ink, mode="L").resize((tw, th), Image.Resampling.BILINEAR)
    grow = 0 if pen is None else round((pen * canvas - ink_stroke_width(ink) * scale) / 2)
    if grow > 0:
        yy, xx = np.mgrid[-grow : grow + 1, -grow : grow + 1]
        padded = np.pad(np.asarray(glyph), grow, constant_values=255)
        thick = ndimage.grey_erosion(padded, footprint=xx * xx + yy * yy <= grow * grow)
        glyph = Image.fromarray(thick.astype(np.uint8), mode="L")
        tw, th = glyph.size
    page = Image.new("L", (canvas, canvas), 255)
    shift = canvas * 0.06 if preset != "clean" else 0.0
    x = round((canvas - tw) / 2 + (gen.uniform(-shift, shift) if shift else 0.0))
    y = round((canvas - th) / 2 + (gen.uniform(-shift, shift) if shift else 0.0))
    page.paste(glyph, (x, y))
    raw = np.asarray(page, dtype=np.float32) / 255.0
    augmented = _augment_symbol(raw, preset=preset, strength=strength, rng=gen)
    pil = Image.fromarray((np.clip(augmented, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    resized = np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    return (255 - resized) if invert else resized


EmnistTask = tuple[
    tuple[np.ndarray, ...], str, bool, float, float, int, float, bool, int, int, bool
]

_NO_GLYPH = np.zeros((0, 0), dtype=np.uint8)
"""Placeholder glyph for a character EMNIST lacks and :func:`compose_word` draws (``.``)."""


def _pick_emnist(
    pool: dict[str, np.ndarray], options: str, rng: np.random.Generator, where: str
) -> np.ndarray:
    """A random EMNIST image of any character in ``options`` (``.``: the placeholder)."""
    if options == ".":
        return _NO_GLYPH
    avail = [c for c in options if c in pool and len(pool[c])]
    if not avail:
        raise ValueError(f"EMNIST {where} file has none of {options!r}")
    c = avail[int(rng.integers(len(avail)))]
    return pool[c][int(rng.integers(len(pool[c])))]


def _render_emnist_task(task: EmnistTask) -> np.ndarray:
    """One EMNIST sample: the word's glyphs composed (or the coda's circle and cross
    overlaid), then placed like a stylus template (augmented at a jittered size for
    ``train``, clean otherwise)."""
    glyphs, text, overlay, lo, hi, size, strength, invert, seed, j, train = task
    rng = np.random.default_rng([seed, 5, j])
    word = compose_sign(text, glyphs, rng) if overlay else compose_word(glyphs, rng, chars=text)
    if train:
        return render_ink_sample(
            word,
            size=size,
            fraction=float(rng.uniform(lo, hi)),
            preset="tablet",
            strength=strength,
            invert=invert,
            rng=rng,
        )
    return render_ink_sample(
        word, size=size, fraction=0.5 * (lo + hi), preset="clean", invert=invert, rng=rng
    )


def _run_tasks[A, T](
    fn: Callable[[A], T], tasks: Sequence[A], workers: int | None, chunksize: int
) -> Iterator[T]:
    """``fn`` over ``tasks`` in order: inline for ``workers=1`` or small jobs, else pooled."""
    if workers == 1 or len(tasks) < 2 * chunksize:
        for task in tasks:
            yield fn(task)
        return
    for task, outcome in map_pool(fn, tasks, workers=workers, chunksize=chunksize):
        if not outcome.ok or outcome.value is None:
            name = getattr(fn, "__name__", "task")
            raise RuntimeError(f"{name} failed on {task!r}: {outcome.error}")
        yield outcome.value


def _render_task(
    task: tuple[str, int, str, float, bool, int, int, int, bool],
) -> tuple[np.ndarray, int]:
    """One synthetic sample ``(image, source)``; seeded by ``(seed, class, k)``."""
    name, size, preset, strength, invert, seed, cls_idx, k, templates = task
    rng = np.random.default_rng([seed, cls_idx, k])
    return render_symbol_with_source(
        name,
        size=size,
        preset=preset,
        strength=strength,
        invert=invert,
        rng=rng,
        stroke_templates=templates,
    )


def _sample_fraction(sample: StrokeSample) -> float:
    """Canvas fraction of a pen sample: its own (from a staff space) or HOMUS's size rule."""
    return sample.fraction if sample.fraction is not None else stroke_fraction(sample.strokes)


def _render_hand_task(
    task: tuple[tuple[Stroke, ...], float, int, float, bool, int, int, int, bool, float],
) -> np.ndarray:
    """One real pen sample (HOMUS or contributed): augmented at its jittered natural
    size for ``train`` (pen 0.05-0.13 wide, a pressure brush with probability
    ``brush_p``), else deterministic the way an editor would send it."""
    strokes, fraction, size, strength, invert, seed, j, k, train, brush_p = task
    rng = np.random.default_rng([seed, 3, j, k])
    if train:
        return render_strokes_sample(
            strokes,
            size=size,
            fraction=float(np.clip(fraction * rng.uniform(0.85, 1.15), 0.08, 0.9)),
            width=float(rng.uniform(0.05, 0.13)),
            preset="tablet",
            strength=strength,
            invert=invert,
            brush=bool(rng.random() < brush_p),
            rng=rng,
        )
    return render_strokes_sample(
        strokes,
        size=size,
        fraction=fraction,
        width=EDITOR_PEN_WIDTH,
        preset="clean",
        invert=invert,
        rng=rng,
    )


FornesTask = tuple[np.ndarray, float, float, int, float, bool, int, int, int, bool]


def _render_fornes_task(task: FornesTask) -> np.ndarray:
    """One Fornés scan placed like an EMNIST word, its strokes thickened to a pen: for
    ``train`` augmented at a jittered size with a pen 0.05-0.13 wide (HOMUS's), else
    clean with the editor's pen."""
    image, lo, hi, size, strength, invert, seed, j, k, train = task
    rng = np.random.default_rng([seed, SOURCE_FORNES, j, k])
    if train:
        return render_ink_sample(
            image,
            size=size,
            fraction=float(rng.uniform(lo, hi)),
            preset="tablet",
            strength=strength,
            invert=invert,
            rng=rng,
            pen=float(rng.uniform(0.05, 0.13)),
        )
    return render_ink_sample(
        image,
        size=size,
        fraction=0.5 * (lo + hi),
        preset="clean",
        invert=invert,
        rng=rng,
        pen=EDITOR_PEN_WIDTH,
    )


def _generate_symbol_samples(
    samples_per_class: int,
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    seed: int = 0,
    workers: int | None = 1,
    classes: Sequence[str] | None = None,
    stroke_templates: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    specs = symbol_specs(groups, classes)
    class_names = tuple(s.name for s in specs)
    tasks = [
        (spec.name, size, preset, strength, invert, seed, cls_idx, k, stroke_templates)
        for cls_idx, spec in enumerate(specs)
        for k in range(samples_per_class)
    ]
    images = np.empty((len(tasks), size, size), dtype=np.uint8)
    labels = np.array([t[6] for t in tasks], dtype=np.int64)
    sources = np.empty(len(tasks), dtype=np.uint8)
    for idx, (image, source) in enumerate(_run_tasks(_render_task, tasks, workers, 256)):
        images[idx] = image
        sources[idx] = source
    return images, labels, class_names, sources


def generate_symbol_samples(
    samples_per_class: int,
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    seed: int = 0,
    workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Synthesize ``samples_per_class`` images for each class in ``groups``.

    Returns ``(images, labels, class_names)`` with ``images`` of shape
    ``(N, size, size)`` uint8 and ``labels`` of shape ``(N,)`` int64. Each
    sample is seeded by ``(seed, class, index)``, so ``workers`` only changes
    the speed.
    """
    images, labels, class_names, _ = _generate_symbol_samples(
        samples_per_class,
        groups=groups,
        size=size,
        preset=preset,
        strength=strength,
        invert=invert,
        seed=seed,
        workers=workers,
    )
    return images, labels, class_names


def generate_symbol_samples_with_sources(
    samples_per_class: int,
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    seed: int = 0,
    workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """:func:`generate_symbol_samples` plus per-sample source codes (uint8,
    ``SOURCE_FONT`` or ``SOURCE_STROKE``)."""
    return _generate_symbol_samples(
        samples_per_class,
        groups=groups,
        size=size,
        preset=preset,
        strength=strength,
        invert=invert,
        seed=seed,
        workers=workers,
    )


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


def _is_single_dot(bbox: BBox | None) -> bool:
    if bbox is None:
        return False
    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    return w > 0 and h > 0 and max(w, h) / min(w, h) < 1.6


def _metersig_class(el: Element, glyphs: Sequence[Glyph]) -> str | None:
    """Symbol class of a ``meterSig`` element from its SMuFL glyphs."""
    if el.bbox is None or not glyphs:
        return None
    if len(glyphs) == 1:
        return _GLYPH_TO_SYMBOL.get(glyphs[0].code)
    y_mid = 0.5 * (el.bbox[1] + el.bbox[3])
    upper: list[tuple[float, str]] = []
    lower: list[tuple[float, str]] = []
    for g in glyphs:
        if g.code not in _DIGIT_CODES or g.bbox is None:
            return None
        row = upper if 0.5 * (g.bbox[1] + g.bbox[3]) < y_mid else lower
        row.append((g.bbox[0], g.code[-1]))
    top = "".join(d for _, d in sorted(upper))
    bot = "".join(d for _, d in sorted(lower))
    return _TIMESIG_BY_COMPOSE.get((top, bot))


_TEXT_MARKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^d\.?\s*s\.?$", re.IGNORECASE), "dalSegno"),
    (re.compile(r"^d\.?\s*c\.?$", re.IGNORECASE), "daCapo"),
    (re.compile(r"^fine\.?$", re.IGNORECASE), "fine"),
)
"""Directions whose whole text is a navigation mark; ``D.C. al Fine`` and friends are
left out, their box holds more than the mark."""


def _text_mark_class(text: str) -> str | None:
    """Navigation class of a ``dir`` / ``repeatMark`` text, or ``None``."""
    flat = " ".join(text.split())
    for pattern, cls in _TEXT_MARKS:
        if pattern.match(flat):
            return cls
    return None


def _standalone_note_class(
    el: Element, page: PageAnnotation, parts: Sequence[int], codes: set[str]
) -> str | None:
    """Symbol class of an unbeamed, non-chord ``note`` element, or ``None``."""
    if "grace" in el.classes or any(
        _SKIP_NOTE_ANCESTORS.intersection(anc.classes) for anc in page.ancestors(el.parent)
    ):
        return None
    dur = el.attrs.get("dur")
    if dur in _NOTE_BY_DUR:
        return _NOTE_BY_DUR[dur]
    has_stem = any("stem" in page.elements[p].classes for p in parts)
    if "E246" in codes or "E247" in codes:
        return "note64th"
    if "E244" in codes or "E245" in codes:
        return "note32nd"
    if "E242" in codes or "E243" in codes:
        return "note16th"
    if "E240" in codes or "E241" in codes:
        return "note8th"
    if has_stem and "E0A3" in codes:
        return "noteHalf"
    if has_stem and "E0A4" in codes:
        return "noteQuarter"
    if not has_stem and "E0A2" in codes:
        return "noteWhole"
    return None


def page_symbol_boxes(
    page: PageAnnotation,
    image_size: tuple[int, int],
    *,
    allowed: set[str] | None = None,
) -> list[tuple[str, BBox]]:
    """``(class_name, pixel_bbox)`` pairs of every target symbol on a canonical page."""
    return [
        (cls, box)
        for cls, box, _ in page_symbol_boxes_with_context(page, image_size, allowed=allowed)
    ]


def page_symbol_boxes_with_context(
    page: PageAnnotation,
    image_size: tuple[int, int],
    *,
    allowed: set[str] | None = None,
) -> list[tuple[str, BBox, SymbolContext | None]]:
    """``(class_name, pixel_bbox, context)`` of every target symbol on a canonical page;
    the context (:mod:`musicset.context`) is computed from the page-unit box."""
    width, height = image_size
    geometry = PageGeometry(page)
    sx, sy = width / page.width, height / page.height
    out: list[tuple[str, BBox, SymbolContext | None]] = []

    def to_px(b: BBox) -> BBox | None:
        x1, y1 = max(0.0, min(float(width), b[0] * sx)), max(0.0, min(float(height), b[1] * sy))
        x2, y2 = max(0.0, min(float(width), b[2] * sx)), max(0.0, min(float(height), b[3] * sy))
        return (x1, y1, x2, y2) if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0 else None

    # Pre-index note body parts (notehead/stem/flag) and meterSig/note glyphs
    note_parts: dict[int, list[int]] = {}
    part_to_note: dict[int, int] = {}
    metersig_children: dict[int, list[int]] = {}
    for idx, el in enumerate(page.elements):
        if not el.classes:
            continue
        cur = el.parent
        while cur is not None:
            anc = page.elements[cur]
            if anc.classes and anc.classes[0] == "note" and el.classes[0] in _NOTE_BODY_CLASSES:
                note_parts.setdefault(cur, []).append(idx)
                part_to_note[idx] = cur
                break
            if anc.classes and anc.classes[0] == "meterSig":
                metersig_children.setdefault(cur, []).append(idx)
                break
            cur = anc.parent

    metersig_glyphs: dict[int, list[Glyph]] = {}
    note_codes: dict[int, set[str]] = {}
    for glyph in page.glyphs:
        if glyph.bbox is None or glyph.element is None:
            continue
        el_idx: int | None = glyph.element
        if el_idx in part_to_note:
            note_codes.setdefault(part_to_note[el_idx], set()).add(glyph.code)
        while el_idx is not None:
            anc = page.elements[el_idx]
            if anc.classes and anc.classes[0] == "meterSig":
                metersig_glyphs.setdefault(el_idx, []).append(glyph)
                break
            if anc.classes and anc.classes[0] == "note" and glyph.element == el_idx:
                note_codes.setdefault(el_idx, set()).add(glyph.code)
                break
            el_idx = anc.parent

    # Elements: fingerings 1-5, ornaments/pedal, barlines, dots, meterSigs, standalone notes
    # Track elements covered by element boxes so we don't double-emit their glyph
    covered_elements: set[int] = set()
    for idx, el in enumerate(page.elements):
        if el.bbox is None or el.synthetic or not el.classes:
            continue
        cls: str | None = None
        box: BBox | None = el.bbox
        cover: list[int] = [idx]
        if "fing" in el.classes and el.text is not None:
            digit = el.text.strip()
            if digit in ("1", "2", "3", "4", "5"):
                cls = f"fing{digit}"
        elif el.classes[0] in _ELEMENT_SYMBOLS:
            cls = el.classes[0]
        elif el.classes[0] == "barLine" and (not el.refined or "barlineSingle" in el.refined):
            cls = "barlineSingle"
        elif el.classes[0] in ("dir", "repeatMark") and el.text is not None:
            cls = _text_mark_class(el.text)
        elif el.classes[0] == "dots" and _is_single_dot(el.bbox):
            # Verovio draws augmentation dots as plain circles, so there is no glyph to map;
            # a chord's dots share one tall box, which is skipped
            cls = "augmentationDot"
        elif el.classes[0] == "meterSig":
            cls = _metersig_class(el, metersig_glyphs.get(idx, ()))
            cover = [idx, *metersig_children.get(idx, ())]
        elif el.classes[0] == "note":
            parts = note_parts.get(idx, [])
            cls = _standalone_note_class(el, page, parts, note_codes.get(idx, set()))
            if cls is not None and parts:
                body_box: BBox | None = None
                for p in parts:
                    body_box = union(body_box, page.elements[p].bbox)
                if body_box is not None:
                    box = body_box
                # Cover only notehead/stem/flag so attached artic/accid/dots still emit
                cover = parts
        if cls is not None and box is not None and (allowed is None or cls in allowed):
            px_box = to_px(box)
            if px_box is not None:
                out.append((cls, px_box, geometry.context(box)))
                covered_elements.update(cover)

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
            out.append((cls, px_box, geometry.context(glyph.bbox)))

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

    names = [s.name for s in symbol_specs(groups)]
    return [(cls, patch) for cls, patch, _ in _page_crops(page, grey, names, size, invert)]


def _page_crops(
    page: PageAnnotation, grey: np.ndarray, classes: Sequence[str], size: int, invert: bool
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(class, patch, context vector)`` crops of one page, for the named classes."""
    h, w = grey.shape[:2]
    allowed = set(classes)
    crops: list[tuple[str, np.ndarray, np.ndarray]] = []
    for cls, bbox, ctx in page_symbol_boxes_with_context(page, (w, h), allowed=allowed):
        patch = _crop_box_to_patch(grey, bbox, size, invert=invert)
        if patch is not None:
            crops.append((cls, patch, context_vector(ctx)))
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


_PAGE_STEM_RE = re.compile(r"_page\d+$")


def _score_of_stem(stem: str) -> str:
    """``<score>_page<N>`` -> ``<score>``."""
    return _PAGE_STEM_RE.sub("", stem)


def _extract_page_task(
    task: tuple[Path, Path, tuple[str, ...], int, bool],
) -> list[tuple[str, np.ndarray, str, np.ndarray]]:
    """``(class, patch, score, context)`` crops of one canonical ``pages/`` page."""
    page_path, img_path, classes, size, invert = task
    page = PageAnnotation.load(page_path)
    with Image.open(img_path) as im:
        grey = np.asarray(im.convert("L"), dtype=np.uint8)
    return [
        (cls, patch, page.score, ctx)
        for cls, patch, ctx in _page_crops(page, grey, classes, size, invert)
    ]


def _extract_annotation_task(
    task: tuple[Path, Path, dict[str, str], frozenset[str], int, bool],
) -> list[tuple[str, np.ndarray, str, np.ndarray]]:
    """``(class, patch, score, context)`` crops of one ``annotations/*.json`` page; the
    detection view has no staff geometry, so the context is unknown (zeros)."""
    ann_path, img_path, fing_ids, allowed, size, invert = task
    record = json.loads(ann_path.read_text(encoding="utf-8"))
    with Image.open(img_path) as im:
        grey = np.asarray(im.convert("L"), dtype=np.uint8)
    score = _score_of_stem(ann_path.stem)
    crops: list[tuple[str, np.ndarray, str, np.ndarray]] = []
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
            crops.append((cls, patch, score, context_vector(None)))
    return crops


def _extract_dataset_symbols(
    dataset: Path | Sequence[Path],
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    invert: bool = False,
    max_per_class: int = 0,
    seed: int = 0,
    workers: int | None = 1,
    classes: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    roots = [dataset] if isinstance(dataset, Path) else list(dataset)
    specs = symbol_specs(groups, classes)
    class_names = tuple(s.name for s in specs)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    allowed = frozenset(class_to_idx)

    page_tasks: list[tuple[Path, Path, tuple[str, ...], int, bool]] = []
    ann_tasks: list[tuple[Path, Path, dict[str, str], frozenset[str], int, bool]] = []
    for root in roots:
        pages_dir = root / "pages"
        imgs_dir = root / "imgs"
        ann_dir = root / "annotations"
        page_files = sorted(pages_dir.glob(f"*{PAGE_SUFFIX}")) if pages_dir.is_dir() else []
        if page_files:
            for page_path in page_files:
                img_path = imgs_dir / f"{page_file_stem(page_path)}.png"
                if img_path.exists():
                    page_tasks.append((page_path, img_path, class_names, size, invert))
        elif ann_dir.is_dir():
            fing_ids = _fingering_id_map(pages_dir)
            for ann_path in sorted(ann_dir.glob("*.json")):
                img_path = imgs_dir / f"{ann_path.stem}.png"
                if img_path.exists():
                    ann_tasks.append((ann_path, img_path, fing_ids, allowed, size, invert))

    by_class: dict[int, list[tuple[np.ndarray, str, np.ndarray]]] = {
        i: [] for i in range(len(class_names))
    }
    for crops in _run_tasks(_extract_page_task, page_tasks, workers, 8):
        for cls, patch, score, ctx in crops:
            by_class[class_to_idx[cls]].append((patch, score, ctx))
    for crops in _run_tasks(_extract_annotation_task, ann_tasks, workers, 8):
        for cls, patch, score, ctx in crops:
            by_class[class_to_idx[cls]].append((patch, score, ctx))

    rng = np.random.default_rng(seed)
    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    scores_list: list[str] = []
    contexts_list: list[np.ndarray] = []
    for cls_idx, patches in by_class.items():
        if not patches:
            continue
        if max_per_class > 0 and len(patches) > max_per_class:
            order = rng.permutation(len(patches))[:max_per_class]
            patches = [patches[int(i)] for i in order]
        images_list.extend(p for p, _, _ in patches)
        scores_list.extend(sc for _, sc, _ in patches)
        contexts_list.extend(c for _, _, c in patches)
        labels_list.extend([cls_idx] * len(patches))

    if not images_list:
        return (
            np.empty((0, size, size), dtype=np.uint8),
            np.empty(0, dtype=np.int64),
            class_names,
            np.empty(0, dtype=str),
            np.empty((0, CONTEXT_DIM), dtype=np.float32),
        )
    return (
        np.stack(images_list, axis=0),
        np.array(labels_list, dtype=np.int64),
        class_names,
        np.array(scores_list, dtype=str),
        np.stack(contexts_list, axis=0).astype(np.float32),
    )


def extract_dataset_symbols(
    dataset: Path | Sequence[Path],
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    invert: bool = False,
    max_per_class: int = 0,
    seed: int = 0,
    workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract symbol crops across one or more built ``musicset`` dataset directories.

    Reads canonical ``pages/*.json.gz`` when present, falling back to
    ``annotations/*.json`` (for datasets produced by ``musicset augment``).
    ``workers`` (``1``: inline, ``None``: musicset's default pool) only changes
    the speed.
    """
    images, labels, class_names, _, _ = _extract_dataset_symbols(
        dataset,
        groups=groups,
        size=size,
        invert=invert,
        max_per_class=max_per_class,
        seed=seed,
        workers=workers,
    )
    return images, labels, class_names


def extract_dataset_symbols_with_scores(
    dataset: Path | Sequence[Path],
    *,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    invert: bool = False,
    max_per_class: int = 0,
    seed: int = 0,
    workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """:func:`extract_dataset_symbols` plus the score name each crop came from,
    the grouping key of the train/val/test split."""
    images, labels, class_names, scores, _ = _extract_dataset_symbols(
        dataset,
        groups=groups,
        size=size,
        invert=invert,
        max_per_class=max_per_class,
        seed=seed,
        workers=workers,
    )
    return images, labels, class_names, scores


# =============================================================================
# Dataset assembly, splits, saving and loading
# =============================================================================


@dataclass(frozen=True, slots=True)
class SymbolDatasetReport:
    out: Path
    classes: tuple[str, ...]
    counts: dict[str, int]
    splits: dict[str, int]
    source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    """``{split: {source name: samples}}``."""
    removed: dict[str, int] = field(default_factory=dict)
    """Samples dropped: ``blank`` (under ``min_ink`` ink pixels) and ``duplicate``."""


def _normalized_ratios(ratios: tuple[float, float, float]) -> tuple[float, float, float]:
    r_train, r_val, r_test = ratios
    total_r = r_train + r_val + r_test
    if total_r <= 0 or min(ratios) < 0:
        raise ValueError(f"invalid split ratios {ratios}")
    return r_train / total_r, r_val / total_r, r_test / total_r


def _stratified_split_indices(
    labels: np.ndarray,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Per-class random split of sample indices into ``train``, ``val``, ``test``."""
    _, r_val, r_test = _normalized_ratios(ratios)
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
    return {k: np.array(v, dtype=np.int64) for k, v in buckets.items()}


def _group_split_indices(
    groups: np.ndarray,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Split sample indices so every group (score) lands in exactly one split.

    Groups are shuffled and dealt to ``test``, then ``val``, until each holds
    its share of the samples; the remaining groups are ``train``.
    """
    _, r_val, r_test = _normalized_ratios(ratios)
    names, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(names))
    want = {"test": r_test * len(groups), "val": r_val * len(groups)}
    filled = {"test": 0, "val": 0}
    assign = np.full(len(names), "train", dtype=object)
    for g in rng.permutation(len(names)):
        for split in ("test", "val"):
            if filled[split] < want[split]:
                assign[g] = split
                filled[split] += int(sizes[g])
                break
    split_of_sample = assign[inverse]
    return {
        name: np.flatnonzero(split_of_sample == name).astype(np.int64)
        for name in ("train", "val", "test")
    }


def _fill_contexts(
    contexts: np.ndarray,
    labels: np.ndarray,
    fill: np.ndarray,
    pools: dict[int, np.ndarray],
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Give the ``fill`` rows a context drawn from ``pools[label]`` with probability ``p``."""
    out = contexts.copy()
    for cls_idx in np.unique(labels[fill]).tolist():
        pool = pools.get(cls_idx)
        if pool is None:
            continue
        rows = np.flatnonzero(fill & (labels == cls_idx))
        chosen = rows[rng.random(len(rows)) < p]
        out[chosen] = pool[rng.integers(len(pool), size=len(chosen))]
    return out


def _ink_pixels(images: np.ndarray, invert: bool) -> np.ndarray:
    if len(images) == 0:
        return np.zeros(0, dtype=np.int64)
    ink = images > 127 if invert else images < 128
    return ink.reshape(len(images), -1).sum(axis=1)


def _image_hashes(images: np.ndarray) -> list[bytes]:
    return [hashlib.blake2b(img.tobytes(), digest_size=16).digest() for img in images]


def build_symbol_dataset(
    out: Path,
    *,
    samples_per_class: int = 200,
    from_dataset: Path | Sequence[Path] | None = None,
    max_crops_per_class: int = 0,
    homus: Path | None = None,
    strokes: Sequence[Path] = (),
    emnist: Path | None = None,
    emnist_per_class: int = 1000,
    fornes: Path | None = None,
    fornes_copies: int = 1,
    profile: str | None = None,
    homus_copies: int = 1,
    homus_brush: float = 0.0,
    context_p: float = 0.7,
    groups: Sequence[str] = SYMBOL_GROUPS,
    size: int = 32,
    preset: str = "mixed",
    strength: float = 1.0,
    invert: bool = False,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_ink: int = 4,
    dedup: bool = True,
    write_png: bool = True,
    seed: int = 0,
    workers: int | None = 1,
) -> SymbolDatasetReport:
    """Build a complete MNIST-like symbol dataset into ``out``.

    Synthetic samples are split per class at random; score crops are split by
    score, so crops of one score never land in two splits. ``homus`` adds the
    HOMUS pen strokes (``SOURCE_HAND``) split by writer: train samples are
    rendered ``homus_copies`` times with tablet augmentation at their jittered
    natural size (as a pressure brush with probability ``homus_brush``),
    val/test samples once and deterministically with ``EDITOR_PEN_WIDTH``, the way an
    editor would send them, and their raw strokes are kept in
    ``strokes_val.json`` / ``strokes_test.json`` for stroke-level evaluation.
    Patches with fewer than ``min_ink`` ink pixels are dropped, and with
    ``dedup`` a byte-identical image is kept in one split only (``test`` wins
    over ``val`` over ``train``), since identical engravings recur across pages
    and scores.

    Every sample gets a conditioning vector (:mod:`musicset.context`,
    ``c_<split>``): score crops their real one, contributed strokes the one
    their editor recorded, and every other sample, with probability
    ``context_p``, one drawn from the real contexts of training crops of its
    class (else "unknown", all zeros), so the classifier learns where each
    symbol lives without ever needing the context.

    Writes:
    * ``out/dataset.npz`` (``x_<split>``, ``y_<split>``, ``s_<split>`` source
      codes indexing ``SOURCE_NAMES``, ``w_<split>`` writer ids (``-1`` for
      synthetic and crops), ``c_<split>`` context vectors, ``classes``);

    ``emnist`` (the directory of NIST's ``byclass`` files) adds ``emnist_per_class``
    real handwritten words per fingering, dynamic and navigation class
    (:mod:`musicset.emnist`), train and val from EMNIST's train file, test from its
    test file. With ``homus`` and a ``coda`` class, each HOMUS writer also contributes
    codas composed from their own strokes (:func:`musicset.homus.compose_homus_codas`).

    ``fornes`` (the ``Music_Symbols.zip`` archive, :mod:`musicset.fornes`) adds scanned
    handwritten clefs and accidentals (``SOURCE_FORNES``), split by writer within two
    strata — the writers with accidentals and those with clefs only — so val and test
    hold both; train scans are rendered ``fornes_copies`` times. Its split, renders and
    contexts draw from their own generator, so adding it leaves every other sample of a
    build as it was: only the order changes, and with it which copy of an image repeated
    within a split survives deduplication (22 of 148,791 samples in the battuta build,
    the same image and label with another context).

    ``profile`` (``SYMBOL_PROFILES``) narrows the classes to one consumer's set:
    ``battuta`` for the editor's stylus model, ``kakigori`` for the OMR pipeline, which is
    engraved only (no stroke templates, and ``homus`` / ``strokes`` / ``emnist`` /
    ``fornes`` refused).
    * ``out/labels.json`` (class list, index map, group map, build settings,
      per-class, per-split and per-source counts, removed samples);
    * ``out/strokes_<split>.json`` for HOMUS val/test samples;
    * ``out/<split>/<class>/<index>.png`` when ``write_png=True``.
    """
    if samples_per_class < 0:
        raise ValueError("samples_per_class must be >= 0")
    if (
        samples_per_class == 0
        and not from_dataset
        and homus is None
        and not strokes
        and not emnist
        and fornes is None
    ):
        raise ValueError(
            "need samples_per_class > 0, from_dataset, homus, strokes, emnist or fornes"
        )

    profile_classes: tuple[str, ...] | None = None
    stroke_templates = True
    if profile is not None:
        if profile not in SYMBOL_PROFILES:
            raise ValueError(f"unknown profile {profile!r}: {sorted(SYMBOL_PROFILES)}")
        prof = SYMBOL_PROFILES[profile]
        handwriting = homus is not None or strokes or emnist is not None or fornes is not None
        if not prof.handwriting and handwriting:
            raise ValueError(
                f"profile {profile!r} is engraved only: drop homus, strokes, emnist, fornes"
            )
        profile_classes, stroke_templates = prof.classes, prof.handwriting
    specs = symbol_specs(groups, profile_classes)
    if not specs:
        raise ValueError(f"no class left in groups {list(groups)} for profile {profile!r}")
    class_names = tuple(s.name for s in specs)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    rng = np.random.default_rng(seed)
    removed = {"blank": 0, "duplicate": 0}
    pieces: dict[
        str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {
        "train": [],
        "val": [],
        "test": [],
    }
    crop_scores: dict[str, int] = {}
    homus_writers: dict[str, int] = {}
    stroke_records: dict[str, list[dict[str, Any]]] = {}
    hand_piece_index: dict[str, int] = {}

    def inked_mask(x: np.ndarray) -> np.ndarray:
        inked = _ink_pixels(x, invert) >= min_ink
        removed["blank"] += int((~inked).sum())
        return inked

    def no_writer(n: int) -> np.ndarray:
        return np.full(n, -1, dtype=np.int32)

    def unknown_context(n: int) -> np.ndarray:
        return np.zeros((n, CONTEXT_DIM), dtype=np.float32)

    if samples_per_class > 0:
        sx, sy, _, ss = _generate_symbol_samples(
            samples_per_class,
            groups=groups,
            size=size,
            preset=preset,
            strength=strength,
            invert=invert,
            seed=seed,
            workers=workers,
            classes=profile_classes,
            stroke_templates=stroke_templates,
        )
        m = inked_mask(sx)
        sx, sy, ss = sx[m], sy[m], ss[m]
        for name, idx in _stratified_split_indices(sy, split, rng).items():
            n = len(idx)
            pieces[name].append(
                (sx[idx], sy[idx], ss[idx], no_writer(n), unknown_context(n), np.ones(n, bool))
            )

    if from_dataset:
        cx, cy, _, cg, cc = _extract_dataset_symbols(
            from_dataset,
            groups=groups,
            size=size,
            invert=invert,
            max_per_class=max_crops_per_class,
            seed=seed,
            workers=workers,
            classes=profile_classes,
        )
        cs = np.full(len(cy), SOURCE_CROP, dtype=np.uint8)
        m = inked_mask(cx)
        cx, cy, cs, cg, cc = cx[m], cy[m], cs[m], cg[m], cc[m]
        if len(cx):
            idx = _group_split_indices(cg, split, rng)
            crop_scores = {name: len(np.unique(cg[i])) for name, i in idx.items()}
            for name, i in idx.items():
                pieces[name].append(
                    (cx[i], cy[i], cs[i], no_writer(len(i)), cc[i], np.zeros(len(i), bool))
                )

    if emnist is not None and emnist_per_class > 0:
        r_train, r_val, r_test = _normalized_ratios(split)
        emnist_classes = [n for n in class_names if n in EMNIST_SYMBOLS or n in EMNIST_SIGNS]
        wanted = {c for n in emnist_classes for w in EMNIST_SYMBOLS.get(n, ()) for c in w}
        wanted |= {c for n in emnist_classes for cs in EMNIST_SIGNS.get(n, ()) for c in cs}
        chars = "".join(sorted(wanted - {"."}))
        glyph_pools: dict[str, dict[str, np.ndarray]] = {}
        for file_split in ("train", "test"):
            xs, cs = load_emnist(emnist, file_split, chars=chars)
            glyph_pools[file_split] = {c: xs[cs == c] for c in chars}
        n_test = round(emnist_per_class * r_test / max(1e-9, r_train + r_val))
        for name, file_split, n_words in (
            ("train", "train", emnist_per_class),
            ("test", "test", n_test),
        ):
            em_tasks: list[EmnistTask] = []
            em_labels: list[int] = []
            pool = glyph_pools[file_split]

            for cls_name in emnist_classes:
                spec = SYMBOL_BY_NAME[cls_name]
                lo, hi = spec.scale_range
                overlay = cls_name in EMNIST_SIGNS
                for k in range(n_words):
                    j = class_to_idx[cls_name] * 1_000_000 + k + (500_000 if name == "test" else 0)
                    if overlay:
                        word = cls_name
                        glyphs = tuple(
                            _pick_emnist(pool, cs, rng, file_split) for cs in EMNIST_SIGNS[cls_name]
                        )
                    else:
                        words = EMNIST_SYMBOLS[cls_name]
                        word = words[int(rng.integers(len(words)))]
                        glyphs = tuple(_pick_emnist(pool, c, rng, file_split) for c in word)
                    em_tasks.append(
                        (
                            glyphs,
                            word,
                            overlay,
                            lo,
                            hi,
                            size,
                            strength,
                            invert,
                            seed,
                            j,
                            name == "train",
                        )
                    )
                    em_labels.append(class_to_idx[cls_name])
            if not em_tasks:
                continue
            em_x = np.empty((len(em_tasks), size, size), dtype=np.uint8)
            for n, image in enumerate(_run_tasks(_render_emnist_task, em_tasks, workers, 64)):
                em_x[n] = image
            em_y = np.array(em_labels, dtype=np.int64)
            em_mask = inked_mask(em_x)
            em_x, em_y = em_x[em_mask], em_y[em_mask]
            em_s = np.full(len(em_y), SOURCE_EMNIST, dtype=np.uint8)
            if name == "train":
                # EMNIST's train file feeds train and val; its test file (other writers) feeds test
                for part, idx in _stratified_split_indices(
                    em_y, (r_train, r_val, 0.0), rng
                ).items():
                    if len(idx):
                        n_ = len(idx)
                        pieces[part].append(
                            (
                                em_x[idx],
                                em_y[idx],
                                em_s[idx],
                                no_writer(n_),
                                unknown_context(n_),
                                np.ones(n_, bool),
                            )
                        )
            else:
                n_ = len(em_y)
                pieces["test"].append(
                    (em_x, em_y, em_s, no_writer(n_), unknown_context(n_), np.ones(n_, bool))
                )

    hand: list[StrokeSample] = []
    if homus is not None:
        homus_samples = load_homus(homus)
        hand += [smp for smp in homus_samples if smp.label in class_to_idx]
        if "coda" in class_to_idx:
            hand += compose_homus_codas(homus_samples, seed=seed)
    if strokes:
        hand += load_stroke_contributions(list(strokes), classes=set(class_to_idx))
    if hand:
        samples = hand
        if samples:
            writers = np.array([str(smp.writer) for smp in samples])
            idx = _group_split_indices(writers, split, rng)
            homus_writers = {name: len(np.unique(writers[i])) for name, i in idx.items()}
            for name, i in idx.items():
                train = name == "train"
                copies = max(1, homus_copies) if train else 1
                tasks = [
                    (
                        samples[j].strokes,
                        _sample_fraction(samples[j]),
                        size,
                        strength,
                        invert,
                        seed,
                        int(j),
                        k,
                        train,
                        homus_brush,
                    )
                    for j in i.tolist()
                    for k in range(copies)
                ]
                hx = np.empty((len(tasks), size, size), dtype=np.uint8)
                hy = np.array(
                    [class_to_idx[samples[j].label] for j in i.tolist() for _ in range(copies)],
                    dtype=np.int64,
                )
                for n, image in enumerate(_run_tasks(_render_hand_task, tasks, workers, 64)):
                    hx[n] = image
                hs = np.full(len(hy), SOURCE_HAND, dtype=np.uint8)
                order_j = [j for j in i.tolist() for _ in range(copies)]
                hw = np.array([samples[j].writer for j in order_j], dtype=np.int32)
                hc = (
                    np.stack([context_vector(samples[j].context) for j in order_j]).astype(
                        np.float32
                    )
                    if order_j
                    else unknown_context(0)
                )
                hfill = np.array([samples[j].context is None for j in order_j], dtype=bool)
                m = inked_mask(hx)
                hand_piece_index[name] = len(pieces[name])
                pieces[name].append((hx[m], hy[m], hs[m], hw[m], hc[m], hfill[m]))
                if not train:
                    kept = [j for j, ok in zip(order_j, m.tolist(), strict=True) if ok]
                    stroke_records[name] = [
                        {
                            "writer": samples[j].writer,
                            "label": samples[j].label,
                            "fraction": _sample_fraction(samples[j]),
                            "strokes": [[list(p) for p in stroke] for stroke in samples[j].strokes],
                        }
                        for j in kept
                    ]

    # Fornés scans: their own generator for the split, the renders and the contexts, so the
    # rest of the build draws exactly what it drew without them
    side_rng = np.random.default_rng([seed, SOURCE_FORNES])
    side_pieces: dict[
        str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {"train": [], "val": [], "test": []}
    fornes_writers: dict[str, int] = {}
    if fornes is not None:
        scans = [smp for smp in load_fornes(fornes) if smp.label in class_to_idx]
        fo_ids = fornes_writer_ids(scans)
        fo_writers = np.array([smp.writer for smp in scans], dtype=str)
        with_accidentals = {
            s.writer for s in scans if SYMBOL_BY_NAME[s.label].group == "accidental"
        }
        stratum = np.array([w in with_accidentals for w in fo_writers], dtype=bool)
        fo_rows: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for part in (stratum, ~stratum):
            rows = np.flatnonzero(part)
            for name, i in _group_split_indices(fo_writers[rows], split, side_rng).items():
                fo_rows[name] += rows[i].tolist()
        for name, rows_ in fo_rows.items():
            if not rows_:
                continue
            fornes_writers[name] = len({fo_writers[j] for j in rows_})
            train = name == "train"
            copies = max(1, fornes_copies) if train else 1
            order_j = [j for j in rows_ for _ in range(copies)]
            fo_tasks: list[FornesTask] = [
                (
                    scans[j].image,
                    *SYMBOL_BY_NAME[scans[j].label].scale_range,
                    size,
                    strength,
                    invert,
                    seed,
                    j,
                    k,
                    train,
                )
                for j in rows_
                for k in range(copies)
            ]
            fx = np.empty((len(fo_tasks), size, size), dtype=np.uint8)
            for n, image in enumerate(_run_tasks(_render_fornes_task, fo_tasks, workers, 64)):
                fx[n] = image
            fy = np.array([class_to_idx[scans[j].label] for j in order_j], dtype=np.int64)
            fw = np.array([fo_ids[scans[j].writer] for j in order_j], dtype=np.int32)
            m = inked_mask(fx)
            n_ = int(m.sum())
            side_pieces[name].append(
                (
                    fx[m],
                    fy[m],
                    np.full(n_, SOURCE_FORNES, dtype=np.uint8),
                    fw[m],
                    unknown_context(n_),
                    np.ones(n_, bool),
                )
            )

    # Contexts for samples without a real one: drawn from the real crops of their class
    # (with family fallback for rare note durations / meters, and split-local pools when >= 10)
    stemmed_idxs = [
        class_to_idx[c]
        for c in ("noteHalf", "noteQuarter", "note8th", "note16th", "note32nd", "note64th")
        if c in class_to_idx
    ]
    timesig_idxs = [i for i, s in enumerate(specs) if s.group == "timesig"]

    def build_pools(
        split_pieces: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> dict[int, np.ndarray]:
        raw: dict[int, np.ndarray] = {}
        for _, py, _, _, pc, pfill in split_pieces:
            real = ~pfill & (pc[:, 0] > 0)
            for cls_idx in np.unique(py[real]).tolist():
                rows = pc[real & (py == cls_idx)]
                raw[cls_idx] = np.concatenate([raw[cls_idx], rows]) if cls_idx in raw else rows
        out = {k: v for k, v in raw.items() if len(v) >= 10}
        for family in (stemmed_idxs, timesig_idxs):
            family_rows = [raw[i] for i in family if i in raw]
            if family_rows:
                merged = np.concatenate(family_rows)
                if len(merged) >= 10:
                    for i in family:
                        out.setdefault(i, merged)
        return out

    pools = build_pools(pieces["train"])
    for name, parts in pieces.items():
        split_pools = pools if name == "train" else {**pools, **build_pools(parts)}
        for n_, (px, py, ps, pw, pc, pfill) in enumerate(parts):
            if pfill.any():
                parts[n_] = (
                    px,
                    py,
                    ps,
                    pw,
                    _fill_contexts(pc, py, pfill, split_pools, context_p, rng),
                    pfill,
                )
        for px, py, ps, pw, pc, pfill in side_pieces[name]:
            fc = _fill_contexts(pc, py, pfill, split_pools, context_p, side_rng)
            parts.append((px, py, ps, pw, fc, pfill))
    for name, records in stroke_records.items():
        piece_c = pieces[name][hand_piece_index[name]][4]
        for record, ctx in zip(records, piece_c, strict=True):
            record["context"] = [round(float(v), 4) for v in ctx]

    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        parts = pieces[name]
        x = np.concatenate([p[0] for p in parts]) if parts else np.empty((0, size, size), np.uint8)
        y = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, dtype=np.int64)
        s = np.concatenate([p[2] for p in parts]) if parts else np.empty(0, dtype=np.uint8)
        w = np.concatenate([p[3] for p in parts]) if parts else np.empty(0, dtype=np.int32)
        c = np.concatenate([p[4] for p in parts]) if parts else unknown_context(0)
        order = rng.permutation(len(y))
        splits[name] = (x[order], y[order], s[order], w[order], c[order])

    if dedup:
        seen: set[bytes] = set()
        for name in ("test", "val", "train"):
            x, y, s, w, c = splits[name]
            keep = np.ones(len(y), dtype=bool)
            for i, h in enumerate(_image_hashes(x)):
                if h in seen:
                    keep[i] = False
                else:
                    seen.add(h)
            removed["duplicate"] += int((~keep).sum())
            splits[name] = (x[keep], y[keep], s[keep], w[keep], c[keep])

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz",
        x_train=splits["train"][0],
        y_train=splits["train"][1],
        s_train=splits["train"][2],
        w_train=splits["train"][3],
        c_train=splits["train"][4],
        x_val=splits["val"][0],
        y_val=splits["val"][1],
        s_val=splits["val"][2],
        w_val=splits["val"][3],
        c_val=splits["val"][4],
        x_test=splits["test"][0],
        y_test=splits["test"][1],
        s_test=splits["test"][2],
        w_test=splits["test"][3],
        c_test=splits["test"][4],
        classes=np.array(class_names),
    )
    for name, records in stroke_records.items():
        (out / f"strokes_{name}.json").write_text(json.dumps(records), encoding="utf-8")

    if write_png:
        for split_name, (xs_, ys_, _, _, _) in splits.items():
            if not len(xs_):
                continue
            split_dir = out / split_name
            for cls_name in class_names:
                (split_dir / cls_name).mkdir(parents=True, exist_ok=True)
            counters: Counter[int] = Counter()
            for img, cls_idx in zip(xs_, ys_, strict=True):
                c = int(cls_idx)
                k = counters[c]
                counters[c] = k + 1
                Image.fromarray(img, mode="L").save(
                    split_dir / class_names[c] / f"{k:06d}.png", compress_level=1
                )

    labels = np.concatenate([v[1] for v in splits.values()])
    counts = {name: int((labels == i).sum()) for i, name in enumerate(class_names)}
    split_counts = {k: len(v[1]) for k, v in splits.items()}
    source_counts = {
        k: {SOURCE_NAMES[c]: int((v[2] == c).sum()) for c in range(len(SOURCE_NAMES))}
        for k, v in splits.items()
    }
    by_group: dict[str, list[str]] = {}
    for spec in specs:
        by_group.setdefault(spec.group, []).append(spec.name)
    roots = [from_dataset] if isinstance(from_dataset, Path) else list(from_dataset or [])

    meta: dict[str, Any] = {
        "profile": profile,
        "stroke_templates": stroke_templates,
        "classes": list(class_names),
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "groups": by_group,
        "size": size,
        "invert": invert,
        "preset": preset,
        "strength": strength,
        "samples_per_class": samples_per_class,
        "from_dataset": [str(p) for p in roots],
        "max_crops_per_class": max_crops_per_class,
        "homus": None if homus is None else str(homus),
        "emnist": None if emnist is None else str(emnist),
        "emnist_per_class": emnist_per_class,
        "fornes": None if fornes is None else str(fornes),
        "fornes_copies": fornes_copies,
        "strokes": [str(p) for p in strokes],
        "homus_copies": homus_copies,
        "homus_brush": homus_brush,
        "split": list(split),
        "min_ink": min_ink,
        "dedup": dedup,
        "seed": seed,
        "source_names": list(SOURCE_NAMES),
        "counts": counts,
        "splits": split_counts,
        "source_counts": source_counts,
        "crop_scores": crop_scores,
        "homus_writers": homus_writers,
        "fornes_writers": fornes_writers,
        "context_features": list(CONTEXT_FEATURES),
        "context_p": context_p,
        "context_pools": {class_names[k]: len(v) for k, v in sorted(pools.items())},
        "context_counts": {k: int((v[4][:, 0] > 0).sum()) for k, v in splits.items()},
        "removed": removed,
    }
    (out / "labels.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return SymbolDatasetReport(
        out=out,
        classes=class_names,
        counts=counts,
        splits=split_counts,
        source_counts=source_counts,
        removed=removed,
    )


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


def load_symbol_sources(path: Path, split: str = "train") -> np.ndarray | None:
    """Per-sample source codes (indices into ``SOURCE_NAMES``) for ``split``, or
    ``None`` for an archive built before sources were recorded."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}: ('train', 'val', 'test')")
    npz_path = path / "dataset.npz" if path.is_dir() else path
    with np.load(npz_path) as data:
        return data.get(f"s_{split}")


__all__ = [
    "EDITOR_PEN_WIDTH",
    "PRESETS",
    "SOURCE_CROP",
    "SOURCE_EMNIST",
    "SOURCE_FONT",
    "SOURCE_FORNES",
    "SOURCE_HAND",
    "SOURCE_NAMES",
    "SOURCE_STROKE",
    "SYMBOLS",
    "SYMBOL_BY_NAME",
    "SYMBOL_GROUPS",
    "SymbolDatasetReport",
    "SymbolSpec",
    "build_symbol_dataset",
    "extract_dataset_symbols",
    "extract_dataset_symbols_with_scores",
    "extract_page_symbols",
    "generate_symbol_samples",
    "generate_symbol_samples_with_sources",
    "ink_stroke_width",
    "load_symbol_dataset",
    "load_symbol_sources",
    "normalize_strokes",
    "page_symbol_boxes",
    "render_ink_sample",
    "render_strokes_sample",
    "render_symbol",
    "render_symbol_with_source",
    "symbol_specs",
]
