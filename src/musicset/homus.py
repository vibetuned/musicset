"""HOMUS: the Handwritten Online Musical Symbols dataset (Calvo-Zaragoza & Oncina, 2014).

15,200 symbols drawn on a tablet by 100 writers, stored as pen strokes: one
text file per symbol, the class name on the first line and one stroke per
following line as ``x,y;x,y;...``. Coordinates share one canvas across the
whole dataset (about 210 x 257 pen units), so a symbol's drawn size is
meaningful: a G clef spans ~127 units, an augmentation dot ~8.

:func:`load_homus` reads the archive (zip or extracted directory) into
:class:`StrokeSample` records whose ``label`` is already the ``musicset``
symbol class, and :func:`stroke_fraction` turns a drawing's size into the
canvas fraction :func:`musicset.symbols.render_strokes_sample` should use.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musicset.context import SymbolContext

Point = tuple[float, float]
Stroke = tuple[Point, ...]

HOMUS_CLASSES: dict[str, str] = {
    "Barline": "barlineSingle",
    "C-Clef": "cClef",
    "Dot": "augmentationDot",
    "Double-Sharp": "accidentalDoubleSharp",
    "Eighth-Note": "note8th",
    "Eighth-Rest": "rest8th",
    "F-Clef": "fClef",
    "Flat": "accidentalFlat",
    "G-Clef": "gClef",
    "Half-Note": "noteHalf",
    "Natural": "accidentalNatural",
    "Quarter-Note": "noteQuarter",
    "Quarter-Rest": "restQuarter",
    "Sharp": "accidentalSharp",
    "Sixteenth-Note": "note16th",
    "Sixteenth-Rest": "rest16th",
    "Sixty-Four-Note": "note64th",
    "Sixty-Four-Rest": "rest64th",
    "Thirty-Two-Note": "note32nd",
    "Thirty-Two-Rest": "rest32nd",
    "Whole-Half-Rest": "restWholeHalf",
    "Whole-Note": "noteWhole",
}
"""HOMUS class name -> ``musicset`` symbol class. HOMUS's ten time-signature classes are
left out on purpose: stacked digits hurt every other class (fingerings above all)."""

HOMUS_CANVAS = 150.0
"""Pen units that fill the symbol canvas: a G clef (~127 units) spans ~85% of it."""

CANVAS_STAFF_SPACES = 8.8
"""Staff spaces that fill the symbol canvas, the same size rule in score units.

HOMUS writers drew at natural proportions: a whole note spans ~24 pen units, a
quarter note ~55, a sharp ~52, a G clef ~127, i.e. one staff space is ~17 pen
units, so the 150-unit canvas is ~8.8 staff spaces. An editor that knows its
staff space (pixels between two staff lines) uses
:func:`stroke_fraction_from_staff_space` and lands on the same scale.
"""


@dataclass(frozen=True, slots=True)
class StrokeSample:
    writer: int
    label: str
    """``musicset`` symbol class (see ``HOMUS_CLASSES``)."""
    strokes: tuple[Stroke, ...]
    source: str = ""
    """File the sample came from, for error messages."""
    fraction: float | None = None
    """Canvas fraction the drawing should span; ``None`` means ``stroke_fraction`` (HOMUS units)."""
    context: SymbolContext | None = None
    """Where in the score the mark was drawn, when the contributor's editor recorded it."""


def parse_homus(text: str) -> tuple[str, tuple[Stroke, ...]]:
    """``(HOMUS class name, strokes)`` of one symbol file; empty strokes are dropped."""
    lines = [line.strip() for line in text.splitlines()]
    if not lines or not lines[0]:
        raise ValueError("empty HOMUS file")
    strokes: list[Stroke] = []
    for line in lines[1:]:
        points: list[Point] = []
        for token in line.split(";"):
            token = token.strip()
            if not token:
                continue
            x, y = token.split(",")
            points.append((float(x), float(y)))
        if points:
            strokes.append(tuple(points))
    return lines[0], tuple(strokes)


def _writer_of(path: str) -> int:
    """``HOMUS/<writer>/<writer>-<k>.txt`` -> writer number."""
    parts = Path(path).parts
    return (
        int(parts[-2])
        if len(parts) >= 2 and parts[-2].isdigit()
        else int(Path(path).stem.split("-")[0])
    )


def _iter_files(path: Path) -> Iterator[tuple[str, str]]:
    """``(member path, text)`` for every symbol file in a zip or directory."""
    if path.is_dir():
        for file in sorted(path.rglob("*.txt")):
            yield str(file.relative_to(path)), file.read_text(encoding="utf-8", errors="replace")
        return
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".txt"):
                yield name, archive.read(name).decode("utf-8", "replace")


def load_homus(path: Path, *, classes: dict[str, str] = HOMUS_CLASSES) -> list[StrokeSample]:
    """Every HOMUS symbol under ``path`` (zip or directory) whose class is in ``classes``."""
    samples: list[StrokeSample] = []
    for name, text in _iter_files(path):
        try:
            homus_label, strokes = parse_homus(text)
        except ValueError as e:
            raise ValueError(f"{path}:{name}: {e}") from e
        label = classes.get(homus_label)
        if label is None or not strokes:
            continue
        samples.append(StrokeSample(_writer_of(name), label, strokes, source=name))
    return samples


def stroke_span(strokes: Sequence[Sequence[Point]]) -> float:
    """Longer side of the strokes' bounding box, in pen units."""
    points = [p for stroke in strokes for p in stroke]
    if not points:
        raise ValueError("no stroke points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def stroke_fraction(
    strokes: Sequence[Sequence[Point]],
    canvas_units: float = HOMUS_CANVAS,
    lo: float = 0.08,
    hi: float = 0.90,
) -> float:
    """Fraction of the symbol canvas a drawing should span, from its size in pen units.

    Keeps the writers' relative sizes (a dot stays small, a clef large), which
    a bounding-box normalisation would throw away.
    """
    return float(min(hi, max(lo, stroke_span(strokes) / canvas_units)))


def stroke_fraction_from_staff_space(
    strokes: Sequence[Sequence[Point]],
    staff_space: float,
    lo: float = 0.08,
    hi: float = 0.90,
) -> float:
    """Canvas fraction for strokes drawn in a score editor whose staff space (distance
    between two staff lines, in the strokes' units) is ``staff_space``."""
    if staff_space <= 0:
        raise ValueError(f"staff_space must be positive, got {staff_space}")
    return float(min(hi, max(lo, stroke_span(strokes) / (CANVAS_STAFF_SPACES * staff_space))))


CODA_SPACES = (3.0, 4.6)
"""Height of a composed handwritten coda in staff spaces (engraved ones span 2.5-4.9;
handwritten ones run a little larger)."""


def _principal(stroke: Sequence[Point]) -> tuple[np.ndarray, np.ndarray] | None:
    """Coordinates of a stroke along and across its main direction, centred."""
    arr = np.asarray(stroke, dtype=np.float64)
    if len(arr) < 2:
        return None
    centred = arr - arr.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along, across = centred @ vt[0], centred @ vt[1]
    if np.ptp(along) <= 1e-9:
        return None
    return along, across


def compose_homus_codas(
    samples: Sequence[StrokeSample], *, per_writer: int = 6, seed: int = 0
) -> list[StrokeSample]:
    """Handwritten codas from each HOMUS writer's own pen strokes (HOMUS has none).

    One of the writer's whole-note ovals, made roughly round, is crossed by two of
    their barlines, one stood upright and one laid level, both overshooting the circle
    by 1.3-1.7 times, the way a coda is drawn: circle first, then the cross. Every
    composite keeps its writer, so a split by writer still holds, and is sized to
    ``CODA_SPACES`` staff spaces in HOMUS pen units. Writers without a whole note or a
    barline give none.
    """
    ovals: dict[int, list[StrokeSample]] = {}
    bars: dict[int, list[StrokeSample]] = {}
    for smp in samples:
        if smp.label == "noteWhole":
            ovals.setdefault(smp.writer, []).append(smp)
        elif smp.label == "barlineSingle":
            bars.setdefault(smp.writer, []).append(smp)
    unit = HOMUS_CANVAS / CANVAS_STAFF_SPACES  # pen units per staff space
    out: list[StrokeSample] = []
    for writer in sorted(set(ovals) & set(bars)):
        rng = np.random.default_rng([seed, 11, writer])
        for k in range(per_writer):
            oval = ovals[writer][int(rng.integers(len(ovals[writer])))]
            pts = np.array([p for stroke in oval.strokes for p in stroke], dtype=np.float64)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            size = np.maximum(hi - lo, 1e-9)
            centre = 0.5 * (lo + hi)
            aspect = float(rng.uniform(0.85, 1.15))
            strokes: list[np.ndarray] = [
                (np.asarray(stroke, dtype=np.float64) - centre) / size * (aspect, 1.0)
                for stroke in oval.strokes
            ]
            for upright in (True, False):
                bar = bars[writer][int(rng.integers(len(bars[writer])))]
                axes = _principal(max(bar.strokes, key=len))
                if axes is None:
                    continue
                along, across = axes
                length = float(rng.uniform(1.3, 1.7))
                a = along / np.ptp(along) * length
                c = across / np.ptp(along) * length
                line = np.stack([c, a], axis=1) if upright else np.stack([a, c], axis=1)
                tilt = np.deg2rad(float(rng.normal(0.0, 4.0)))
                rot = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
                strokes.append(line @ rot.T + rng.uniform(-0.06, 0.06, size=2))
            allp = np.concatenate(strokes)
            height = float(np.ptp(allp[:, 1]))
            if height <= 1e-9:
                continue
            scale = float(rng.uniform(*CODA_SPACES)) * unit / height
            origin = allp.min(axis=0)
            composed = tuple(
                tuple((float(x), float(y)) for x, y in (st - origin) * scale) for st in strokes
            )
            out.append(StrokeSample(writer, "coda", composed, source=f"homus-coda/{writer}-{k}"))
    return out


__all__ = [
    "CANVAS_STAFF_SPACES",
    "CODA_SPACES",
    "HOMUS_CANVAS",
    "HOMUS_CLASSES",
    "StrokeSample",
    "compose_homus_codas",
    "load_homus",
    "parse_homus",
    "stroke_fraction",
    "stroke_fraction_from_staff_space",
    "stroke_span",
]
