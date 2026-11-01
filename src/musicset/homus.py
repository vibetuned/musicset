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

Point = tuple[float, float]
Stroke = tuple[Point, ...]

HOMUS_CLASSES: dict[str, str] = {
    "12-8-Time": "timeSig12over8",
    "2-2-Time": "timeSig2over2",
    "2-4-Time": "timeSig2over4",
    "3-4-Time": "timeSig3over4",
    "3-8-Time": "timeSig3over8",
    "4-4-Time": "timeSig4over4",
    "6-8-Time": "timeSig6over8",
    "9-8-Time": "timeSig9over8",
    "Barline": "barlineSingle",
    "C-Clef": "cClef",
    "Common-Time": "timeSigCommon",
    "Cut-Time": "timeSigCutCommon",
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
"""HOMUS class name -> ``musicset`` symbol class."""

HOMUS_CANVAS = 150.0
"""Pen units that fill the symbol canvas: a G clef (~127 units) spans ~85% of it."""


@dataclass(frozen=True, slots=True)
class StrokeSample:
    writer: int
    label: str
    """``musicset`` symbol class (see ``HOMUS_CLASSES``)."""
    strokes: tuple[Stroke, ...]
    source: str = ""
    """File the sample came from, for error messages."""


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


__all__ = [
    "HOMUS_CANVAS",
    "HOMUS_CLASSES",
    "StrokeSample",
    "load_homus",
    "parse_homus",
    "stroke_fraction",
    "stroke_span",
]
