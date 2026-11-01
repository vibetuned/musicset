"""Where a symbol sits in the score: the conditioning vector a classifier can use.

The same pen mark means different things in different places (a dot next to a
notehead is an augmentation dot, above it a staccato; a vertical bar on the
staff is a barline, above a note a fingering 1). :func:`context_vector` turns a
handful of geometric facts, all in staff spaces so they are device independent,
into ``CONTEXT_DIM`` floats roughly in ``[-1, 1]``. The facts come from a
rendered page (:func:`page_symbol_contexts`) or from a score editor's
contribution record (:func:`context_from_record`), and are zero with
``valid = 0`` when unknown.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from musicset.annotate.geometry import BBox
from musicset.annotate.schema import PageAnnotation

CONTEXT_FEATURES: tuple[str, ...] = (
    "valid",
    "y_staff",
    "x_measure",
    "barline",
    "has_note",
    "note_dx",
    "note_dy",
    "width",
    "height",
)
"""Feature order of :func:`context_vector`.

* ``valid``: 1 when the context is known, else the whole vector is 0;
* ``y_staff``: symbol centre in staff spaces from the staff's middle line,
  positive downwards, divided by 6;
* ``x_measure``: position inside the measure, left barline -1 to right barline 1;
* ``barline``: distance to the nearest barline in staff spaces, divided by 8, clipped;
* ``has_note``: 1 when a notehead (not the symbol's own) lies within 8 staff spaces;
* ``note_dx``, ``note_dy``: symbol centre minus that notehead's centre, staff spaces / 8;
* ``width``, ``height``: symbol size in staff spaces / 8, clipped.
"""
CONTEXT_DIM = len(CONTEXT_FEATURES)
NOTE_RANGE = 8.0
"""Staff spaces within which a notehead counts as the symbol's neighbour."""
NOTEHEAD_CODES = frozenset({"E0A0", "E0A1", "E0A2", "E0A3", "E0A4", "E0A5", "E0A9", "E0B3"})


@dataclass(frozen=True, slots=True)
class SymbolContext:
    """Raw facts about a symbol's place in the score, in staff spaces."""

    y_staff: float
    x_measure: float
    """0 at the left barline, 1 at the right (0.5 when no measure is known)."""
    barline: float
    note_dx: float | None
    note_dy: float | None
    width: float
    height: float


def context_vector(ctx: SymbolContext | None) -> np.ndarray:
    """``CONTEXT_DIM`` float32 features for ``ctx``; all zero when ``None``."""
    if ctx is None:
        return np.zeros(CONTEXT_DIM, dtype=np.float32)
    has_note = ctx.note_dx is not None and ctx.note_dy is not None
    dx = ctx.note_dx if ctx.note_dx is not None else 0.0
    dy = ctx.note_dy if ctx.note_dy is not None else 0.0
    return np.array(
        [
            1.0,
            max(-2.0, min(2.0, ctx.y_staff / 6.0)),
            max(-1.0, min(1.0, ctx.x_measure * 2.0 - 1.0)),
            max(0.0, min(1.0, ctx.barline / 8.0)),
            1.0 if has_note else 0.0,
            max(-1.0, min(1.0, dx / NOTE_RANGE)),
            max(-1.0, min(1.0, dy / NOTE_RANGE)),
            max(0.0, min(1.5, ctx.width / 8.0)),
            max(0.0, min(1.5, ctx.height / 8.0)),
        ],
        dtype=np.float32,
    )


def context_from_record(
    record: dict[str, Any], strokes_bbox: BBox | None, staff_space: float
) -> SymbolContext | None:
    """Context of a contributed pen sample from its ``context`` dict.

    The dict carries ``y_staff`` (staff spaces from the middle line, down is
    positive), ``x_measure`` (0-1), ``barline`` (staff spaces), optional
    ``note_dx`` / ``note_dy`` (staff spaces, ``null`` when no note is near) and
    optionally ``width`` / ``height`` (staff spaces, else from the strokes).
    """
    raw = record.get("context")
    if not isinstance(raw, dict) or "y_staff" not in raw or staff_space <= 0:
        return None
    if "width" in raw and "height" in raw:
        width, height = float(raw["width"]), float(raw["height"])
    elif strokes_bbox is not None:
        width = (strokes_bbox[2] - strokes_bbox[0]) / staff_space
        height = (strokes_bbox[3] - strokes_bbox[1]) / staff_space
    else:
        return None
    dx, dy = raw.get("note_dx"), raw.get("note_dy")
    return SymbolContext(
        y_staff=float(raw["y_staff"]),
        x_measure=float(raw.get("x_measure", 0.5)),
        barline=float(raw.get("barline", NOTE_RANGE)),
        note_dx=None if dx is None else float(dx),
        note_dy=None if dy is None else float(dy),
        width=width,
        height=height,
    )


def _centre(b: BBox | Sequence[float]) -> tuple[float, float]:
    return 0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))


class PageGeometry:
    """Staves (five-line boxes), measures and noteheads of a page, for context lookups."""

    def __init__(self, page: PageAnnotation) -> None:
        self.staves: list[BBox] = [
            e.bbox for e in page.elements if e.classes and e.classes[0] == "system-staff" and e.bbox
        ]
        self.measures: list[BBox] = [
            e.bbox for e in page.elements if e.classes and e.classes[0] == "measure" and e.bbox
        ]
        self.noteheads: np.ndarray = np.array(
            [_centre(g.bbox) for g in page.glyphs if g.code in NOTEHEAD_CODES and g.bbox],
            dtype=np.float64,
        ).reshape(-1, 2)

    def context(self, bbox: BBox | Sequence[float]) -> SymbolContext | None:
        """Context of a symbol box in page units, or ``None`` without a staff."""
        if not self.staves:
            return None
        cx, cy = _centre(bbox)

        def staff_distance(st: BBox) -> float:
            dy = 0.0 if st[1] <= cy <= st[3] else min(abs(cy - st[1]), abs(cy - st[3]))
            dx = 0.0 if st[0] <= cx <= st[2] else min(abs(cx - st[0]), abs(cx - st[2]))
            return math.hypot(dx, dy)

        staff = min(self.staves, key=staff_distance)
        staff_space = (staff[3] - staff[1]) / 4.0
        if staff_space <= 0:
            return None
        y_mid = 0.5 * (staff[1] + staff[3])
        y_staff = (cy - y_mid) / staff_space

        x_measure, barline = 0.5, NOTE_RANGE
        candidates = [
            m for m in self.measures if m[0] <= cx <= m[2] and m[1] <= staff[3] and m[3] >= staff[1]
        ]
        if candidates:
            m = min(candidates, key=lambda b: b[2] - b[0])
            width = max(1e-6, m[2] - m[0])
            x_measure = (cx - m[0]) / width
            barline = min(cx - m[0], m[2] - cx) / staff_space

        note_dx = note_dy = None
        if len(self.noteheads):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            inside = (
                (self.noteheads[:, 0] >= x1)
                & (self.noteheads[:, 0] <= x2)
                & (self.noteheads[:, 1] >= y1)
                & (self.noteheads[:, 1] <= y2)
            )
            others = self.noteheads[~inside]
            if len(others):
                d = np.hypot(others[:, 0] - cx, others[:, 1] - cy) / staff_space
                k = int(np.argmin(d))
                if d[k] <= NOTE_RANGE:
                    note_dx = (cx - others[k, 0]) / staff_space
                    note_dy = (cy - others[k, 1]) / staff_space

        return SymbolContext(
            y_staff=y_staff,
            x_measure=x_measure,
            barline=barline,
            note_dx=note_dx,
            note_dy=note_dy,
            width=(float(bbox[2]) - float(bbox[0])) / staff_space,
            height=(float(bbox[3]) - float(bbox[1])) / staff_space,
        )


def page_symbol_contexts(
    page: PageAnnotation, boxes: Sequence[BBox | Sequence[float]]
) -> list[SymbolContext | None]:
    """Contexts of symbol boxes given in page units."""
    geometry = PageGeometry(page)
    return [geometry.context(b) for b in boxes]


__all__ = [
    "CONTEXT_DIM",
    "CONTEXT_FEATURES",
    "NOTE_RANGE",
    "PageGeometry",
    "SymbolContext",
    "context_from_record",
    "context_vector",
    "page_symbol_contexts",
]
