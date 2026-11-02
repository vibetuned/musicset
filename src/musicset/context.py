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

from musicset.annotate.geometry import BBox, union
from musicset.annotate.schema import PageAnnotation

CONTEXT_FEATURES: tuple[str, ...] = (
    "valid",
    "in_staff",
    "above_staff",
    "below_staff",
    "y_staff",
    "x_measure",
    "barline",
    "has_note",
    "note_above",
    "note_below",
    "note_left",
    "note_right",
    "note_dx",
    "note_dy",
    "width",
    "height",
)
"""Feature order of :func:`context_vector`. Flags are 0 or 1, the rest roughly in [-1, 1].

Where the symbol sits on the staff:

* ``valid``: 1 when the context is known, else the whole vector is 0;
* ``in_staff`` / ``above_staff`` / ``below_staff``: one-hot of the symbol centre's
  vertical zone, between the outer staff lines (where rests and notes live) or
  outside them (where fingerings, dynamics and most articulations live);
* ``y_staff``: the centre in staff spaces from the middle line, down positive, / 6;
* ``x_measure``: position between the measure's barlines, left -1 to right 1;
* ``barline``: distance to the nearest barline in staff spaces / 8, clipped.

Where it sits relative to the nearest note, never the symbol's own. A note is its
body: the bounding box of its notehead, stem and flag, or of every notehead plus the
stem and flag of its chord, without accidentals, dots or articulations (an editor sees
the note that way before the mark is added). The body's centre is the reference, so
the offset no longer depends on whether the stem points up or down at the notehead:

* ``has_note``: 1 when a body lies within ``NOTE_RANGE`` staff spaces;
* ``note_above`` / ``note_below``: the symbol centre is above the body's top or below
  its bottom, within ``NOTE_MARGIN`` of its sides (staccato, tenuto, accent, fermata,
  fingering);
* ``note_left`` / ``note_right``: the centre is left of the body's left edge or right
  of its right edge, within ``NOTE_MARGIN`` of its top and bottom (accidental on the
  left, augmentation dot on the right);
* ``note_dx``, ``note_dy``: the exact offset, symbol centre minus body centre, / 8;

and its size: ``width``, ``height`` in staff spaces / 8, clipped.
"""
CONTEXT_DIM = len(CONTEXT_FEATURES)
NOTE_RANGE = 8.0
"""Staff spaces within which a note's body counts as the symbol's neighbour."""
STAFF_HALF_HEIGHT = 2.0
"""Staff spaces from the middle line to the outer lines: the ``in_staff`` zone."""
NOTE_MARGIN = 0.5
"""Staff spaces a symbol may stick out past a body's sides and still count as above or
below it (or past its top and bottom and still count as left or right of it)."""
NOTE_BODY_PARTS = frozenset(("notehead", "stem", "flag"))
"""Element kinds that make up a note's body."""
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
    """Symbol centre minus the nearest note body's centre; ``None`` without one."""
    width: float
    height: float
    note_w: float | None = None
    note_h: float | None = None
    """Width and height of that body (``None`` or 0: a point, e.g. an older record)."""


def context_vector(ctx: SymbolContext | None) -> np.ndarray:
    """``CONTEXT_DIM`` float32 features for ``ctx``; all zero when ``None``."""
    if ctx is None:
        return np.zeros(CONTEXT_DIM, dtype=np.float32)
    has_note = ctx.note_dx is not None and ctx.note_dy is not None
    dx = ctx.note_dx if ctx.note_dx is not None else 0.0
    dy = ctx.note_dy if ctx.note_dy is not None else 0.0
    hw = 0.5 * (ctx.note_w or 0.0)
    hh = 0.5 * (ctx.note_h or 0.0)
    over = has_note and abs(dx) <= hw + NOTE_MARGIN
    beside = has_note and abs(dy) <= hh + NOTE_MARGIN
    return np.array(
        [
            1.0,
            1.0 if abs(ctx.y_staff) <= STAFF_HALF_HEIGHT else 0.0,
            1.0 if ctx.y_staff < -STAFF_HALF_HEIGHT else 0.0,
            1.0 if ctx.y_staff > STAFF_HALF_HEIGHT else 0.0,
            max(-2.0, min(2.0, ctx.y_staff / 6.0)),
            max(-1.0, min(1.0, ctx.x_measure * 2.0 - 1.0)),
            max(0.0, min(1.0, ctx.barline / 8.0)),
            1.0 if has_note else 0.0,
            1.0 if over and dy < -hh else 0.0,
            1.0 if over and dy > hh else 0.0,
            1.0 if beside and dx < -hw else 0.0,
            1.0 if beside and dx > hw else 0.0,
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
    ``note_dx`` / ``note_dy`` (staff spaces from the nearest note body's centre,
    ``null`` when no note is near) with that body's ``note_w`` / ``note_h``, and
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
    near = dx is not None and dy is not None
    return SymbolContext(
        y_staff=float(raw["y_staff"]),
        x_measure=float(raw.get("x_measure", 0.5)),
        barline=float(raw.get("barline", NOTE_RANGE)),
        note_dx=None if dx is None else float(dx),
        note_dy=None if dy is None else float(dy),
        width=width,
        height=height,
        note_w=float(raw.get("note_w") or 0.0) if near else None,
        note_h=float(raw.get("note_h") or 0.0) if near else None,
    )


def _centre(b: BBox | Sequence[float]) -> tuple[float, float]:
    return 0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))


def note_bodies(page: PageAnnotation) -> np.ndarray:
    """``(N, 4)`` page-unit boxes of the page's note bodies: noteheads, stem and flag of
    each note, or of each chord as a whole. Pages without note elements fall back to
    their notehead glyphs."""
    boxes: dict[int, BBox] = {}
    for el in page.elements:
        if el.bbox is None or not el.classes or el.classes[0] not in NOTE_BODY_PARTS:
            continue
        owner: int | None = None
        cur = el.parent
        while cur is not None:
            anc = page.elements[cur]
            kind = anc.classes[0] if anc.classes else ""
            if kind == "chord":
                owner = cur
                break
            if kind == "note" and owner is None:
                owner = cur  # keep climbing: the note may belong to a chord
            elif kind in ("layer", "staff", "measure"):
                break
            cur = anc.parent
        merged = union(boxes.get(owner), el.bbox) if owner is not None else None
        if owner is not None and merged is not None:
            boxes[owner] = merged
    if boxes:
        return np.array(list(boxes.values()), dtype=np.float64).reshape(-1, 4)
    return np.array(
        [g.bbox for g in page.glyphs if g.code in NOTEHEAD_CODES and g.bbox], dtype=np.float64
    ).reshape(-1, 4)


class PageGeometry:
    """Staves (five-line boxes), measures and note bodies of a page, for context lookups."""

    def __init__(self, page: PageAnnotation) -> None:
        self.staves: list[BBox] = [
            e.bbox for e in page.elements if e.classes and e.classes[0] == "system-staff" and e.bbox
        ]
        self.measures: list[BBox] = [
            e.bbox for e in page.elements if e.classes and e.classes[0] == "measure" and e.bbox
        ]
        self.notes: np.ndarray = note_bodies(page)

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

        note_dx = note_dy = note_w = note_h = None
        if len(self.notes):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            ncx = 0.5 * (self.notes[:, 0] + self.notes[:, 2])
            ncy = 0.5 * (self.notes[:, 1] + self.notes[:, 3])
            # a body centred inside the symbol's box is the symbol itself (a note crop)
            own = (ncx >= x1) & (ncx <= x2) & (ncy >= y1) & (ncy <= y2)
            others, ocx, ocy = self.notes[~own], ncx[~own], ncy[~own]
            if len(others):
                gap_x = np.maximum(np.maximum(others[:, 0] - cx, cx - others[:, 2]), 0.0)
                gap_y = np.maximum(np.maximum(others[:, 1] - cy, cy - others[:, 3]), 0.0)
                d = np.hypot(gap_x, gap_y) / staff_space
                # nearest body edge first, nearest centre among bodies the symbol touches
                k = int(np.argmin(d + 1e-3 * np.hypot(ocx - cx, ocy - cy) / staff_space))
                if d[k] <= NOTE_RANGE:
                    note_dx = (cx - ocx[k]) / staff_space
                    note_dy = (cy - ocy[k]) / staff_space
                    note_w = (others[k, 2] - others[k, 0]) / staff_space
                    note_h = (others[k, 3] - others[k, 1]) / staff_space

        return SymbolContext(
            y_staff=y_staff,
            x_measure=x_measure,
            barline=barline,
            note_dx=note_dx,
            note_dy=note_dy,
            width=(float(bbox[2]) - float(bbox[0])) / staff_space,
            height=(float(bbox[3]) - float(bbox[1])) / staff_space,
            note_w=note_w,
            note_h=note_h,
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
    "NOTE_BODY_PARTS",
    "NOTE_MARGIN",
    "NOTE_RANGE",
    "STAFF_HALF_HEIGHT",
    "PageGeometry",
    "SymbolContext",
    "context_from_record",
    "context_vector",
    "note_bodies",
    "page_symbol_contexts",
]
