"""Pen strokes contributed from a score editor (Battuta's "contribute to dataset").

One sample per line (JSON Lines, ``*.jsonl``) or a JSON list (``*.json``)::

    {"label": "noteQuarter", "writer": "7c1e…", "staff_space": 12.5,
     "strokes": [[[x, y, t_ms, pressure], ...], ...],
     "context": {...}, "device": "apple-pencil", "app": "battuta/0.4.0",
     "time": "2026-11-01T12:00:00Z"}

* ``label``: a ``musicset`` symbol class (:data:`musicset.symbols.SYMBOL_BY_NAME`);
  what the contributor meant, even when another class looks the same;
* ``writer``: opaque contributor id (any string), only used to keep one
  person's drawings in one split;
* ``staff_space``: pixels between two staff lines in the strokes' coordinates,
  which fixes the drawing's natural size (:func:`stroke_fraction_from_staff_space`);
* ``strokes``: one list of points per pen-down; a point is ``[x, y]``,
  ``[x, y, t_ms]`` or ``[x, y, t_ms, pressure]``, extra values are kept out of
  the way for later;
* ``context`` (optional): where in the score the mark was made, the raw facts of
  :func:`musicset.context.context_from_record`, all in staff spaces: ``y_staff``
  (centre from the middle line, down is positive), ``x_measure`` (0 at the left
  barline, 1 at the right), ``barline`` (distance to the nearest one),
  ``note_dx`` / ``note_dy`` (centre minus the nearest existing notehead, ``null``
  when none is near) and optionally ``width`` / ``height``. It becomes the
  conditioning vector; without it the sample trains with "context unknown".

:func:`load_stroke_contributions` turns files or directories of them into
:class:`~musicset.homus.StrokeSample` records that ``musicset symbols --strokes``
mixes with HOMUS under the ``hand`` source.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from musicset.context import context_from_record
from musicset.homus import Stroke, StrokeSample, stroke_fraction_from_staff_space


def parse_contribution(
    record: dict[str, Any], *, source: str = "", classes: set[str] | None = None
) -> StrokeSample | None:
    """One sample from a contribution record; ``None`` for an unknown label or no ink."""
    label = str(record.get("label", ""))
    if classes is not None and label not in classes:
        return None
    writer = str(record.get("writer", "")).strip()
    if not label or not writer:
        raise ValueError(f"{source}: a contribution needs 'label' and 'writer'")
    staff_space = float(record.get("staff_space", 0.0))
    strokes: list[Stroke] = []
    for raw in record.get("strokes", []):
        points = tuple((float(p[0]), float(p[1])) for p in raw if len(p) >= 2)
        if points:
            strokes.append(points)
    if not strokes:
        return None
    xs = [p[0] for stroke in strokes for p in stroke]
    ys = [p[1] for stroke in strokes for p in stroke]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return StrokeSample(
        writer=0,
        label=label,
        strokes=tuple(strokes),
        source=f"{source}:{writer}",
        fraction=stroke_fraction_from_staff_space(strokes, staff_space),
        context=context_from_record(record, bbox, staff_space),
    )


def _iter_records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    files = (
        sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl"))
        if path.is_dir()
        else [path]
    )
    for file in files:
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".jsonl":
            for n, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    yield f"{file}:{n}", json.loads(line)
        else:
            data = json.loads(text)
            records = data.get("samples", []) if isinstance(data, dict) else data
            for n, record in enumerate(records, 1):
                yield f"{file}:{n}", record


def load_stroke_contributions(
    paths: Path | Sequence[Path], *, classes: set[str] | None = None
) -> list[StrokeSample]:
    """Every contributed sample under ``paths`` (files or directories).

    ``writer`` ids are opaque strings in the files; here they become integers
    starting at 1000, in order of first appearance, so they never collide with
    HOMUS writers 1-100. The mapping is deterministic for a fixed set of files.
    """
    roots = [paths] if isinstance(paths, Path) else list(paths)
    writer_ids: dict[str, int] = {}
    samples: list[StrokeSample] = []
    for root in roots:
        for source, record in _iter_records(root):
            sample = parse_contribution(record, source=source, classes=classes)
            if sample is None:
                continue
            key = str(record["writer"]).strip()
            writer = writer_ids.setdefault(key, 1000 + len(writer_ids))
            samples.append(
                StrokeSample(
                    writer,
                    sample.label,
                    sample.strokes,
                    sample.source,
                    sample.fraction,
                    sample.context,
                )
            )
    return samples


__all__ = ["load_stroke_contributions", "parse_contribution"]
