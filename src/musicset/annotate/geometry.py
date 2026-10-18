"""Affine transforms and bounding boxes in SVG user space.

The arithmetic deliberately mirrors the order of operations of the original
kakigori extractor (matrices composed root-first, one command at a time) so
that boxes stay bit-identical to datasets built with it.
"""

import math
import re

type Matrix = tuple[float, float, float, float, float, float]
type BBox = tuple[float, float, float, float]
"""Axis-aligned box as (x1, y1, x2, y2)."""

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

_COMMAND_RE = re.compile(r"(translate|scale|rotate|matrix)\s*\(([^)]+)\)")
_ARGS_RE = re.compile(r"[, ]+")


def multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """Compose two SVG matrices (a, b, c, d, e, f): the result applies m2 first."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def parse_transform(transform: str | None) -> Matrix:
    """Parse an SVG ``transform`` attribute into a single matrix."""
    matrix = IDENTITY
    if not transform:
        return matrix

    for cmd, args_str in _COMMAND_RE.findall(transform):
        args = [float(val) for val in _ARGS_RE.split(args_str.strip()) if val]
        if not args:
            continue

        if cmd == "translate":
            ty = args[1] if len(args) > 1 else 0.0
            cmd_matrix: Matrix = (1.0, 0.0, 0.0, 1.0, args[0], ty)
        elif cmd == "scale":
            sy = args[1] if len(args) > 1 else args[0]
            cmd_matrix = (args[0], 0.0, 0.0, sy, 0.0, 0.0)
        elif cmd == "rotate":
            cx = args[1] if len(args) > 1 else 0.0
            cy = args[2] if len(args) > 2 else 0.0
            rad = math.radians(args[0])
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rot: Matrix = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if cx != 0.0 or cy != 0.0:
                to_origin: Matrix = (1.0, 0.0, 0.0, 1.0, cx, cy)
                back: Matrix = (1.0, 0.0, 0.0, 1.0, -cx, -cy)
                cmd_matrix = multiply(multiply(to_origin, rot), back)
            else:
                cmd_matrix = rot
        else:  # matrix
            if len(args) < 6:
                continue
            cmd_matrix = (args[0], args[1], args[2], args[3], args[4], args[5])

        matrix = multiply(matrix, cmd_matrix)

    return matrix


def transform_bbox(matrix: Matrix, box: BBox) -> BBox:
    """Axis-aligned bounds of ``box`` after applying ``matrix``."""
    a, b, c, d, e, f = matrix
    x1, y1, x2, y2 = box
    corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    xs = [a * x + c * y + e for x, y in corners]
    ys = [b * x + d * y + f for x, y in corners]
    return min(xs), min(ys), max(xs), max(ys)


def pad(box: BBox, amount: float) -> BBox:
    """Grow a box by ``amount`` on every side (used for half stroke widths)."""
    x1, y1, x2, y2 = box
    return x1 - amount, y1 - amount, x2 + amount, y2 + amount


def union(a: BBox | None, b: BBox | None) -> BBox | None:
    """Smallest box containing both; ``None`` is the empty box."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
