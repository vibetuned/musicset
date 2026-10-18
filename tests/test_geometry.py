import math

import pytest

from musicset.annotate.geometry import (
    IDENTITY,
    multiply,
    parse_transform,
    transform_bbox,
    union,
)


def test_empty_transform_is_identity():
    assert parse_transform(None) == IDENTITY
    assert parse_transform("") == IDENTITY


def test_translate_then_scale_applies_right_to_left():
    m = parse_transform("translate(10, 20) scale(2)")
    # a point is scaled first, then translated
    assert transform_bbox(m, (1, 1, 2, 2)) == (12, 22, 14, 24)


def test_scale_with_flip_keeps_box_ordered():
    assert transform_bbox(parse_transform("scale(1,-1)"), (0, 1, 2, 3)) == (0, -3, 2, -1)


def test_rotate_about_center():
    m = parse_transform("rotate(90, 1, 1)")
    x1, y1, x2, y2 = transform_bbox(m, (0, 0, 2, 1))
    assert (x1, y1, x2, y2) == pytest.approx((1, 0, 2, 2))


def test_matrix_command_and_composition():
    m = parse_transform("matrix(1 0 0 1 5 6)")
    assert m == (1, 0, 0, 1, 5, 6)
    assert multiply(m, parse_transform("translate(1)")) == (1, 0, 0, 1, 6, 6)


def test_rotation_matrix_values():
    a, b, c, d, _, _ = parse_transform("rotate(30)")
    assert (a, b, c, d) == pytest.approx(
        (
            math.cos(math.pi / 6),
            math.sin(math.pi / 6),
            -math.sin(math.pi / 6),
            math.cos(math.pi / 6),
        )
    )


def test_union_treats_none_as_empty():
    assert union(None, None) is None
    assert union(None, (0, 0, 1, 1)) == (0, 0, 1, 1)
    assert union((0, 0, 1, 1), (2, -1, 3, 0.5)) == (0, -1, 3, 1)
