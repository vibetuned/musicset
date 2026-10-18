import numpy as np
import pytest

from musicset.augment import PRESETS, Augmenter
from musicset.augment.ops import (
    REFERENCE_HEIGHT,
    Binarize,
    InkSpread,
    Perspective,
    Rotate,
    Sample,
    Warp,
)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _ink_box(image: np.ndarray) -> tuple[float, ...]:
    ys, xs = np.nonzero(image < 0.5)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


@pytest.mark.parametrize("op", [Rotate(8.0), Perspective(0.08), Warp((0.03, 0.04))])
def test_geometric_steps_move_boxes_with_the_ink(op):
    image = np.ones((300, 400), np.float32)
    image[100:140, 150:250] = 0.0
    sample = Sample(image, np.array([[150, 100, 250, 140]], np.float32))
    op(sample, np.random.default_rng(0), 1.0)
    assert sample.log, "the step must have done something"
    assert _iou(_ink_box(sample.image), sample.boxes[0]) > 0.85


def test_ink_spread_thickens_strokes_and_grows_boxes():
    image = np.ones((REFERENCE_HEIGHT, 50), np.float32)
    image[1000, :] = 0.0  # a 1 px staff line
    sample = Sample(image, np.array([[0, 1000, 50, 1001]], np.float32))
    InkSpread((1, 1))(sample, np.random.default_rng(0), 1.0)
    assert (sample.image[:, 25] < 0.5).sum() == 3
    assert sample.boxes.tolist() == [[-1, 999, 51, 1002]]


def test_binarize_is_black_and_white():
    sample = Sample(np.linspace(0, 1, 10000, dtype=np.float32).reshape(100, 100), np.zeros((0, 4)))
    Binarize()(sample, np.random.default_rng(0), 1.0)
    assert set(np.unique(sample.image)) <= {0.0, 1.0}


@pytest.fixture(scope="module")
def page() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    image = np.full((600, 420), 255, np.uint8)
    boxes = []
    for y in range(80, 520, 90):  # staves of five lines with a few "noteheads"
        for k in range(5):
            image[y + 8 * k, 30:390] = 0
        for x in rng.integers(40, 370, 4):
            image[y + 10 : y + 18, x : x + 10] = 0
            boxes.append([x, y + 10, x + 10, y + 18])
    return image, np.array(boxes, np.float32)


@pytest.mark.parametrize("preset", [*PRESETS, "mixed"])
def test_presets_return_rgb_and_boxes_on_the_page(page, preset):
    image, boxes = page
    out = Augmenter(preset)(image, boxes, rng=3)
    assert out.image.shape == (600, 420, 3) and out.image.dtype == np.uint8
    assert out.boxes.shape == boxes.shape and out.keep.shape == (len(boxes),)
    assert (out.boxes[:, [0, 2]] >= 0).all() and (out.boxes[:, [0, 2]] <= 420).all()
    assert (out.boxes[:, [1, 3]] >= 0).all() and (out.boxes[:, [1, 3]] <= 600).all()


def test_seeded_augmentation_is_reproducible(page):
    image, boxes = page
    a = Augmenter("mixed")(image, boxes, rng=11)
    b = Augmenter("mixed")(image, boxes, rng=11)
    c = Augmenter("mixed")(image, boxes, rng=12)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.boxes, b.boxes)
    assert a.log == b.log and (a.log != c.log or not np.array_equal(a.image, c.image))


def test_clean_preset_and_unknown_preset(page):
    image, boxes = page
    out = Augmenter("clean")(image, boxes, rng=0)
    assert np.array_equal(out.image[..., 0], image) and np.array_equal(out.boxes, boxes)
    with pytest.raises(ValueError, match="unknown preset"):
        Augmenter("vintage")
