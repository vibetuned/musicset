"""EMNIST (Cohen et al., 2017): NIST handwritten digits and letters as 28x28 images.

Real handwriting for the classes HOMUS lacks: digits ``1``-``5`` for the
fingerings and the lowercase letters ``p m f s z`` for the dynamics, which
:func:`compose_word` strings together into ``pp``, ``mf``, ``sfz``... the way
engraved dynamics are set.

Reads the ``byclass`` split (62 classes, the only one that keeps lowercase
letters apart from uppercase) from NIST's ``gzip.zip``:
``emnist-byclass-{train,test}-{images-idx3,labels-idx1}-ubyte.gz`` and
``emnist-byclass-mapping.txt`` in one directory. EMNIST stores images
transposed and white-on-black; :func:`load_emnist` returns them upright and
black-on-white like every other musicset symbol. The archive carries no writer
ids, so EMNIST's own train/test files are the only writer separation available.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

EMNIST_SPLITS = ("train", "test")
BYCLASS_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
"""Label index -> character of the ``byclass`` split (the mapping file says the same)."""

EMNIST_SYMBOLS: dict[str, tuple[str, ...]] = {
    "fing1": ("1",),
    "fing2": ("2",),
    "fing3": ("3",),
    "fing4": ("4",),
    "fing5": ("5",),
    "dynamicPiano": ("p",),
    "dynamicMezzo": ("m",),
    "dynamicForte": ("f",),
    "dynamicSforzando": ("s", "sf"),
    "dynamicZ": ("z",),
    "dynamicPP": ("pp",),
    "dynamicMP": ("mp",),
    "dynamicMF": ("mf",),
    "dynamicFF": ("ff",),
    "dynamicFFF": ("fff",),
    "dynamicForzando": ("fz",),
    "dynamicSforzato": ("sfz",),
}
"""Symbol class -> the words written with EMNIST characters that mean it."""


def read_idx(path: Path) -> np.ndarray:
    """An IDX file (gzip or plain) as a numpy array."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = f.read(4)
        if magic[:2] != b"\x00\x00":
            raise ValueError(f"{path}: not an IDX file")
        dtype = {
            8: np.uint8,
            9: np.int8,
            11: np.int16,
            12: np.int32,
            13: np.float32,
            14: np.float64,
        }[magic[2]]
        ndim = magic[3]
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.dtype(dtype).newbyteorder(">"))
    return data.reshape(shape)


def _file(root: Path, split: str, kind: str) -> Path:
    for name in (
        f"emnist-byclass-{split}-{kind}-ubyte.gz",
        f"emnist-byclass-{split}-{kind}-ubyte",
    ):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"{root}: no emnist-byclass-{split}-{kind}-ubyte[.gz]")


def _mapping(root: Path) -> dict[int, str]:
    mapping = root / "emnist-byclass-mapping.txt"
    if not mapping.exists():
        return dict(enumerate(BYCLASS_CHARS))
    out: dict[int, str] = {}
    for line in mapping.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[int(parts[0])] = chr(int(parts[1]))
    return out


def load_emnist(
    root: Path, split: str = "train", chars: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(images, labels)`` of the ``byclass`` ``split``: upright black-on-white uint8
    ``(N, 28, 28)`` images and their characters as a ``<U1`` array; ``chars`` keeps
    only those characters."""
    if split not in EMNIST_SPLITS:
        raise ValueError(f"unknown EMNIST split {split!r}: {EMNIST_SPLITS}")
    images = read_idx(_file(root, split, "images-idx3"))
    labels = read_idx(_file(root, split, "labels-idx1"))
    if images.ndim != 3 or len(images) != len(labels):
        raise ValueError(f"{root}: images {images.shape} do not match labels {labels.shape}")
    mapping = _mapping(root)
    chars_arr = np.array([mapping.get(int(label), "?") for label in labels], dtype="<U1")
    # EMNIST stores each image transposed and white-on-black
    upright = 255 - np.transpose(images, (0, 2, 1))
    if chars is not None:
        keep = np.isin(chars_arr, list(chars))
        upright, chars_arr = upright[keep], chars_arr[keep]
    return np.ascontiguousarray(upright), chars_arr


def ink_bbox(image: np.ndarray, threshold: int = 128) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` of the ink (``< threshold``) in a black-on-white image."""
    ink = image < threshold
    if not ink.any():
        return None
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def compose_word(
    glyphs: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    gap: tuple[float, float] = (-0.05, 0.2),
    jitter: float = 0.12,
) -> np.ndarray:
    """Black-on-white glyph images written side by side as one word.

    Each glyph is cropped to its ink, scaled to a common height, and placed with
    a gap of ``gap`` times that height (negative overlaps slightly, as joined
    handwriting does) and a vertical offset of up to ``jitter`` heights.
    """
    crops: list[np.ndarray] = []
    for glyph in glyphs:
        box = ink_bbox(glyph)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        crops.append(glyph[y0:y1, x0:x1])
    if not crops:
        raise ValueError("compose_word needs at least one inked glyph")
    height = max(c.shape[0] for c in crops)
    from PIL import Image

    scaled: list[np.ndarray] = []
    for c in crops:
        scale = height / c.shape[0]
        width = max(1, round(c.shape[1] * scale))
        scaled.append(
            np.asarray(
                Image.fromarray(c, mode="L").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        )
    offsets = [round(rng.uniform(-jitter, jitter) * height) for _ in scaled]
    gaps = [round(rng.uniform(*gap) * height) for _ in scaled[1:]]
    total_w = sum(s.shape[1] for s in scaled) + sum(gaps)
    pad = int(jitter * height) + 1
    canvas = np.full((height + 2 * pad, max(1, total_w)), 255, dtype=np.uint8)
    x = 0
    for i, s in enumerate(scaled):
        y = pad + offsets[i]
        region = canvas[y : y + height, x : x + s.shape[1]]
        np.minimum(region, s, out=region)  # ink over ink stays ink
        x += s.shape[1] + (gaps[i] if i < len(gaps) else 0)
    box = ink_bbox(canvas)
    assert box is not None
    x0, y0, x1, y1 = box
    return canvas[y0:y1, x0:x1]


__all__ = [
    "BYCLASS_CHARS",
    "EMNIST_SPLITS",
    "EMNIST_SYMBOLS",
    "compose_word",
    "ink_bbox",
    "load_emnist",
    "read_idx",
]
