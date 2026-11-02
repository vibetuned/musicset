"""EMNIST (Cohen et al., 2017): NIST handwritten digits and letters as 28x28 images.

Real handwriting for the classes HOMUS lacks: digits ``1``-``5`` for the
fingerings, the lowercase letters ``p m f s z`` for the dynamics, which
:func:`compose_word` strings together into ``pp``, ``mf``, ``sfz``... the way
engraved dynamics are set, the letters of ``D.S.``, ``D.C.`` and ``Fine`` (the
periods, which EMNIST lacks, are drawn), and the signs :func:`compose_sign` builds
from characters: the coda (an ``O`` crossed by an ``X`` turned upright), the segno
(an ``S`` slashed by a ``1`` laid diagonally, with two dots) and the turn (an ``S``
mirrored and laid on its side; a plain ``S`` on its side is the inverted turn).

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
    "dalSegno": ("D.S.",),
    "daCapo": ("D.C.",),
    "fine": ("Fine",),
}
"""Symbol class -> the words written with EMNIST characters that mean it (``.`` is drawn)."""

EMNIST_SIGNS: dict[str, tuple[str, ...]] = {
    "coda": ("O0", "Xx"),
    "segno": ("S", "1lI"),
    "turn": ("S",),
}
"""Symbol class -> the characters of the glyphs its sign is built from (any one of each
string), put together by :func:`compose_sign`. Capital ``S`` only: writers make a
lowercase ``s`` wide, which flattens the segno and stands the turn upright."""

X_HEIGHT = 0.62
"""Height of an x-height lowercase letter (``a c e m n ...``) in cap heights."""
DESCENT = 0.3
"""How far a descender (``g j p q y``) reaches below the baseline, in cap heights."""
_X_HEIGHT_CHARS = frozenset("acemnorsuvwxz")
_DESCENDER_CHARS = frozenset("gjpqy")


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


def _char_box(ch: str | None) -> tuple[float, float]:
    """``(height, drop)`` of a written character in cap heights: how tall it is and how far
    its bottom sits below the baseline. EMNIST scales every character to the same box, so
    a lowercase ``m`` would otherwise come out as tall as a ``D``."""
    if ch is None or not ch.islower():
        return 1.0, 0.0
    if ch in _DESCENDER_CHARS:
        return X_HEIGHT + DESCENT, DESCENT
    if ch in _X_HEIGHT_CHARS:
        return X_HEIGHT, 0.0
    if ch == "i":
        return 0.8, 0.0
    return 1.0, 0.0  # ascenders: b d f h k l t


def _dot(cap: int, rng: np.random.Generator) -> np.ndarray:
    """A written period: a filled, slightly uneven blob ``0.12``-``0.2`` cap heights wide."""
    h = max(2, round(cap * rng.uniform(0.12, 0.2)))
    w = max(2, round(h * rng.uniform(0.8, 1.25)))
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((yy + 0.5 - h / 2) / (h / 2)) ** 2 + ((xx + 0.5 - w / 2) / (w / 2)) ** 2 <= 1.0
    return np.where(inside, 0, 255).astype(np.uint8)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(image, mode="L").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _ink_crop(image: np.ndarray) -> np.ndarray | None:
    box = ink_bbox(image)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def compose_word(
    glyphs: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    chars: str | None = None,
    gap: tuple[float, float] = (-0.05, 0.2),
    jitter: float = 0.12,
) -> np.ndarray:
    """Black-on-white glyph images written side by side as one word.

    Each glyph is cropped to its ink and placed on a common baseline with a gap of
    ``gap`` cap heights (negative overlaps slightly, as joined handwriting does) and a
    vertical offset of up to ``jitter`` cap heights. ``chars`` names the character each
    glyph writes: lowercase letters then keep their x-height and descenders, and a
    ``.`` (which EMNIST lacks) is drawn as a dot, its glyph entry ignored. Without it
    every glyph is scaled to the full cap height.
    """
    if chars is not None and len(chars) != len(glyphs):
        raise ValueError(f"{len(glyphs)} glyphs for {len(chars)} characters {chars!r}")
    items: list[tuple[np.ndarray | None, str | None]] = []
    for i, glyph in enumerate(glyphs):
        ch = chars[i] if chars is not None else None
        if ch == ".":
            items.append((None, ch))
            continue
        crop = _ink_crop(glyph)
        if crop is not None:
            items.append((crop, ch))
    letters = [c for c, _ in items if c is not None]
    if not letters:
        raise ValueError("compose_word needs at least one inked glyph")
    cap = max(c.shape[0] for c in letters)

    pieces: list[tuple[np.ndarray, int]] = []  # (image, bottom below the baseline in px)
    for crop, ch in items:
        if crop is None:
            pieces.append((_dot(cap, rng), 0))
            continue
        h_frac, drop = _char_box(ch)
        height = max(2, round(cap * h_frac))
        width = max(1, round(crop.shape[1] * height / crop.shape[0]))
        pieces.append((_resize(crop, width, height), round(drop * cap)))

    offsets = [round(rng.uniform(-jitter, jitter) * cap) for _ in pieces]
    gaps = [round(rng.uniform(*gap) * cap) for _ in pieces[1:]]
    total_w = max(1, sum(img.shape[1] for img, _ in pieces) + sum(gaps))
    pad = int(jitter * cap) + 1
    baseline = pad + cap
    canvas = np.full((cap + round(DESCENT * cap) + 2 * pad, total_w), 255, dtype=np.uint8)
    x = 0
    for i, (img, drop) in enumerate(pieces):
        x = max(0, min(x, total_w - img.shape[1]))
        bottom = baseline + drop + offsets[i]
        y = max(0, bottom - img.shape[0])
        region = canvas[y : y + img.shape[0], x : x + img.shape[1]]
        np.minimum(region, img[: region.shape[0], : region.shape[1]], out=region)
        x += img.shape[1] + (gaps[i] if i < len(gaps) else 0)
    out = _ink_crop(canvas)
    assert out is not None
    return out


def compose_cross_circle(
    circle: np.ndarray,
    cross: np.ndarray,
    rng: np.random.Generator,
    *,
    twist: float = 45.0,
    twist_jitter: float = 10.0,
    overshoot: tuple[float, float] = (1.25, 1.65),
) -> np.ndarray:
    """A handwritten coda from two characters: ``circle`` (an ``O`` or ``0``) made roughly
    round, crossed by ``cross`` (an ``X``) turned by ``twist`` degrees, give or take
    ``twist_jitter``, so its strokes stand upright and level and overshoot the circle by
    ``overshoot`` times its size. Black on white, cropped to the ink."""
    from PIL import Image

    ring = _ink_crop(circle)
    arms = _ink_crop(cross)
    if ring is None or arms is None:
        raise ValueError("compose_cross_circle needs two inked glyphs")
    size = 48
    aspect = float(rng.uniform(0.85, 1.15))
    ring = _resize(ring, max(2, round(size * aspect)), size)
    angle = twist + float(rng.uniform(-twist_jitter, twist_jitter))
    turned = Image.fromarray(255 - arms, mode="L").rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0
    )
    arms_up = _ink_crop(255 - np.asarray(turned, dtype=np.uint8))
    if arms_up is None:
        raise ValueError("the cross lost its ink when turned")
    span = size * float(rng.uniform(*overshoot))
    scale = span / max(arms_up.shape)
    arms_up = _resize(
        arms_up, max(2, round(arms_up.shape[1] * scale)), max(2, round(arms_up.shape[0] * scale))
    )
    side = max(*ring.shape, *arms_up.shape) + 2 * round(0.1 * size) + 2
    canvas = np.full((side, side), 255, dtype=np.uint8)
    for img, wobble in ((ring, 0.0), (arms_up, 0.06)):
        dx, dy = (round(float(rng.uniform(-wobble, wobble)) * size) for _ in range(2))
        y = max(0, min(side - img.shape[0], (side - img.shape[0]) // 2 + dy))
        x = max(0, min(side - img.shape[1], (side - img.shape[1]) // 2 + dx))
        region = canvas[y : y + img.shape[0], x : x + img.shape[1]]
        np.minimum(region, img, out=region)
    out = _ink_crop(canvas)
    assert out is not None
    return out


def compose_slashed_s(
    letter: np.ndarray,
    stroke: np.ndarray,
    rng: np.random.Generator,
    *,
    tilt: float = -38.0,
    tilt_jitter: float = 8.0,
) -> np.ndarray:
    """A handwritten segno: ``letter`` (an ``S``) slashed from lower left to upper right by
    ``stroke`` (a ``1``, ``l`` or ``I``) turned ``tilt`` degrees, give or take
    ``tilt_jitter``, with a drawn dot right of the upper half and one left of the lower
    half, where engraved segni put them. Black on white, cropped to the ink."""
    from PIL import Image

    s_ink = _ink_crop(letter)
    bar = _ink_crop(stroke)
    if s_ink is None or bar is None:
        raise ValueError("compose_slashed_s needs two inked glyphs")
    size = 48
    s_ink = _resize(s_ink, max(2, round(s_ink.shape[1] * size / s_ink.shape[0])), size)
    angle = tilt + float(rng.uniform(-tilt_jitter, tilt_jitter))
    turned = Image.fromarray(255 - bar, mode="L").rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0
    )
    slash = _ink_crop(255 - np.asarray(turned, dtype=np.uint8))
    if slash is None:
        raise ValueError("the slash lost its ink when turned")
    reach = size * float(rng.uniform(1.0, 1.25))
    scale = reach / slash.shape[0]
    slash = _resize(
        slash, max(2, round(slash.shape[1] * scale)), max(2, round(slash.shape[0] * scale))
    )
    side = max(*s_ink.shape, *slash.shape) + size
    canvas = np.full((side, side), 255, dtype=np.uint8)
    mid = side // 2
    for img, wobble in ((s_ink, 0.0), (slash, 0.05)):
        dx, dy = (round(float(rng.uniform(-wobble, wobble)) * size) for _ in range(2))
        y = max(0, min(side - img.shape[0], mid - img.shape[0] // 2 + dy))
        x = max(0, min(side - img.shape[1], mid - img.shape[1] // 2 + dx))
        region = canvas[y : y + img.shape[0], x : x + img.shape[1]]
        np.minimum(region, img, out=region)
    half_w = 0.5 * s_ink.shape[1]
    for side_sign in (1.0, -1.0):  # right dot high, left dot low
        dot = _dot(size, rng)
        cx = mid + side_sign * (half_w + size * float(rng.uniform(0.0, 0.12)))
        cy = mid - side_sign * size * float(rng.uniform(0.08, 0.2))
        y = max(0, min(side - dot.shape[0], round(cy - dot.shape[0] / 2)))
        x = max(0, min(side - dot.shape[1], round(cx - dot.shape[1] / 2)))
        region = canvas[y : y + dot.shape[0], x : x + dot.shape[1]]
        np.minimum(region, dot, out=region)
    out = _ink_crop(canvas)
    assert out is not None
    return out


def compose_turned_s(
    letter: np.ndarray,
    rng: np.random.Generator,
    *,
    turn: float = 90.0,
    turn_jitter: float = 8.0,
    squash: tuple[float, float] = (0.75, 1.0),
) -> np.ndarray:
    """A handwritten turn: ``letter`` (an ``S``) mirrored, then laid on its side by
    ``turn`` degrees (give or take ``turn_jitter``) and flattened by ``squash``. Without
    the mirror an ``S`` on its side is the inverted turn, whichever way it is turned
    (an ``S`` looks the same upside down). Black on white, cropped to the ink."""
    from PIL import Image

    ink = _ink_crop(letter)
    if ink is None:
        raise ValueError("compose_turned_s needs an inked glyph")
    mirrored = np.ascontiguousarray(ink[:, ::-1])
    angle = turn + float(rng.uniform(-turn_jitter, turn_jitter))
    turned = Image.fromarray(255 - mirrored, mode="L").rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0
    )
    out = _ink_crop(255 - np.asarray(turned, dtype=np.uint8))
    if out is None:
        raise ValueError("the turn lost its ink when turned")
    height = max(2, round(out.shape[0] * float(rng.uniform(*squash))))
    return _resize(out, out.shape[1], height)


def compose_sign(cls: str, glyphs: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """The ``EMNIST_SIGNS`` sign ``cls`` from its glyphs."""
    if cls == "coda":
        return compose_cross_circle(glyphs[0], glyphs[1], rng)
    if cls == "segno":
        return compose_slashed_s(glyphs[0], glyphs[1], rng)
    if cls == "turn":
        return compose_turned_s(glyphs[0], rng)
    raise ValueError(f"no sign composition for {cls!r}")


__all__ = [
    "BYCLASS_CHARS",
    "DESCENT",
    "EMNIST_SIGNS",
    "EMNIST_SPLITS",
    "EMNIST_SYMBOLS",
    "X_HEIGHT",
    "compose_cross_circle",
    "compose_sign",
    "compose_slashed_s",
    "compose_turned_s",
    "compose_word",
    "ink_bbox",
    "load_emnist",
    "read_idx",
]
