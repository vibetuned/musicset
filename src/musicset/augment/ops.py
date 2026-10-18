"""Degradation steps that turn a clean engraving into a scan, boxes included.

The image is float32 with 1.0 = paper: grey (H, W) through the ink and
scanner stages, RGB (H, W, 3) once :class:`Paper` has tinted it. Steps that
move pixels move the boxes (x1, y1, x2, y2 in pixels) with them; steps that
grow ink grow the boxes.

``strength`` scales each step's magnitude (1 = the preset's ranges, 0 = none)
so a viewer slider can sweep from clean to heavy.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class Sample:
    image: np.ndarray
    boxes: np.ndarray
    log: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


def _channels(img: np.ndarray, fn) -> np.ndarray:
    """Apply a 2-D function to a grey image or to each channel of an RGB one."""
    if img.ndim == 2:
        return fn(img)
    return np.stack([fn(img[..., c]) for c in range(img.shape[2])], axis=-1)


def _pil_transform(img: np.ndarray, method, data) -> np.ndarray:
    h, w = img.shape[:2]

    def one(channel: np.ndarray) -> np.ndarray:
        out = Image.fromarray(np.ascontiguousarray(channel), mode="F").transform(
            (w, h), method, data, Image.Resampling.BILINEAR, fillcolor=1.0
        )
        return np.asarray(out, dtype=np.float32)

    return _channels(img, one)


def _smooth_noise(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Noise in [0, 1] with features about ``scale`` pixels across."""
    h, w = shape
    small = rng.random((max(2, round(h / scale)), max(2, round(w / scale)))).astype(np.float32)
    big = np.asarray(Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC))
    lo, hi = float(big.min()), float(big.max())
    return (big - lo) / (hi - lo + 1e-6)


REFERENCE_HEIGHT = 2970
"""Page height (px) pixel-sized parameters are given at: A4 at Verovio's default size."""


def _res(s: Sample) -> float:
    """Pixel parameters scale with the page's resolution."""
    return s.image.shape[0] / REFERENCE_HEIGHT


def _corners(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    return np.stack([np.stack([x1, y1], -1), np.stack([x2, y1], -1),
                     np.stack([x1, y2], -1), np.stack([x2, y2], -1)], axis=1)  # fmt: skip


def _bounds(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points.min(axis=1), points.max(axis=1)], axis=1).astype(np.float32)


def _project(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    ones = np.ones((*points.shape[:-1], 1), dtype=np.float64)
    p = np.concatenate([points, ones], axis=-1) @ h.T
    return p[..., :2] / p[..., 2:3]


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 matrix mapping four src points onto four dst points."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    a = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return np.append(a, 1.0).reshape(3, 3)


def _warp_projective(sample: Sample, h: np.ndarray) -> None:
    """Apply a forward homography to the image (PIL wants the inverse) and boxes."""
    inv = np.linalg.inv(h)
    inv /= inv[2, 2]
    sample.image = _pil_transform(sample.image, Image.Transform.PERSPECTIVE, tuple(inv.ravel()[:8]))
    if len(sample.boxes):
        sample.boxes = _bounds(_project(_corners(sample.boxes), h))


# -- ink ----------------------------------------------------------------------


@dataclass
class InkSpread:
    """Ink bleeding into the paper: every stroke thickens by ``radius`` pixels
    on each side (at the reference resolution).

    The opposite of a max filter (which grows the paper and erases thin
    lines). Scanned plates show staff lines 2.4-5.6 px thick on a 2970 px
    page (0.15-0.30 of a staff space); Verovio draws 1-2.5 px, so one pixel a
    side, plus the sub-pixel growth of a high binarization threshold, closes
    the gap.
    """

    radius: tuple[int, int] = (1, 1)
    p: float = 1.0
    name: str = "ink_spread"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        r = round(rng.integers(self.radius[0], self.radius[1] + 1) * strength * _res(s))
        if r < 1:
            return
        s.image = _channels(s.image, lambda c: ndimage.minimum_filter(c, size=2 * r + 1))
        s.boxes = s.boxes + np.array([-r, -r, r, r], dtype=np.float32)
        s.log.append(f"{self.name}(r={r})")


@dataclass
class InkFade:
    """Worn or unevenly inked plate: ink lightens in large patches."""

    amount: tuple[float, float] = (0.2, 0.5)
    scale: tuple[float, float] = (40, 160)
    p: float = 1.0
    name: str = "ink_fade"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        amount = rng.uniform(*self.amount) * strength
        field_ = _smooth_noise(s.image.shape[:2], rng.uniform(*self.scale), rng)
        fade = (1 - amount * field_)[..., None] if s.image.ndim == 3 else 1 - amount * field_
        s.image = 1 - (1 - s.image) * fade
        s.log.append(f"{self.name}({amount:.2f})")


@dataclass
class ShowThrough:
    """The reverse side seen through thin paper: a faint mirrored ghost."""

    alpha: tuple[float, float] = (0.04, 0.12)
    p: float = 1.0
    name: str = "show_through"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        alpha = rng.uniform(*self.alpha) * strength
        shift = tuple(int(v) for v in rng.integers(-40, 41, size=2))
        ghost = np.roll(s.image[:, ::-1], shift, axis=(0, 1))
        ghost = _channels(ghost, lambda c: ndimage.gaussian_filter(c, rng.uniform(1.5, 3.0)))
        s.image = s.image * (1 - alpha * (1 - ghost))
        s.log.append(f"{self.name}({alpha:.2f})")


# -- paper --------------------------------------------------------------------

PAPER = (
    (1.00, 1.00, 1.00),
    (0.96, 0.94, 0.86),
    (0.93, 0.88, 0.74),
    (0.90, 0.90, 0.88),
    (0.88, 0.81, 0.67),
)
INK = ((0.04, 0.04, 0.04), (0.14, 0.11, 0.08), (0.10, 0.10, 0.14))


@dataclass
class Paper:
    """Paper tone, grain and stains; turns the grey image into RGB."""

    texture: tuple[float, float] = (0.01, 0.05)
    stains: tuple[float, float] = (0.0, 0.12)
    p: float = 1.0
    name: str = "paper"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        if s.image.ndim == 3:
            return
        paper = np.array(PAPER[rng.integers(len(PAPER))], dtype=np.float32)
        ink = np.array(INK[rng.integers(len(INK))], dtype=np.float32)
        paper = 1 - (1 - paper) * strength
        grey = s.image[..., None]
        img = ink + (paper - ink) * grey
        shape = s.image.shape
        grain = rng.uniform(*self.texture) * strength
        tex = _smooth_noise(shape, 3, rng) * 0.5 + _smooth_noise(shape, 25, rng) * 0.5
        stain = rng.uniform(*self.stains) * strength
        blot = np.clip(_smooth_noise(shape, rng.uniform(150, 500), rng) * 2 - 1, 0, 1)
        img = img * (1 - grain * tex[..., None]) * (1 - stain * blot[..., None])
        s.image = img.astype(np.float32)
        s.log.append(f"{self.name}(grain={grain:.2f}, stain={stain:.2f})")


# -- scanner geometry ---------------------------------------------------------


@dataclass
class Rotate:
    """Page placed slightly askew on the scanner."""

    degrees: float = 1.0
    p: float = 1.0
    name: str = "rotate"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        angle = np.radians(rng.uniform(-self.degrees, self.degrees) * strength)
        w, h = s.size
        cx, cy = w / 2, h / 2
        c, sn = np.cos(angle), np.sin(angle)
        hmat = np.array(
            [[c, -sn, cx - c * cx + sn * cy], [sn, c, cy - sn * cx - c * cy], [0, 0, 1]]
        )
        _warp_projective(s, hmat)
        s.log.append(f"{self.name}({np.degrees(angle):.2f}°)")


@dataclass
class Perspective:
    """Photographed rather than scanned: corners pulled independently."""

    shift: float = 0.03
    """Largest corner displacement, as a fraction of the page size."""
    p: float = 1.0
    name: str = "perspective"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        w, h = s.size
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        jitter = rng.uniform(-1, 1, size=(4, 2)) * self.shift * strength * np.array([w, h])
        _warp_projective(s, _homography(src, src + jitter))
        s.log.append(f"{self.name}({self.shift * strength:.3f})")


@dataclass
class Warp:
    """Page not flat on the glass: rows bend, most near the binding."""

    amplitude: tuple[float, float] = (0.002, 0.008)
    """Vertical displacement, as a fraction of the page height."""
    p: float = 1.0
    name: str = "warp"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        w, h = s.size
        amp = rng.uniform(*self.amplitude) * h * strength
        xs = np.arange(w, dtype=np.float32)
        phase = rng.uniform(0, 2 * np.pi)
        wavelength = rng.uniform(0.8, 2.0) * w
        side = rng.choice([0.0, 1.0])  # binding on the left or the right
        curl = ((xs / w if side else 1 - xs / w) ** 3).astype(np.float32)
        offset = amp * (0.5 * np.sin(2 * np.pi * xs / wavelength + phase) + curl)
        # out(x, y) = in(x, y - offset(x)): shift every column by linear interpolation
        y = np.arange(h, dtype=np.float32)[:, None] - offset[None, :]
        y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        t = (y - np.floor(y)).astype(np.float32)
        cols = np.broadcast_to(np.arange(w), (h, w))
        inside = (y >= 0) & (y <= h - 1)

        def shift(c: np.ndarray) -> np.ndarray:
            out = c[y0, cols] * (1 - t) + c[y1, cols] * t
            return np.where(inside, out, 1.0).astype(np.float32)

        s.image = _channels(s.image, shift)
        if len(s.boxes):
            # a box's rows move by the offsets found under its width
            x1 = np.clip(s.boxes[:, 0].astype(int), 0, w - 1)
            x2 = np.clip(s.boxes[:, 2].astype(int), 0, w - 1)
            lo = np.array([offset[a : b + 1].min() for a, b in zip(x1, x2, strict=True)])
            hi = np.array([offset[a : b + 1].max() for a, b in zip(x1, x2, strict=True)])
            s.boxes = s.boxes + np.stack([0 * lo, lo, 0 * hi, hi], axis=1).astype(np.float32)
        s.log.append(f"{self.name}({amp:.1f}px)")


@dataclass
class Border:
    """Scanner lid shadow along one edge."""

    width: tuple[float, float] = (0.01, 0.05)
    p: float = 1.0
    name: str = "border"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        w, h = s.size
        edge = int(rng.integers(4))
        size = max(1, int(rng.uniform(*self.width) * strength * (w if edge % 2 else h)))
        ramp = np.linspace(0.15, 1.0, size, dtype=np.float32) ** 2
        mask = np.ones((h, w), dtype=np.float32)
        if edge == 0:
            mask[:size] = ramp[:, None]
        elif edge == 1:
            mask[:, -size:] = ramp[::-1][None, :]
        elif edge == 2:
            mask[-size:] = ramp[::-1][:, None]
        else:
            mask[:, :size] = ramp[None, :]
        s.image = s.image * (mask[..., None] if s.image.ndim == 3 else mask)
        s.log.append(f"{self.name}({size}px)")


# -- optics and sensor --------------------------------------------------------


@dataclass
class Blur:
    sigma: tuple[float, float] = (0.4, 1.2)
    p: float = 1.0
    name: str = "blur"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        sigma = rng.uniform(*self.sigma) * strength * _res(s)
        if sigma < 0.1:
            return
        s.image = _channels(s.image, lambda c: ndimage.gaussian_filter(c, sigma))
        s.log.append(f"{self.name}({sigma:.2f})")


@dataclass
class Downscale:
    """Scanned at a lower resolution, then upsampled to the page size."""

    factor: tuple[float, float] = (0.45, 0.8)
    p: float = 1.0
    name: str = "downscale"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        f = 1 - (1 - rng.uniform(*self.factor)) * strength
        w, h = s.size
        small = (max(1, round(w * f)), max(1, round(h * f)))

        def one(c: np.ndarray) -> np.ndarray:
            img = Image.fromarray(np.ascontiguousarray(c), mode="F")
            img = img.resize(small, Image.Resampling.BILINEAR)
            return np.asarray(img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)

        s.image = _channels(s.image, one)
        s.log.append(f"{self.name}({f:.2f})")


@dataclass
class Illumination:
    """Uneven lighting: a gradient and a vignette."""

    strength: tuple[float, float] = (0.05, 0.25)
    p: float = 1.0
    name: str = "illumination"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        w, h = s.size
        amount = rng.uniform(*self.strength) * strength
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        angle = rng.uniform(0, 2 * np.pi)
        grad = np.cos(angle) * xx / w + np.sin(angle) * yy / h
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
        vignette = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2
        light = 1 - amount * np.clip(0.6 * grad + 0.4 * vignette, 0, 1)
        s.image = s.image * (light[..., None] if s.image.ndim == 3 else light)
        s.log.append(f"{self.name}({amount:.2f})")


@dataclass
class Noise:
    sigma: tuple[float, float] = (0.01, 0.04)
    p: float = 1.0
    name: str = "noise"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        sigma = rng.uniform(*self.sigma) * strength
        s.image = s.image + rng.normal(0, sigma, s.image.shape).astype(np.float32)
        s.log.append(f"{self.name}({sigma:.3f})")


@dataclass
class Speckle:
    """Dust and pinholes: small dark dots on the paper, light dots in the ink."""

    density: tuple[float, float] = (5e-6, 4e-5)
    p: float = 1.0
    name: str = "speckle"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        w, h = s.size
        n = int(rng.uniform(*self.density) * strength * w * h)
        if n < 1:
            return
        img = s.image.copy()
        for value, count in ((0.0, n), (1.0, n // 2)):
            ys = rng.integers(0, h, count)
            xs = rng.integers(0, w, count)
            radius = rng.integers(0, 3, count)
            for r in range(3):
                pick = radius == r
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            img[
                                np.clip(ys[pick] + dy, 0, h - 1), np.clip(xs[pick] + dx, 0, w - 1)
                            ] = value
        s.image = img
        s.log.append(f"{self.name}({n})")


@dataclass
class Binarize:
    """Scanner software thresholding to pure black and white: jagged edges."""

    threshold: tuple[float, float] = (0.45, 0.65)
    jitter: tuple[float, float] = (0.0, 0.08)
    p: float = 1.0
    name: str = "binarize"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        grey = s.image.mean(axis=2) if s.image.ndim == 3 else s.image
        t = rng.uniform(*self.threshold)
        jitter = rng.uniform(*self.jitter) * strength
        noisy = grey + rng.normal(0, jitter, grey.shape).astype(np.float32) if jitter else grey
        s.image = (noisy > t).astype(np.float32)
        s.log.append(f"{self.name}(t={t:.2f}, jitter={jitter:.2f})")


@dataclass
class Jpeg:
    quality: tuple[int, int] = (35, 85)
    p: float = 1.0
    name: str = "jpeg"

    def __call__(self, s: Sample, rng: np.random.Generator, strength: float) -> None:
        drawn = int(rng.integers(self.quality[0], self.quality[1] + 1))
        q = int(100 - (100 - drawn) * strength + 0.5)
        u8 = (np.clip(s.image, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, format="JPEG", quality=max(5, min(q, 95)))
        s.image = np.asarray(Image.open(buf), dtype=np.float32) / 255.0
        s.log.append(f"{self.name}(q={q})")
