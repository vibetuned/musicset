"""Fornés: the CVC "Music Symbols" dataset of handwritten clefs and accidentals (Fornés,
Lladós & Sánchez, 2008).

4,094 scanned images, white ink on black, each cropped to one symbol with the staff
removed: flat, natural, sharp and double sharp by 8 writers, treble, bass and alto clefs
by those 8 and 14 more (52 clef files carry no writer name). Most bass clefs lack their
two dots. There are no pen strokes, so the images are placed on the symbol canvas the way
EMNIST words are (:func:`musicset.symbols.render_ink_sample`).

The writer is the file name's first word (``agata_BN_138.bmp``) or, for the clefs, its
second (``sol_agata_BN._9.bmp``). Four of the 8 accidental writers go by another name in
the clef folders (``FORNES_ALIASES``: ``ali`` / ``alicia``, ``josep`` / ``jllados``, ``juan``
/ ``jrodriguez``, ``marcal`` / ``marsal``); each pair is one writer here, so a writer split
never puts one hand on both sides.

:func:`load_fornes` reads the archive (zip or extracted directory) into
:class:`InkSample` records whose ``label`` is already the ``musicset`` symbol class.
The dataset has no licence; its readme asks users to cite the paper and acknowledge the
Document Analysis Group of the Computer Vision Center.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FORNES_CLASSES: dict[str, str] = {
    "ACCIDENTAL_DoubSharp": "accidentalDoubleSharp",
    "ACCIDENTAL_Flat": "accidentalFlat",
    "ACCIDENTAL_Natural": "accidentalNatural",
    "ACCIDENTAL_Sharp": "accidentalSharp",
    "CLEF_Alto": "cClef",
    "CLEF_Bass": "fClef",
    "CLEF_Trebble": "gClef",
}
"""Fornés folder name -> ``musicset`` symbol class."""

FORNES_ALIASES: dict[str, str] = {
    "alicia": "ali",
    "jllados": "josep",
    "jrodriguez": "juan",
    "marsal": "marcal",
}
"""Clef-folder name -> the same writer's accidental-folder name: each pair splits the
dataset's classes between them exactly (one draws only accidentals, the other only clefs)."""

UNKNOWN_WRITER = "?"
"""The writer of a file without a name (``sol10.bmp``): one group of its own."""

FORNES_WRITER_BASE = 2000
"""Writer ids are ``FORNES_WRITER_BASE`` plus the writer's rank, clear of HOMUS writers
(1-100) and contributed ones (from 1000)."""

_CLEF_PREFIXES = ("do", "fa", "sol")


@dataclass(frozen=True, slots=True)
class InkSample:
    writer: str
    label: str
    """``musicset`` symbol class (see ``FORNES_CLASSES``)."""
    image: np.ndarray
    """uint8, black ink on white, cropped to the ink."""
    source: str = ""
    """File the sample came from, for error messages."""


def writer_of(filename: str) -> str:
    """``agata_BN_138.bmp`` -> ``agata``, ``sol_alicia_BN._9.bmp`` -> ``ali`` (the alias
    merged), ``sol10.bmp`` -> ``UNKNOWN_WRITER``."""
    parts = Path(filename).stem.split("_")
    if len(parts) < 3:
        return UNKNOWN_WRITER
    name = (parts[1] if parts[0].lower() in _CLEF_PREFIXES else parts[0]).lower()
    return FORNES_ALIASES.get(name, name)


def ink_image(data: bytes) -> np.ndarray | None:
    """A scan as black ink on white, cropped to the ink; ``None`` without ink. The
    background is the border's majority, so either polarity reads the same."""
    img = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.uint8)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if np.median(border) < 128:
        img = 255 - img
    rows = np.flatnonzero((img < 128).any(axis=1))
    cols = np.flatnonzero((img < 128).any(axis=0))
    if not len(rows) or not len(cols):
        return None
    return np.ascontiguousarray(img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1])


def _iter_files(path: Path) -> Iterator[tuple[str, bytes]]:
    """``(member path, bytes)`` for every image in a zip or directory."""
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() == ".bmp":
                yield str(file.relative_to(path)), file.read_bytes()
        return
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.lower().endswith(".bmp"):
                yield name, archive.read(name)


def load_fornes(path: Path, *, classes: dict[str, str] = FORNES_CLASSES) -> list[InkSample]:
    """Every Fornés image under ``path`` (zip or directory) whose folder is in ``classes``."""
    samples: list[InkSample] = []
    for name, data in _iter_files(path):
        parts = Path(name).parts
        label = classes.get(parts[-2]) if len(parts) >= 2 else None
        if label is None:
            continue
        try:
            image = ink_image(data)
        except OSError as e:
            raise ValueError(f"{path}:{name}: {e}") from e
        if image is not None:
            samples.append(InkSample(writer_of(name), label, image, source=name))
    return samples


def fornes_writer_ids(samples: list[InkSample]) -> dict[str, int]:
    """Writer name -> integer id (``FORNES_WRITER_BASE`` + rank among the names)."""
    return {w: FORNES_WRITER_BASE + i for i, w in enumerate(sorted({s.writer for s in samples}))}
