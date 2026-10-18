"""Engraving with Verovio, and rasterizing its SVG pages."""

from __future__ import annotations

import io
import re
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import Any

from musicset.annotate.geometry import BBox, union
from musicset.annotate.smufl import verovio_data_dir
from musicset.render.options import RenderOptions


class EngraveError(RuntimeError):
    pass


@cache
def _font_glyphs(font: str) -> frozenset[str]:
    """Codepoints (``"ECA5"``) Verovio has glyphs for in a music font."""
    return frozenset(p.stem for p in (verovio_data_dir() / font).glob("*.xml"))


_FONT_RUN = re.compile(r'(<tspan[^>]*\bfont-family=")([^"]+)("[^>]*>)([^<]*)')


def _fallback_smufl_text(svg: str, font: str, fallback: str) -> str:
    """Point SMuFL text runs the engraving font cannot draw at the fallback font.

    Verovio draws notation glyphs a font lacks from its fallback, but writes
    SMuFL characters in text (metronome marks...) with the engraving font
    regardless. Browsers fall back per glyph; cairo draws nothing, so with
    Gootville (no metronome notes) tempo marks lost their note.
    """
    if font == fallback:
        return svg
    have = _font_glyphs(font)

    def fix(m: re.Match[str]) -> str:
        if m.group(2) != font:
            return m.group(0)
        if all(ord(c) < 0xE000 or f"{ord(c):04X}" in have for c in m.group(4)):
            return m.group(0)
        return f"{m.group(1)}{fallback}{m.group(3)}{m.group(4)}"

    return _FONT_RUN.sub(fix, svg)


@dataclass(slots=True)
class Engraving:
    """A score loaded into Verovio and laid out with one set of options."""

    toolkit: Any
    """A ``verovio.toolkit`` (untyped SWIG binding)."""
    options: RenderOptions

    @classmethod
    def load(cls, xml: str, options: RenderOptions) -> Engraving:
        import verovio

        verovio.enableLog(verovio.LOG_OFF)
        verovio.setDefaultResourcePath(str(verovio_data_dir()))
        tk = verovio.toolkit()
        tk.setOptions(options.to_verovio())
        if not tk.loadData(xml):
            raise EngraveError("Verovio could not load the score")
        return cls(tk, options)

    @property
    def version(self) -> str:
        return self.toolkit.getVersion()

    @property
    def page_count(self) -> int:
        return self.toolkit.getPageCount()

    def svg(self, page: int) -> str:
        """SVG of one page (1-based)."""
        fallback = str(self.options.extra.get("fontFallback", "Leipzig"))
        return _fallback_smufl_text(self.toolkit.renderToSVG(page), self.options.font, fallback)

    def pages(self) -> Iterator[tuple[int, str]]:
        for page in range(1, self.page_count + 1):
            yield page, self.svg(page)

    def mei(self) -> str:
        """The score as MEI; xml:ids match the SVG ids."""
        return self.toolkit.getMEI()


@dataclass(slots=True)
class Raster:
    png: bytes
    size: tuple[int, int]
    scale: float
    text_ink: list[BBox | None]
    """Ink box (pixels) of every ``<text>`` element, in document order."""


_TEXT_TAG = re.compile(r"<text(?=[\s>])")
_TEXT_KEY = "data-musicset-text"


@contextmanager
def _recording_text_ink(boxes: dict[int, BBox]) -> Generator[None]:
    """Record the glyph outlines cairosvg lays out for each text element.

    cairosvg builds a text node's outlines as a path, then draws it; the path
    extents at that point are exactly the node's ink. Tspans inherit the key
    attribute, so their ink joins their text element's box.
    """
    import cairosvg.surface as cs

    original = cs.text

    def recording(surface: Any, node: Any, draw_as_text: bool = False) -> Any:
        result = original(surface, node, draw_as_text)
        key = node.get(_TEXT_KEY)
        if not draw_as_text and key is not None:
            ctx = surface.context
            x1, y1, x2, y2 = ctx.path_extents()
            if x2 > x1 and y2 > y1:
                corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                xs, ys = zip(*(ctx.user_to_device(x, y) for x, y in corners), strict=True)
                box = (min(xs), min(ys), max(xs), max(ys))
                boxes[int(key)] = union(boxes.get(int(key)), box) or box
        return result

    # cairosvg compares TAGS entries against surface.text to pick glyph drawing,
    # so both must point at the wrapper (a monkeypatch: untyped by nature)
    cs.TAGS["text"] = cs.TAGS["tspan"] = recording
    cs.text = recording  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        cs.TAGS["text"] = cs.TAGS["tspan"] = original
        cs.text = original


def rasterize(svg: str, scale: float = 1.0) -> Raster:
    """Render an SVG page to PNG on a white background, measuring its text as drawn."""
    import cairosvg
    from PIL import Image

    count = 0

    def tag(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f'{match.group(0)} {_TEXT_KEY}="{count - 1}"'

    keyed = _TEXT_TAG.sub(tag, svg)
    boxes: dict[int, BBox] = {}
    with _recording_text_ink(boxes):
        png = cairosvg.svg2png(
            bytestring=keyed.encode("utf-8"), scale=scale, background_color="white"
        )
    assert isinstance(png, bytes)
    with Image.open(io.BytesIO(png)) as im:
        size = im.size
    return Raster(png, size, scale, [boxes.get(i) for i in range(count)])
