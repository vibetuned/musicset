"""SMuFL glyph names and codepoint groups.

Names come from the font metadata Verovio ships, so the table always matches
the installed renderer. Codepoints are shared by every SMuFL font, which is
why annotations key glyphs by codepoint rather than by drawn shape.
"""

import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

# Flags, by the shortest duration they mark (SMuFL range U+E240-U+E24F:
# up/down pairs for 8th, 16th, 32nd, 64th, ... 1024th)
FLAGS_8TH = frozenset({"E240", "E241"})
FLAGS_16TH = frozenset({"E242", "E243"})
FLAGS_32ND_OR_SHORTER = frozenset(f"E24{d}" for d in "456789ABCDEF")


def verovio_data_dir() -> Path:
    import verovio

    return Path(verovio.__file__).parent / "data"


@cache
def available_fonts() -> tuple[str, ...]:
    """Music fonts Verovio can engrave with (those shipping glyph metadata)."""
    data = verovio_data_dir()
    return tuple(sorted(p.stem for p in data.glob("*.xml") if (data / p.stem).is_dir()))


@cache
def glyph_names(font: str = "Bravura") -> dict[str, str]:
    """Codepoint -> SMuFL canonical name, e.g. ``"E050" -> "gClef"``."""
    root = ET.parse(verovio_data_dir() / f"{font}.xml").getroot()
    return {g.attrib["c"]: g.attrib["n"] for g in root.iter("g") if "c" in g.attrib}


def glyph_name(code: str) -> str | None:
    return glyph_names().get(code)
