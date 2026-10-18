"""Engraving options.

Defaults reproduce the kakigori/gelato renders (Verovio defaults plus a
viewBox and a fixed id seed); every field is a knob the render stage of the
augmentation work will sample.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Semantic attributes written onto SVG groups as data-* (pitch, duration...).
# They do not change the drawing.
DEFAULT_SVG_ATTRIBUTES = (
    "note@pname",
    "note@oct",
    "note@dur",
    "note@dots",
    "note@grace",
    "chord@dur",
    "chord@dots",
    "rest@dur",
    "rest@dots",
)


@dataclass(frozen=True, slots=True)
class RenderOptions:
    font: str = "Leipzig"
    page_width: int = 2100
    page_height: int = 2970
    scale: int = 100
    breaks: str = "auto"
    """``auto`` (Verovio lays out), ``encoded`` (keep the source's line/page breaks), ``none``..."""
    header: str = "auto"
    footer: str = "auto"
    xml_id_seed: int = 42
    svg_attributes: tuple[str, ...] = DEFAULT_SVG_ATTRIBUTES
    extra: dict[str, Any] = field(default_factory=dict)
    """Any other Verovio option, passed through verbatim (camelCase keys)."""

    def to_verovio(self) -> dict[str, Any]:
        return {
            "svgViewBox": True,
            "font": self.font,
            "pageWidth": self.page_width,
            "pageHeight": self.page_height,
            "scale": self.scale,
            "breaks": self.breaks,
            "header": self.header,
            "footer": self.footer,
            "xmlIdSeed": self.xml_id_seed,
            "svgAdditionalAttribute": list(self.svg_attributes),
            **self.extra,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
