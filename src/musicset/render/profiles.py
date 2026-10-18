"""Engraving variety: render options sampled per score.

Every page of a fixed-option build looks alike (one font, hairline staff
lines, the same spacing and page), while real editions vary in all of these.
Measured on IMSLP scans (Schumann op. 68, Durand and a second edition), staff
lines are 0.15-0.30 of a staff space thick; Verovio's defaults draw 0.056, and
its maximum staff line width reaches only 0.15. The heavier end comes from
ink spread in augmentation (:mod:`musicset.augment`).

Ranges stay inside Verovio's published limits (``toolkit.getAvailableOptions``).
"""

from __future__ import annotations

import random
from dataclasses import replace

from musicset.render.options import RenderOptions

FONTS = (("Leipzig", 0.3), ("Bravura", 0.2), ("Gootville", 0.2), ("Leland", 0.2), ("Petaluma", 0.1))
PAGES = (((2100, 2970), 0.6), ((2159, 2794), 0.2), ((2400, 3200), 0.2))  # A4, Letter, large folio


def _pick[T](rng: random.Random, weighted: tuple[tuple[T, float], ...]) -> T:
    values, weights = zip(*weighted, strict=True)
    return rng.choices(values, weights=weights)[0]


def _heavy(rng: random.Random, lo: float, hi: float) -> float:
    """Uniform-ish but leaning to the heavy end: old engravings are bold."""
    return round(rng.triangular(lo, hi, lo + 0.75 * (hi - lo)), 3)


def has_encoded_breaks(musicxml: str) -> bool:
    """The source keeps its own line breaks (MuseScore exports, PDMX)."""
    return 'new-system="yes"' in musicxml or 'new-page="yes"' in musicxml


def sample_render_options(
    rng: random.Random, base: RenderOptions | None = None, *, encoded_breaks: bool = False
) -> RenderOptions:
    """Engraving options for one score. ``encoded_breaks``: the source has line
    breaks to keep (half the time); without any, ``encoded`` would put a whole
    piece on one system."""
    base = base or RenderOptions()
    width, height = _pick(rng, PAGES)
    breaks = "encoded" if encoded_breaks and rng.random() < 0.5 else "auto"
    extra = {
        "staffLineWidth": _heavy(rng, 0.10, 0.30),
        "stemWidth": _heavy(rng, 0.15, 0.50),
        "barLineWidth": _heavy(rng, 0.20, 0.80),
        "thickBarlineThickness": _heavy(rng, 0.8, 2.0),
        "ledgerLineThickness": _heavy(rng, 0.15, 0.50),
        "slurMidpointThickness": round(rng.uniform(0.4, 1.0), 3),
        "slurEndpointThickness": round(rng.uniform(0.05, 0.2), 3),
        "tieMidpointThickness": round(rng.uniform(0.35, 0.9), 3),
        "hairpinThickness": round(rng.uniform(0.1, 0.6), 3),
        "tupletBracketThickness": round(rng.uniform(0.1, 0.5), 3),
        "beamMaxSlope": rng.randint(4, 20),
        "spacingLinear": round(rng.uniform(0.15, 0.45), 3),
        "spacingNonLinear": round(rng.uniform(0.45, 0.75), 3),
        "spacingStaff": rng.randint(6, 20),
        "spacingSystem": rng.randint(2, 16),
        "justifyVertically": rng.random() < 0.3,
        "graceFactor": round(rng.uniform(0.6, 0.8), 3),
        "lyricSize": round(rng.uniform(3.5, 5.5), 2),
        "pageMarginLeft": rng.randint(30, 200),
        "pageMarginRight": rng.randint(30, 200),
        "pageMarginTop": rng.randint(50, 200),
        "pageMarginBottom": rng.randint(50, 200),
    }
    return replace(
        base,
        font=_pick(rng, FONTS),
        page_width=width,
        page_height=height,
        scale=rng.randint(70, 120),
        breaks=breaks,
        header="auto" if rng.random() < 0.8 else "none",
        footer="none" if rng.random() < 0.6 else "auto",
        extra={**base.extra, **extra},
    )
