from pathlib import Path

import pytest

from musicset import Engraving, PageAnnotation, RenderOptions, annotate_svg
from musicset.ids import inject_ids

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_xml() -> str:
    return inject_ids((FIXTURES / "tiny.musicxml").read_text(encoding="utf-8"), "tny")


@pytest.fixture(scope="session")
def tiny_svg(tiny_xml: str) -> str:
    return Engraving.load(tiny_xml, RenderOptions()).svg(1)


@pytest.fixture(scope="session")
def tiny_page(tiny_svg: str) -> PageAnnotation:
    return annotate_svg(tiny_svg, score="tiny", page=1)


@pytest.fixture(scope="session")
def tiny_page_geometric(tiny_xml: str) -> PageAnnotation:
    """The tiny score rendered without data-* attributes: stems read from flags and beams."""
    svg = Engraving.load(tiny_xml, RenderOptions(svg_attributes=())).svg(1)
    return annotate_svg(svg, score="tiny", page=1)


TEMPO = (
    '<direction placement="above"><direction-type><words>Allegro</words></direction-type>'
    "<direction-type><metronome><beat-unit>quarter</beat-unit><beat-unit-dot/>"
    '<per-minute>100</per-minute></metronome></direction-type><sound tempo="150"/></direction>'
)


@pytest.fixture(scope="session")
def tempo_xml(tiny_xml: str) -> str:
    """The tiny score with a metronome mark: SMuFL characters inside text."""
    return tiny_xml.replace("</attributes>", "</attributes>" + TEMPO, 1)
