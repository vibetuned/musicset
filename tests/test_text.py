import re

from musicset import annotate_svg
from musicset.render.engrave import rasterize


def test_every_text_element_is_measured(tiny_svg: str):
    raster = rasterize(tiny_svg)
    assert len(raster.text_ink) == len(re.findall(r"<text[\s>]", tiny_svg))
    assert raster.size == (2100, 2970)
    assert any(box is not None for box in raster.text_ink)


def test_measured_text_box_replaces_the_estimate(tiny_svg: str):
    raster = rasterize(tiny_svg)
    estimated = annotate_svg(tiny_svg)
    measured = annotate_svg(tiny_svg, text_ink=raster.text_ink, ink_scale=raster.scale)
    label_e = next(el for el in estimated.elements if el.classes[0] == "label")
    label_m = next(el for el in measured.elements if el.classes[0] == "label")
    assert label_m.text == label_e.text == "Piano"
    assert label_m.bbox is not None and label_e.bbox is not None
    assert label_m.bbox != label_e.bbox
    x1, y1, x2, y2 = label_m.bbox
    assert 0 < x1 < x2 < measured.width and 0 < y1 < y2 < measured.height


def test_measurement_is_independent_of_raster_scale(tiny_svg: str):
    full = rasterize(tiny_svg)
    half = rasterize(tiny_svg, scale=0.5)
    a = annotate_svg(tiny_svg, text_ink=full.text_ink, ink_scale=1.0)
    b = annotate_svg(tiny_svg, text_ink=half.text_ink, ink_scale=0.5)
    box_a = next(el.bbox for el in a.elements if el.classes[0] == "label")
    box_b = next(el.bbox for el in b.elements if el.classes[0] == "label")
    assert box_a is not None and box_b is not None
    # same box in page units, up to hinting at the coarser scale (< 2 px at scale 1)
    assert all(abs(p - q) < 20 for p, q in zip(box_a, box_b, strict=True))


def test_smufl_text_falls_back_when_the_font_lacks_the_glyph(tempo_xml: str):
    from musicset import Engraving, RenderOptions

    # Gootville has no metronome notes: the run is drawn with the fallback font
    gootville = Engraving.load(tempo_xml, RenderOptions(font="Gootville")).svg(1)
    assert 'font-family="Leipzig"' in gootville and 'font-family="Gootville"' not in gootville
    # Leland has them: untouched
    leland = Engraving.load(tempo_xml, RenderOptions(font="Leland")).svg(1)
    assert 'font-family="Leland"' in leland
