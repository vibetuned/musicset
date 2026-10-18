"""Score datasets for optical music recognition."""

from musicset.annotate.extract import annotate_svg
from musicset.annotate.schema import Element, Glyph, PageAnnotation
from musicset.render.engrave import Engraving, rasterize
from musicset.render.options import RenderOptions
from musicset.views.detection import ClassMap, detection_annotations

__all__ = [
    "ClassMap",
    "Element",
    "Engraving",
    "Glyph",
    "PageAnnotation",
    "RenderOptions",
    "annotate_svg",
    "detection_annotations",
    "rasterize",
]
