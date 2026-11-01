"""Score datasets for optical music recognition."""

from musicset.annotate.extract import annotate_svg
from musicset.annotate.schema import Element, Glyph, PageAnnotation
from musicset.render.engrave import Engraving, rasterize
from musicset.render.options import RenderOptions
from musicset.symbols import (
    SYMBOL_GROUPS,
    SYMBOLS,
    build_symbol_dataset,
    extract_dataset_symbols,
    extract_page_symbols,
    generate_symbol_samples,
    load_symbol_dataset,
    load_symbol_sources,
    render_symbol,
)
from musicset.views.detection import ClassMap, detection_annotations

__all__ = [
    "SYMBOLS",
    "SYMBOL_GROUPS",
    "ClassMap",
    "Element",
    "Engraving",
    "Glyph",
    "PageAnnotation",
    "RenderOptions",
    "annotate_svg",
    "build_symbol_dataset",
    "detection_annotations",
    "extract_dataset_symbols",
    "extract_page_symbols",
    "generate_symbol_samples",
    "load_symbol_dataset",
    "load_symbol_sources",
    "rasterize",
    "render_symbol",
]
