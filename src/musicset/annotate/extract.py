"""Read a canonical :class:`PageAnnotation` off a Verovio SVG page.

Single pass over the SVG tree: every drawing primitive's absolute box is
computed once and folded up into the boxes of its enclosing groups, so the
cost is linear in the size of the page.

The geometric heuristics that refine Verovio's classes (barline style, beam
multiplicity, stem duration) are ported from the kakigori extractor and keep
its numeric behaviour. Known, intentional differences:

* a stem's class comes from its note's written duration when the SVG carries
  it (``data-dur``, see ``RenderOptions.svg_attributes``). The geometric rule
  labels every stem under a beam by the beam's line count, so in mixed groups
  (16th + 8th + 16th) the 8th's stem became ``stem16``: 0.37% of 115k stems in
  kakigori's validation/test sets. The geometric rule remains the fallback;
* stems carrying a 32nd-or-shorter flag are ``stem32`` (kakigori's glyph map
  had no 32nd flags, so those stems fell through to ``stem4``);
* Verovio ``bounding-box`` overlays (option ``svgBoundingBoxes``) are ignored
  instead of being read as ink;
* malformed text attributes skip that text run instead of the whole page.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from svgpathtools import parse_path

from musicset.annotate import smufl
from musicset.annotate.geometry import (
    IDENTITY,
    BBox,
    Matrix,
    multiply,
    pad,
    parse_transform,
    transform_bbox,
    union,
)
from musicset.annotate.schema import Element, Glyph, PageAnnotation

SVG_NS = "http://www.w3.org/2000/svg"
_NS = {"svg": SVG_NS}
_HREF = "{http://www.w3.org/1999/xlink}href"

_G = f"{{{SVG_NS}}}g"
_USE = f"{{{SVG_NS}}}use"
_PATH = f"{{{SVG_NS}}}path"
_RECT = f"{{{SVG_NS}}}rect"
_LINE = f"{{{SVG_NS}}}line"
_TEXT = f"{{{SVG_NS}}}text"
_TSPAN = f"{{{SVG_NS}}}tspan"
_ELLIPSE = f"{{{SVG_NS}}}ellipse"
_DEFS = f"{{{SVG_NS}}}defs"
_DESC = f"{{{SVG_NS}}}desc"

_FIRST_MOVE_X = re.compile(r"M([0-9.]+)")
_NUMBER = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

# Staff lines closer than this (SVG units, vertically) are merged into one
# system-staff box
_SYSTEM_STAFF_MERGE = 1000
# Beam polygons closer than this (SVG units) count as the same beam level
_BEAM_LEVEL_TOLERANCE = 20

type Points = list[tuple[float, ...]]


def annotate_svg(
    svg: str | bytes | Path,
    *,
    score: str = "",
    page: int = 0,
    text_ink: Sequence[BBox | None] | None = None,
    ink_scale: float = 1.0,
) -> PageAnnotation:
    """Extract every element and glyph of one rendered page.

    ``text_ink`` holds the measured ink box (raster pixels at ``ink_scale``) of
    every ``<text>`` element in document order, as ``rasterize`` returns it.
    Without it, text boxes are estimated from font size and character count,
    which overlaps the real ink at IoU ~0.6 only.
    """
    if isinstance(svg, Path):
        svg = svg.read_bytes()
    return _Extractor(ET.fromstring(svg), text_ink, ink_scale).run(score, page)


def _classes(el: ET.Element) -> list[str]:
    return el.get("class", "").split()


def _poly_points(poly: ET.Element) -> Points:
    return [tuple(map(float, pt.split(","))) for pt in poly.get("points", "").split() if "," in pt]


def _poly_width(points: Points) -> float:
    xs = [p[0] for p in points]
    return max(xs) - min(xs)


def _path_bbox(path: ET.Element) -> BBox:
    """Local box of a ``<path>``, grown by half its stroke width."""
    xmin, xmax, ymin, ymax = parse_path(path.get("d", "")).bbox()
    return pad((xmin, ymin, xmax, ymax), float(path.get("stroke-width", 0)) / 2)


def _defs_bboxes(root: ET.Element) -> dict[str, BBox]:
    """Glyph id -> box of the glyph outline in ``<defs>``."""
    boxes: dict[str, BBox] = {}
    for g in root.findall(".//svg:defs/svg:g", _NS):
        gid = g.get("id")
        path = g.find("svg:path", _NS)
        if not gid or path is None:
            continue
        try:
            xmin, xmax, ymin, ymax = parse_path(path.get("d", "")).bbox()
        except Exception:
            continue
        # Verovio stores glyph outlines flipped: transform="scale(1,-1)"
        if "scale(1,-1)" in path.get("transform", "").replace(" ", ""):
            ymin, ymax = -ymax, -ymin
        boxes[gid] = (xmin, ymin, xmax, ymax)
    return boxes


def _beam_levels(stem: ET.Element, polygons: list[Points]) -> int | None:
    """Number of distinct beam lines crossing a stem, or None if unmeasurable."""
    stem_path = stem.find(".//svg:path", _NS)
    if stem_path is None:
        return None
    nums = [float(n) for n in _NUMBER.findall(stem_path.get("d", ""))]
    if len(nums) < 4:
        return None
    stem_x = nums[0]
    stem_ymin, stem_ymax = min(nums[1], nums[3]), max(nums[1], nums[3])

    crossings = []
    for pts in polygons:
        xs = [p[0] for p in pts]
        min_x, max_x = min(xs), max(xs)
        if not min_x - 5 <= stem_x <= max_x + 5:
            continue
        y_at_stem = sum(p[1] for p in pts) / len(pts)
        if max_x - min_x > 0:
            left_ys = [p[1] for p in pts if p[0] - min_x < 5]
            right_ys = [p[1] for p in pts if max_x - p[0] < 5]
            if left_ys and right_ys:
                left_y = sum(left_ys) / len(left_ys)
                right_y = sum(right_ys) / len(right_ys)
                y_at_stem = left_y + (right_y - left_y) * (stem_x - min_x) / (max_x - min_x)
        if stem_ymin - _BEAM_LEVEL_TOLERANCE <= y_at_stem <= stem_ymax + _BEAM_LEVEL_TOLERANCE:
            crossings.append(y_at_stem)

    levels: list[float] = []
    for y in crossings:
        if not any(abs(y - level) < _BEAM_LEVEL_TOLERANCE for level in levels):
            levels.append(y)
    return len(levels)


def _duration_from_levels(levels: int | None, default: int) -> int:
    if levels is None or levels == 0:
        return default
    return {1: 8, 2: 16}.get(levels, 32)


def _stem_for_written_duration(dur: str | None) -> int | None:
    """Stem class of an MEI ``@dur`` (``"8"`` -> 8, ``"breve"`` -> 4), None if absent."""
    if dur is None:
        return None
    if not dur.isdigit():
        return 4
    return max(4, min(int(dur), 32))


class _Extractor:
    def __init__(
        self,
        root: ET.Element,
        text_ink: Sequence[BBox | None] | None = None,
        ink_scale: float = 1.0,
    ) -> None:
        self.root = root
        self.parent = {c: p for p in root.iter() for c in p}
        self._text_ink = text_ink
        self._ink_scale = ink_scale
        self._text_order = {el: i for i, el in enumerate(root.iter(_TEXT))}
        self._ink_to_page = 1.0
        self.defs = _defs_bboxes(root)
        self._matrices: dict[ET.Element, Matrix] = {}
        self._global_beams: list[Points] | None = None

        self.elements: list[Element] = []
        self.glyphs: list[Glyph] = []
        self._texts: dict[int, list[str]] = {}

    # -- transforms ---------------------------------------------------------

    def matrix(self, el: ET.Element) -> Matrix:
        """Absolute transform of ``el``, composed from the root down."""
        chain = []
        node: ET.Element | None = el
        while node is not None and node not in self._matrices:
            chain.append(node)
            node = self.parent.get(node)
        m = self._matrices[node] if node is not None else IDENTITY
        for n in reversed(chain):
            m = multiply(m, parse_transform(n.get("transform")))
            self._matrices[n] = m
        return m

    def _inherited(self, el: ET.Element, attr: str) -> str | None:
        node: ET.Element | None = el
        while node is not None:
            value = node.get(attr)
            if value is not None:
                return value
            node = self.parent.get(node)
        return None

    # -- page ---------------------------------------------------------------

    def run(self, score: str, page: int) -> PageAnnotation:
        inner = self.root.find('.//svg:svg[@class="definition-scale"]', _NS)
        view_box = inner.get("viewBox") if inner is not None else None
        view_box = view_box or self.root.get("viewBox")
        if not view_box:
            raise ValueError("SVG has no viewBox")
        _, _, width, height = map(float, view_box.split())
        pixel_width, pixel_height = self._pixel_size(width, height)
        self._ink_to_page = width / (pixel_width * self._ink_scale)

        self._visit(self.root, element=None, owner=None)
        for index, runs in self._texts.items():
            self.elements[index].text = " ".join(runs)

        meta = {}
        desc = self.root.find(_DESC)
        if desc is not None and desc.text:
            meta["renderer"] = desc.text.removeprefix("Engraved by ").strip()

        return PageAnnotation(
            score=score,
            page=page,
            width=width,
            height=height,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            elements=self.elements,
            glyphs=self.glyphs,
            meta=meta,
        )

    def _pixel_size(self, width: float, height: float) -> tuple[float, float]:
        w, h = self.root.get("width"), self.root.get("height")
        if w and h:
            return float(w.removesuffix("px")), float(h.removesuffix("px"))
        outer = self.root.get("viewBox")
        if outer:
            _, _, ow, oh = map(float, outer.split())
            return ow, oh
        return width, height

    # -- traversal ----------------------------------------------------------

    def _visit(
        self, node: ET.Element, element: int | None, owner: str | None
    ) -> tuple[BBox | None, BBox | None]:
        """Walk ``node``'s subtree, returning its ink box with and without flags."""
        tag = node.tag
        if tag == _DEFS:
            return None, None
        classes = _classes(node)
        if "bounding-box" in classes:
            return None, None

        if tag == _USE:
            box = self._use(node, element, owner)
            return box, box
        if tag == _TEXT:
            box = self._text(node, element)
            return box, box
        if tag in (_PATH, _RECT, _LINE, _ELLIPSE):
            box = self._shape(node)
            return box, box

        index = None
        if tag == _G and classes:
            index = self._open_element(node, classes, element)
            element = index
        node_id = node.get("id")
        if node_id is not None and node.get("class"):
            owner = node_id

        full: BBox | None = None
        no_flags: BBox | None = None
        for child in node:
            child_full, child_no_flags = self._visit(child, element, owner)
            full = union(full, child_full)
            if "flag" not in _classes(child):
                no_flags = union(no_flags, child_no_flags)

        if index is not None:
            # Verovio nests the flag inside the stem group; a stem box covers
            # the stem line only
            self.elements[index].bbox = no_flags if "stem" in classes else full
        return full, no_flags

    def _open_element(self, g: ET.Element, classes: list[str], parent: int | None) -> int:
        if "system" in classes:
            # Staff boxes precede their system; they point at the index the
            # system element is about to take
            clusters = self._system_staves(g)
            index = len(self.elements) + len(clusters)
            for box in clusters:
                self.elements.append(
                    Element(
                        id=g.get("id"),
                        classes=("system-staff",),
                        bbox=box,
                        parent=index,
                        synthetic=True,
                    )
                )
        refined: list[str] = []
        if "barLine" in classes:
            refined.extend(self._barline_class(g))
        if "beam" in classes:
            refined.append(self._beam_class(g))
        if "stem" in classes:
            refined.append(f"stem{self._stem_duration(g)}")

        attrs = {k.removeprefix("data-"): v for k, v in g.attrib.items() if k.startswith("data-")}
        self.elements.append(
            Element(
                id=g.get("id"),
                classes=tuple(classes),
                bbox=None,
                refined=tuple(refined),
                parent=parent,
                attrs=attrs,
            )
        )
        return len(self.elements) - 1

    # -- primitives ---------------------------------------------------------

    def _use(self, use: ET.Element, element: int | None, owner: str | None) -> BBox | None:
        href = use.get(_HREF, "").lstrip("#")
        local = self.defs.get(href)
        box = transform_bbox(self.matrix(use), local) if local is not None else None
        self.glyphs.append(Glyph(code=href.split("-")[0], bbox=box, element=element, owner=owner))
        return box

    def _shape(self, el: ET.Element) -> BBox | None:
        tag = el.tag
        try:
            if tag == _PATH:
                if not el.get("d"):
                    return None
                local = _path_bbox(el)
            elif tag == _RECT:
                x, y = float(el.get("x", 0)), float(el.get("y", 0))
                w, h = float(el.get("width", 0)), float(el.get("height", 0))
                if w <= 0 or h <= 0:
                    return None
                local = (x, y, x + w, y + h)
            elif tag == _LINE:
                x1, y1 = float(el.get("x1", 0)), float(el.get("y1", 0))
                x2, y2 = float(el.get("x2", 0)), float(el.get("y2", 0))
                local = pad(
                    (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                    float(el.get("stroke-width", 0)) / 2,
                )
            else:  # ellipse
                cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
                rx, ry = float(el.get("rx", 0)), float(el.get("ry", 0))
                if rx <= 0 or ry <= 0:
                    return None
                local = (cx - rx, cy - ry, cx + rx, cy + ry)
        except Exception:
            return None
        return transform_bbox(self.matrix(el), local)

    def _text(self, text_el: ET.Element, element: int | None) -> BBox | None:
        """Box of a text element: its measured ink, else an estimate."""
        runs = [el.text.strip() for el in [text_el, *text_el.iter(_TSPAN)] if el.text]
        runs = [r for r in runs if r]
        if element is not None and runs:
            self._texts.setdefault(element, []).extend(runs)

        if self._text_ink is not None:
            ink = self._text_ink[self._text_order[text_el]]
            if ink is not None:
                f = self._ink_to_page
                return ink[0] * f, ink[1] * f, ink[2] * f, ink[3] * f
        return self._estimated_text_box(text_el)

    def _estimated_text_box(self, text_el: ET.Element) -> BBox | None:
        """Estimate from anchor, font size and length (half an em per Latin
        character, a full em per SMuFL character): SVG text has no outline."""
        matrix = self.matrix(text_el)
        box: BBox | None = None
        for el in [text_el, *text_el.iter(_TSPAN)]:
            if not el.text or not el.text.strip():
                continue
            try:
                run = self._text_run_box(el, el.text.strip())
            except ValueError:
                continue
            box = union(box, transform_bbox(matrix, run))
        return box

    def _text_run_box(self, el: ET.Element, content: str) -> BBox:
        x = float(self._inherited(el, "x") or 0.0)
        y = float(self._inherited(el, "y") or 0.0)

        font_size = 100.0
        node: ET.Element | None = el
        while node is not None:
            size = node.get("font-size")
            if size:
                value = float(size.removesuffix("px"))
                if value > 0:
                    font_size = value
                    break
            node = self.parent.get(node)

        is_smufl = any(ord(c) >= 0xE000 for c in content)
        width = len(content) * font_size * (1.0 if is_smufl else 0.5)
        anchor = self._inherited(el, "text-anchor") or "start"
        if anchor == "middle":
            x1, x2 = x - width / 2, x + width / 2
        elif anchor == "end":
            x1, x2 = x - width, x
        else:
            x1, x2 = x, x + width
        # Ascender / descender approximation
        return x1, y - font_size * 0.8, x2, y + font_size * 0.2

    # -- derived classes ----------------------------------------------------

    def _system_staves(self, system: ET.Element) -> list[BBox]:
        """One box per group of staff lines in a system."""
        lines: list[BBox] = []
        for staff in system.findall(".//svg:g", _NS):
            if "staff" not in _classes(staff):
                continue
            for path in staff.findall("./svg:path", _NS):
                if not (path.get("d") and path.get("stroke-width")):
                    continue
                try:
                    lines.append(transform_bbox(self.matrix(path), _path_bbox(path)))
                except Exception:
                    continue

        clusters: list[list[float]] = []
        for x1, y1, x2, y2 in lines:
            y_center = (y1 + y2) / 2
            for cluster in clusters:
                if abs(y_center - (cluster[1] + cluster[3]) / 2) < _SYSTEM_STAFF_MERGE:
                    cluster[0] = min(cluster[0], x1)
                    cluster[2] = max(cluster[2], x2)
                    cluster[1] = min(cluster[1], y1)
                    cluster[3] = max(cluster[3], y2)
                    break
            else:
                clusters.append([x1, y1, x2, y2])
        return [(c[0], c[1], c[2], c[3]) for c in clusters]

    def _barline_class(self, g: ET.Element) -> list[str]:
        paths = g.findall(".//svg:path", _NS)
        if len(paths) == 1:
            return ["barlineDashed" if paths[0].get("stroke-dasharray") else "barlineSingle"]
        if len(paths) < 2:
            return []

        # Group strokes by x so a barline spanning several staves stays single
        by_x: dict[float, list[ET.Element]] = {}
        for p in paths:
            m = _FIRST_MOVE_X.search(p.get("d", ""))
            if m:
                by_x.setdefault(round(float(m.group(1)), 1), []).append(p)
        xs = sorted(by_x)
        if len(xs) == 1:
            dashed = by_x[xs[0]][0].get("stroke-dasharray")
            return ["barlineDashed" if dashed else "barlineSingle"]
        if len(xs) == 2:
            sw1 = float(by_x[xs[0]][0].get("stroke-width", 1))
            sw2 = float(by_x[xs[1]][0].get("stroke-width", 1))
            return ["barlineFinal" if sw1 != sw2 else "barlineDouble"]
        return []

    def _beam_shape(self, beam: ET.Element) -> tuple[int, int, bool]:
        """(polygon count, note count, broken) for a beam group."""
        polygons = beam.findall("./svg:polygon", _NS)
        notes = sum("note" in _classes(g) for g in beam.findall(".//svg:g", _NS))
        broken = False
        if len(polygons) == notes == 2:
            # 8th + 16th under one beam: the secondary beam is visibly shorter
            widths = [_poly_width(pts) for pts in map(_poly_points, polygons) if pts]
            if widths:
                widest = max(widths)
                broken = widest > 0 and any(w < widest * 0.6 for w in widths)
        return len(polygons), notes, broken

    def _beam_class(self, g: ET.Element) -> str:
        polygons, notes, broken = self._beam_shape(g)
        if polygons == 1:
            return "beam8"
        if polygons == notes:
            return "beamBroken" if broken else "beam16"
        if polygons == 2 * notes - 1:
            return "beam32"
        return "beamBroken"

    def _stem_duration(self, stem: ET.Element) -> int:
        """Duration (4, 8, 16, 32) a stem stands for.

        Half and whole notes count as 4: a stem alone does not tell them apart.
        Read from the written duration when the renderer attached it, else
        from the flags and beams drawn around the stem.
        """
        holder = self.parent.get(stem)
        while holder is not None and not {"note", "chord"} & set(_classes(holder)):
            if "beam" in _classes(holder):
                break
            holder = self.parent.get(holder)
        if holder is None or not {"note", "chord"} & set(_classes(holder)):
            return 4

        written = _stem_for_written_duration(holder.get("data-dur"))
        if written is not None:
            return written

        codes = {
            use.get(_HREF, "").lstrip("#").split("-")[0]
            for use in holder.findall(".//svg:use", _NS)
        }
        if codes & smufl.FLAGS_32ND_OR_SHORTER:
            return 32
        if codes & smufl.FLAGS_16TH:
            return 16
        if codes & smufl.FLAGS_8TH:
            return 8

        beam = holder
        while beam is not None and "beam" not in _classes(beam):
            beam = self.parent.get(beam)

        if beam is None:
            # Unbeamed, or a cross-staff beam drawn outside the note's group
            return _duration_from_levels(_beam_levels(stem, self.global_beams()), 4)

        polygons, notes, broken = self._beam_shape(beam)
        if polygons == 1:
            return 8
        if polygons == notes and not broken:
            return 16
        if polygons == 2 * notes - 1:
            return 32
        local = [pts for pts in map(_poly_points, beam.findall("./svg:polygon", _NS)) if pts]
        return _duration_from_levels(_beam_levels(stem, local), 4)

    def global_beams(self) -> list[Points]:
        if self._global_beams is None:
            self._global_beams = []
            for g in self.root.findall(".//svg:g", _NS):
                if {"beam", "beamSpan"} & set(_classes(g)):
                    for poly in g.findall("./svg:polygon", _NS):
                        pts = _poly_points(poly)
                        if pts:
                            self._global_beams.append(pts)
        return self._global_beams
