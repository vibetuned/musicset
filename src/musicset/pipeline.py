"""Dataset builds: score sources in, rendered pages and annotations out.

Output layout (the directory names kakigori and gelato already read)::

    OUT/injected/<score>.mxl              source with stable element ids
    OUT/svgs/<score>_page<N>.svg          engraved pages
    OUT/imgs/<score>_page<N>.png          rasterized pages
    OUT/pages/<score>_page<N>.json.gz     canonical annotation (musicset.annotate)
    OUT/annotations/<score>_page<N>.json  detection view (with a label set)
    OUT/graphs/<score>.json               graph view (with a label set's node roles)
    OUT/kern/<score>.krn                  kern view, read off the annotations and graph
    OUT/mei/<score>.mei                   MEI groundtruth
    OUT/paginated/<score>_page<N>.mxl     page view: the measures drawn on the page
    OUT/abcs/<score>_page<N>.abc          page view as ABC
    OUT/manifest/<score>.json             what was built, with which settings

A score's manifest is written last, so its presence means the score finished.
A rerun skips scores whose manifest matches the current settings and rebuilds
those rendered with other settings. When only the label set changed, the
detection and graph views are re-projected from the stored canonical pages
and MEI, without rendering again.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from musicset.annotate.extract import annotate_svg
from musicset.annotate.schema import SCHEMA_VERSION, PageAnnotation
from musicset.ids import id_prefix, inject_ids
from musicset.labels import LabelSet
from musicset.render.engrave import Engraving, rasterize
from musicset.render.options import RenderOptions
from musicset.render.profiles import has_encoded_breaks, sample_render_options
from musicset.score import MusicXML, read_musicxml, write_mxl
from musicset.views.abc import AbcError, to_abc
from musicset.views.detection import ClassMap, detection_record
from musicset.views.graph import build_edges, graph_record
from musicset.views.kern import score_kern
from musicset.views.paginate import page_musicxml

# Bump when extraction output changes for the same SVG, to invalidate builds
EXTRACTOR_VERSION = 3

PAGE_SUFFIX = ".json.gz"


class Rejected(Exception):
    """The score is valid but excluded by the build's filters."""


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path

    @property
    def injected(self) -> Path:
        return self.root / "injected"

    @property
    def svgs(self) -> Path:
        return self.root / "svgs"

    @property
    def imgs(self) -> Path:
        return self.root / "imgs"

    @property
    def pages(self) -> Path:
        return self.root / "pages"

    @property
    def annotations(self) -> Path:
        return self.root / "annotations"

    @property
    def graphs(self) -> Path:
        return self.root / "graphs"

    @property
    def kern(self) -> Path:
        return self.root / "kern"

    @property
    def mei(self) -> Path:
        return self.root / "mei"

    @property
    def paginated(self) -> Path:
        return self.root / "paginated"

    @property
    def abcs(self) -> Path:
        return self.root / "abcs"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest"

    def page_dirs(self) -> tuple[Path, ...]:
        """Directories holding one file per page."""
        return self.svgs, self.imgs, self.pages, self.annotations, self.paginated, self.abcs

    def create(self) -> None:
        for d in (
            self.injected,
            self.graphs,
            self.kern,
            self.mei,
            self.manifest,
            *self.page_dirs(),
        ):
            d.mkdir(parents=True, exist_ok=True)


def page_stem(score: str, page: int) -> str:
    return f"{score}_page{page}"


def page_file_stem(path: Path) -> str:
    """``x_page1.json.gz`` -> ``x_page1``."""
    return path.name.removesuffix(PAGE_SUFFIX)


@dataclass(frozen=True, slots=True)
class BuildConfig:
    render: RenderOptions = field(default_factory=RenderOptions)
    raster_scale: float = 1.0
    write_svg: bool = True
    write_png: bool = True
    write_mei: bool = False
    write_paginated: bool = False
    write_abc: bool = False
    labels: LabelSet | None = None
    write_graph: bool = False
    """Graph groundtruth; needs a label set with node roles, and keeps the MEI."""
    write_kern: bool = False
    """**kern serialized from the annotations and graph; needs node roles, keeps the MEI."""
    max_pages: int = 0
    """Reject scores engraving to more pages than this (0: no limit)."""
    max_measures_per_system: float = 0
    """Reject layouts cramming more measures per system (0: no limit). Sources
    without line breaks engrave as one unreadable smear; kakigori used 15."""
    max_notes_per_measure: float = 0
    """Reject pages averaging more notes per measure (0: no limit). Sources
    without barlines engrave as one giant measure; kakigori used 40."""
    profile: str = "fixed"
    """``fixed``: every score engraved with ``render``. ``varied``: options
    sampled per score and variant (font, staff size, line weights, spacing,
    page), reproducibly from ``seed`` (:mod:`musicset.render.profiles`)."""
    seed: int = 0
    variants: int = 1
    """Engravings per source; variant k of score x is named ``x__v<k>``."""

    def __post_init__(self) -> None:
        if self.profile not in ("fixed", "varied"):
            raise ValueError(f"unknown render profile {self.profile!r}: fixed or varied")
        if self.variants > 1 and self.profile == "fixed":
            raise ValueError("several variants of a score need the varied render profile")
        if (self.write_graph or self.write_kern) and (
            self.labels is None or not self.labels.has_graph_roles
        ):
            raise ValueError("graphs and kern need a label set with node roles (structure.json)")

    @property
    def needs_edges(self) -> bool:
        return self.write_graph or self.write_kern

    @property
    def keeps_mei(self) -> bool:
        return self.write_mei or self.needs_edges

    def score_id(self, source: Path, variant: int) -> str:
        return source.stem if self.variants == 1 else f"{source.stem}__v{variant}"

    def options_for(self, score: str, musicxml: str) -> RenderOptions:
        """The engraving options of one score (and variant)."""
        if self.profile == "fixed":
            return self.render
        rng = random.Random(f"{self.seed}:{score}")
        return sample_render_options(rng, self.render, encoded_breaks=has_encoded_breaks(musicxml))

    def render_fingerprint(self) -> str:
        payload = json.dumps(
            [
                self.render.fingerprint(),
                self.profile,
                self.seed,
                self.raster_scale,
                self.write_svg,
                self.write_png,
                self.keeps_mei,
                self.write_paginated,
                self.write_abc,
                self.max_pages,
                self.max_measures_per_system,
                self.max_notes_per_measure,
                EXTRACTOR_VERSION,
                SCHEMA_VERSION,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def labels_fingerprint(self) -> str | None:
        if self.labels is None:
            return None
        return f"{self.labels.fingerprint()}-{int(self.write_graph)}{int(self.write_kern)}"


@dataclass(slots=True)
class ScoreReport:
    score: str
    status: str
    """``built``, ``reprojected`` or ``skipped``."""
    pages: int


def _write_json(path: Path, data: Any, *, indent: int | None = None) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    tmp.replace(path)


def _density_violation(page: PageAnnotation, cfg: BuildConfig) -> str | None:
    kinds = [el.classes for el in page.elements]
    systems = kinds.count(("system",))
    measures = kinds.count(("measure",))
    notes = kinds.count(("note",))
    limit = cfg.max_measures_per_system
    if limit and systems and measures / systems > limit:
        return f"page {page.page}: {measures / systems:.0f} measures per system > {limit}"
    limit = cfg.max_notes_per_measure
    if limit and measures and notes / measures > limit:
        return f"page {page.page}: {notes / measures:.0f} notes per measure > {limit}"
    return None


def _remove_pages(layout: Layout, score: str) -> None:
    pattern = re.compile(rf"{re.escape(score)}_page\d+(\.json\.gz|\.\w+)")
    for d in layout.page_dirs():
        for path in d.glob(f"{score}_page*"):
            if pattern.fullmatch(path.name):
                path.unlink()


def _project_labels(
    layout: Layout,
    cfg: BuildConfig,
    score: str,
    pages: list[PageAnnotation],
    sizes: list[tuple[int, int]],
    mei: str | None,
) -> dict[str, Any]:
    """Detection view of every page, then the score's graph and kern.

    Returns manifest fields (a kern serialization failure is recorded, not raised).
    """
    assert cfg.labels is not None
    records = []
    for page, size in zip(pages, sizes, strict=True):
        stem = page_stem(score, page.page)
        record = detection_record(page, cfg.labels.classes, f"{stem}.png", size)
        _write_json(layout.annotations / f"{stem}.json", record, indent=2)
        records.append(record)

    info: dict[str, Any] = {}
    if not cfg.needs_edges:
        return info
    assert mei is not None
    annotations = [r["annotations"] for r in records]
    edges = build_edges(mei, annotations, cfg.labels.node_roles)
    if cfg.write_graph:
        graph = graph_record(
            edges,
            mei_file=f"{score}.mei",
            json_files=[f"{page_stem(score, p.page)}.json" for p in pages],
        )
        _write_json(layout.graphs / f"{score}.json", graph)
    if cfg.write_kern:
        try:
            kern = score_kern(annotations, edges, cfg.labels.node_roles)
            (layout.kern / f"{score}.krn").write_text(kern, encoding="utf-8")
            info["kern"] = True
        except Exception as e:
            # the serializer is heuristic: one unreadable score must not lose the rest
            (layout.kern / f"{score}.krn").unlink(missing_ok=True)
            info["kern"] = False
            info["kern_error"] = f"{type(e).__name__}: {e}"[:500]
    return info


def build_score(task: tuple[Path, Layout, BuildConfig, int]) -> ScoreReport:
    """Build every artifact of one score. Runs in an isolated worker process."""
    source, layout, cfg, variant = task
    score = cfg.score_id(source, variant)
    manifest_path = layout.manifest / f"{score}.json"
    render_fp = cfg.render_fingerprint()
    labels_fp = cfg.labels_fingerprint()

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("render_fingerprint") == render_fp:
            if cfg.labels is None or manifest.get("labels_fingerprint") == labels_fp:
                return ScoreReport(score, "skipped", len(manifest["pages"]))
            infos = manifest["pages"]
            pages = [
                PageAnnotation.load(layout.pages / f"{page_stem(score, i['page'])}{PAGE_SUFFIX}")
                for i in infos
            ]
            mei_path = layout.mei / f"{score}.mei"
            mei = mei_path.read_text(encoding="utf-8") if cfg.needs_edges else None
            sizes = [(i["image_size"][0], i["image_size"][1]) for i in infos]
            manifest.pop("kern", None)
            manifest.pop("kern_error", None)
            manifest.update(_project_labels(layout, cfg, score, pages, sizes, mei))
            manifest["labels"] = cfg.labels.name
            manifest["labels_fingerprint"] = labels_fp
            _write_json(manifest_path, manifest, indent=2)
            return ScoreReport(score, "reprojected", len(infos))
        manifest_path.unlink()

    _remove_pages(layout, score)
    musicxml = read_musicxml(source)
    musicxml.text = inject_ids(musicxml.text, id_prefix(score))
    write_mxl(layout.injected / f"{score}.mxl", musicxml)

    options = cfg.options_for(score, musicxml.text)
    engraving = Engraving.load(musicxml.text, options)
    if cfg.max_pages and engraving.page_count > cfg.max_pages:
        (layout.injected / f"{score}.mxl").unlink()
        raise Rejected(f"{engraving.page_count} pages > {cfg.max_pages}")

    pages: list[PageAnnotation] = []
    sizes: list[tuple[int, int]] = []
    infos: list[dict[str, Any]] = []
    for number, svg in engraving.pages():
        stem = page_stem(score, number)
        info: dict[str, Any] = {"page": number}
        if cfg.write_svg:
            (layout.svgs / f"{stem}.svg").write_text(svg, encoding="utf-8")

        # Rasterize first: text boxes are measured while cairo lays the text out
        raster = rasterize(svg, cfg.raster_scale)
        if cfg.write_png:
            (layout.imgs / f"{stem}.png").write_bytes(raster.png)
        size = raster.size
        info["image_size"] = list(size)

        page = annotate_svg(
            svg, score=score, page=number, text_ink=raster.text_ink, ink_scale=raster.scale
        )
        page.meta["render_fingerprint"] = render_fp
        if (violation := _density_violation(page, cfg)) is not None:
            _remove_pages(layout, score)
            (layout.injected / f"{score}.mxl").unlink()
            raise Rejected(violation)
        page.save(layout.pages / f"{stem}{PAGE_SUFFIX}")

        if cfg.write_paginated or cfg.write_abc:
            page_xml = page_musicxml(musicxml.text, page.ids())
            if cfg.write_paginated:
                sliced = MusicXML(page_xml, name=musicxml.name, container=musicxml.container)
                write_mxl(layout.paginated / f"{stem}.mxl", sliced)
            if cfg.write_abc:
                try:
                    abc = to_abc(page_xml)
                    (layout.abcs / f"{stem}.abc").write_text(abc, encoding="utf-8")
                    info["abc"] = True
                except AbcError as e:
                    info["abc"] = False
                    info["abc_error"] = str(e)[:500]

        pages.append(page)
        sizes.append(size)
        infos.append(info)

    mei = engraving.mei() if cfg.keeps_mei else None
    if mei is not None:
        (layout.mei / f"{score}.mei").write_text(mei, encoding="utf-8")
    views = _project_labels(layout, cfg, score, pages, sizes, mei) if cfg.labels else {}

    manifest = {
        "score": score,
        "source": str(source),
        "pages": infos,
        "renderer": engraving.version,
        "render_profile": cfg.profile,
        "render_options": asdict(options),
        "raster_scale": cfg.raster_scale,
        "render_fingerprint": render_fp,
        "labels": cfg.labels.name if cfg.labels else None,
        "labels_fingerprint": labels_fp,
        **views,
    }
    _write_json(manifest_path, manifest, indent=2)
    return ScoreReport(score, "built", len(infos))


def annotate_file(task: tuple[Path, Path]) -> int:
    """Canonical annotation of an existing SVG page; returns the element count."""
    svg_path, out_dir = task
    stem = svg_path.stem
    match = re.fullmatch(r"(.+)_page(\d+)", stem)
    score, number = (match.group(1), int(match.group(2))) if match else (stem, 1)
    svg = svg_path.read_text(encoding="utf-8")
    raster = rasterize(svg)
    page = annotate_svg(svg, score=score, page=number, text_ink=raster.text_ink)
    page.save(out_dir / f"{stem}{PAGE_SUFFIX}")
    return len(page.elements)


def detection_file(task: tuple[Path, Path, ClassMap, tuple[int, int] | None]) -> int:
    """Detection view of a stored canonical page; returns the box count."""
    page_path, out_dir, class_map, size = task
    page = PageAnnotation.load(page_path)
    if size is None:
        size = (round(page.pixel_width), round(page.pixel_height))
    stem = page_file_stem(page_path)
    record = detection_record(page, class_map, f"{stem}.png", size)
    _write_json(out_dir / f"{stem}.json", record, indent=2)
    return len(record["annotations"])


def graph_file(task: tuple[Path, list[Path], Path, dict[str, list[str]]]) -> int:
    """Graph view from an MEI file and its pages' detection files; returns the edge count."""
    mei_path, annotation_paths, out_dir, node_roles = task
    pages = [json.loads(p.read_text(encoding="utf-8"))["annotations"] for p in annotation_paths]
    edges = build_edges(mei_path, pages, node_roles)
    record = graph_record(
        edges, mei_file=mei_path.name, json_files=[p.name for p in annotation_paths]
    )
    _write_json(out_dir / f"{mei_path.stem}.json", record)
    return len(edges)


def kern_file(task: tuple[Path, list[Path], Path, dict[str, list[str]]]) -> int:
    """Kern view from an MEI file and its pages' detection files; returns the line count."""
    mei_path, annotation_paths, out_dir, node_roles = task
    pages = [json.loads(p.read_text(encoding="utf-8"))["annotations"] for p in annotation_paths]
    kern = score_kern(pages, build_edges(mei_path, pages, node_roles), node_roles)
    (out_dir / f"{mei_path.stem}.krn").write_text(kern, encoding="utf-8")
    return kern.count("\n")


def augment_page(task: tuple[Path, Path, Path, str, float, int, int]) -> int:
    """Augmented copies of one page and its detection file; returns copies written.

    Copy k of ``x_page3`` is ``x__aug<k>_page3`` (png + json, boxes moved with
    the ink, boxes pushed off the page dropped, the applied steps logged).
    """
    import io
    import zlib

    import numpy as np
    from PIL import Image

    from musicset.augment import Augmenter

    img_path, ann_path, out_root, preset, strength, seed, copies = task
    record = json.loads(ann_path.read_text(encoding="utf-8"))
    anns = record["annotations"]
    boxes = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
    with Image.open(img_path) as im:
        image = np.asarray(im.convert("RGB"))
    score, _, number = img_path.stem.rpartition("_page")
    augmenter = Augmenter(preset, strength)
    for k in range(1, copies + 1):
        rng = np.random.default_rng([seed, zlib.crc32(img_path.stem.encode()), k])
        out = augmenter(image, boxes, rng)
        stem = f"{score}__aug{k}_page{number}"
        buf = io.BytesIO()
        Image.fromarray(out.image).save(buf, format="PNG")
        (out_root / "imgs" / f"{stem}.png").write_bytes(buf.getvalue())
        kept = [
            {**a, "bbox": [round(float(v), 2) for v in b]}
            for a, b, keep in zip(anns, out.boxes, out.keep, strict=True)
            if keep
        ]
        h, w = out.image.shape[:2]
        result = {
            "image": f"{stem}.png",
            "width": w,
            "height": h,
            "annotations": kept,
            "augment": {"source": img_path.name, "preset": out.preset, "steps": out.log},
        }
        _write_json(out_root / "annotations" / f"{stem}.json", result, indent=2)
    return copies


def augmented_graphs(dataset: Path, out_root: Path, copies: int) -> int:
    """Graph copies for augmented pages: same edges, page files renamed."""
    written = 0
    for path in sorted((dataset / "graphs").glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        for k in range(1, copies + 1):
            renamed = [
                re.sub(r"_page(\d+)\.json$", rf"__aug{k}_page\1.json", f)
                for f in graph.get("json_files", [])
            ]
            if not all((out_root / "annotations" / f).exists() for f in renamed):
                continue
            _write_json(
                out_root / "graphs" / f"{path.stem}__aug{k}.json", {**graph, "json_files": renamed}
            )
            written += 1
    return written
