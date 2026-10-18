import json
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES

from musicset.cli import main
from musicset.labels import LabelSet
from musicset.pipeline import BuildConfig, Layout, Rejected, build_score
from musicset.render.options import RenderOptions

LITE = LabelSet.load("lite")


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(FIXTURES / "tiny.musicxml", src / "tiny.musicxml")
    return src


def _build(sources: Path, layout: Layout, cfg: BuildConfig):
    layout.create()
    return build_score((sources / "tiny.musicxml", layout, cfg, 0))


def test_build_writes_every_artifact(sources: Path, tmp_path: Path):
    layout = Layout(tmp_path / "out")
    cfg = BuildConfig(labels=LITE, write_graph=True, write_paginated=True, write_abc=True)
    report = _build(sources, layout, cfg)
    assert (report.status, report.pages) == ("built", 1)
    for path in (
        layout.injected / "tiny.mxl",
        layout.svgs / "tiny_page1.svg",
        layout.imgs / "tiny_page1.png",
        layout.pages / "tiny_page1.json.gz",
        layout.annotations / "tiny_page1.json",
        layout.graphs / "tiny.json",
        layout.mei / "tiny.mei",
        layout.paginated / "tiny_page1.mxl",
        layout.abcs / "tiny_page1.abc",
    ):
        assert path.exists(), path

    record = json.loads((layout.annotations / "tiny_page1.json").read_text())
    assert (record["image"], record["width"], record["height"]) == ("tiny_page1.png", 2100, 2970)
    classes = [a["class"] for a in record["annotations"]]
    assert classes.count("noteheadBlack") == 8 and classes.count("stem8") == 2
    # MEI ids are the SVG ids: the graph groundtruth joins on them (noteheads are
    # owned by their note; stems exist only in the engraving, not in MEI)
    mei = (layout.mei / "tiny.mei").read_text()
    heads = [a["id"] for a in record["annotations"] if a["class"] == "noteheadBlack"]
    assert all(f'xml:id="{i}"' in mei for i in heads)

    graph = json.loads((layout.graphs / "tiny.json").read_text())
    assert len(graph["edge_index"][0]) == len(graph["edge_index"][1]) == len(graph["y"]) > 0
    assert max(max(graph["edge_index"][0]), max(graph["edge_index"][1])) < graph["num_nodes"]
    assert set(graph["y"]) <= {1, 2, 3, 4, 5} and 3 in graph["y"]
    assert graph["json_files"] == ["tiny_page1.json"]

    abc = (layout.abcs / "tiny_page1.abc").read_text()
    assert "K:G" in abc and "M:4/4" in abc
    manifest = json.loads((layout.manifest / "tiny.json").read_text())
    assert manifest["pages"][0]["abc"] is True
    assert manifest["labels"] == "lite"


def test_rerun_skips_and_label_change_reprojects(sources: Path, tmp_path: Path):
    layout = Layout(tmp_path / "out")
    _build(sources, layout, BuildConfig(labels=LITE, write_graph=True))
    assert _build(sources, layout, BuildConfig(labels=LITE, write_graph=True)).status == "skipped"

    svg_mtime = (layout.svgs / "tiny_page1.svg").stat().st_mtime_ns
    notes_only = tmp_path / "notes-only"
    notes_only.mkdir()
    (notes_only / "config.json").write_text(json.dumps({"target_classes": ["noteheadBlack"]}))
    (notes_only / "smufl_mapping.json").write_text(json.dumps({"E0A4": "noteheadBlack"}))
    # same outputs (the graph build kept MEI), only the label set differs
    cfg = BuildConfig(labels=LabelSet.load(notes_only), write_mei=True)
    assert _build(sources, layout, cfg).status == "reprojected"
    assert (layout.svgs / "tiny_page1.svg").stat().st_mtime_ns == svg_mtime
    record = json.loads((layout.annotations / "tiny_page1.json").read_text())
    assert {a["class"] for a in record["annotations"]} == {"noteheadBlack"}


def test_render_change_rebuilds(sources: Path, tmp_path: Path):
    layout = Layout(tmp_path / "out")
    _build(sources, layout, BuildConfig())
    report = _build(sources, layout, BuildConfig(raster_scale=0.5))
    assert report.status == "built"
    manifest = json.loads((layout.manifest / "tiny.json").read_text())
    assert manifest["pages"] == [{"page": 1, "image_size": [1050, 1485]}]


def test_graph_needs_node_roles():
    with pytest.raises(ValueError, match="node roles"):
        BuildConfig(write_graph=True)


def test_page_limit_rejects_and_cleans_up(sources: Path, tmp_path: Path):
    # 40 copies of the measure on short pages engrave to several pages
    xml = (sources / "tiny.musicxml").read_text()
    start, end = xml.index("<measure"), xml.index("</part>")
    (sources / "tiny.musicxml").write_text(xml[:start] + xml[start:end] * 40 + xml[end:])
    layout = Layout(tmp_path / "out")
    with pytest.raises(Rejected):
        _build(sources, layout, BuildConfig(max_pages=1, render=RenderOptions(page_height=800)))
    assert not (layout.injected / "tiny.mxl").exists()


def test_density_gate_rejects_and_cleans_up(sources: Path, tmp_path: Path):
    layout = Layout(tmp_path / "out")
    with pytest.raises(Rejected, match="notes per measure"):
        _build(sources, layout, BuildConfig(max_notes_per_measure=5))
    assert not any(layout.svgs.iterdir()) and not any(layout.imgs.iterdir())
    assert not (layout.injected / "tiny.mxl").exists()


def test_cli_build_detection_and_graph(sources: Path, tmp_path: Path):
    out = tmp_path / "out"
    main(["build", str(sources), str(out), "--workers", "1", "--no-png", "--labels", "lite"])
    assert (out / "failures.jsonl").read_text() == ""
    main(["detection", str(out / "pages"), str(tmp_path / "det"), "--labels", "lite-pruned"])
    pruned = json.loads((tmp_path / "det" / "tiny_page1.json").read_text())
    assert pruned["annotations"] and "fing" not in {a["class"] for a in pruned["annotations"]}

    main(["build", str(sources), str(out), "--workers", "1", "--labels", "lite", "--graph"])
    graph_dir = tmp_path / "g"
    main(["graph", str(out / "mei"), str(out / "annotations"), str(graph_dir), "--labels", "lite"])
    standalone = json.loads((graph_dir / "tiny.json").read_text())
    assert standalone == json.loads((out / "graphs" / "tiny.json").read_text())


def test_sources_inside_the_dataset_are_not_mixed_with_outputs(tmp_path: Path):
    dataset = tmp_path / "data"
    (dataset / "sources").mkdir(parents=True)
    shutil.copy(FIXTURES / "tiny.musicxml", dataset / "sources" / "tiny.musicxml")
    for _ in range(2):  # the rerun must not pick up injected/ or paginated/ files
        main(["build", str(dataset), str(dataset), "--workers", "1", "--paginated", "--no-png"])
    assert sorted(p.name for p in (dataset / "manifest").iterdir()) == ["tiny.json"]


def test_cli_augment_copies_pages_and_graphs(sources: Path, tmp_path: Path):
    out = tmp_path / "out"
    main(["build", str(sources), str(out), "--workers", "1", "--labels", "lite", "--graph"])
    aug = tmp_path / "aug"
    main(["augment", str(out), str(aug), "--copies", "2", "--preset", "engraved", "--workers", "1"])
    for k in (1, 2):
        record = json.loads((aug / "annotations" / f"tiny__aug{k}_page1.json").read_text())
        assert record["image"] == f"tiny__aug{k}_page1.png" and record["augment"]["steps"]
        assert record["annotations"]
        assert (aug / "imgs" / f"tiny__aug{k}_page1.png").exists()
        graph = json.loads((aug / "graphs" / f"tiny__aug{k}.json").read_text())
        assert graph["json_files"] == [f"tiny__aug{k}_page1.json"]


def test_pages_with_staves_off_the_page_are_rejected(tiny_page):
    import copy

    from musicset.pipeline import _overflow

    assert _overflow(tiny_page) is None
    page = copy.deepcopy(tiny_page)
    staff = next(e for e in page.elements if e.classes[:1] == ("staff",))
    assert staff.bbox is not None
    staff.bbox = (staff.bbox[0], staff.bbox[1], staff.bbox[2], page.height + 50)
    assert "runs off the page" in (_overflow(page) or "")
