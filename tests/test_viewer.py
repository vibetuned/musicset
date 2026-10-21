import os
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from musicset.labels import LabelSet
from musicset.pipeline import BuildConfig, Layout, build_score
from musicset.viewer import Dataset, Viewer


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("viewer")
    (root / "src").mkdir()
    shutil.copy(FIXTURES / "tiny.musicxml", root / "src" / "tiny.musicxml")
    layout = Layout(root / "out")
    layout.create()
    cfg = BuildConfig(
        labels=LabelSet.load("lite"), write_graph=True, write_abc=True, write_kern=True
    )
    build_score((root / "src" / "tiny.musicxml", layout, cfg, 0))
    return layout.root


def test_viewer_shows_boxes_edges_and_target(dataset: Path):
    app = QApplication.instance() or QApplication([])
    window = Viewer(Dataset(dataset, LabelSet.load("lite")))
    page = window.pages
    assert len(page.box_items) == len(window.data.annotations("tiny_page1"))
    assert page.edge_items and all(not i.isVisible() for _, items in page.edge_items for i in items)
    page.edge_boxes[3].setChecked(True)
    assert any(item.isVisible() for kind, items in page.edge_items if kind == 3 for item in items)
    assert page.target_title.text() == "ABC target" and "K:G" in page.target.toPlainText()

    page.class_filter._set_group(next(iter(page.class_filter.group_boxes)), False)
    hidden = page.class_filter.group_boxes[next(iter(page.class_filter.group_boxes))][1]
    assert all(not rect.isVisible() for cls, (rect, _) in page.box_items if cls in hidden)

    window.tabs.setCurrentIndex(1)
    structure = window.structure
    assert structure.subgraph.number_of_nodes() > 0
    structure.min_degree.setValue(3)
    degree = dict(structure.graph.degree())
    assert structure.subgraph.number_of_nodes() > 0
    assert all(degree[n] >= 3 for n in structure.subgraph.nodes)
    assert not window.grab().isNull()
    app.processEvents()


def test_augment_tab_renders_three_panels(dataset: Path, tmp_path: Path):
    from PIL import Image

    scan = tmp_path / "scan.png"
    Image.new("RGB", (400, 600), "white").save(scan)
    app = QApplication.instance() or QApplication([])
    window = Viewer(Dataset(dataset, LabelSet.load("lite"), [scan]))
    window.tabs.setCurrentIndex(2)
    tab = window.augment
    tab.preset.setCurrentText("engraved")
    tab.scan.setCurrentIndex(1)
    tab.refresh()
    assert len(tab.scenes[0].items()) > 1 and len(tab.scenes[1].items()) > 1
    assert len(tab.scenes[2].items()) == 1  # the scan, no boxes
    assert tab.steps.text() and "clean" in tab.stats.text()
    tab.vary.setChecked(True)
    tab.refresh()
    assert tab.titles[0].text().startswith("clean: ")
    app.processEvents()


def test_structure_nodes_are_glyphs_with_labels(dataset: Path):
    from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsSimpleTextItem

    app = QApplication.instance() or QApplication([])
    window = Viewer(Dataset(dataset, LabelSet.load("lite")))
    window.tabs.setCurrentIndex(1)
    tab = window.structure
    assert tab.items and tab.edges

    note = next(item for item in tab.items.values() if item.cls == "note")
    children = note.childItems()
    assert any(isinstance(c, QGraphicsPathItem) and not c.path().isEmpty() for c in children)
    labels = [c.text() for c in children if isinstance(c, QGraphicsSimpleTextItem)]
    assert labels == ["note"]

    tab.labels.setCurrentText("id")
    note = tab.items[note.node_id]
    assert [c.text() for c in note.childItems() if isinstance(c, QGraphicsSimpleTextItem)] == [
        note.node_id
    ]

    # hover keeps a node and its neighbours, dims the rest
    tab.highlight(note)
    near = {
        note.node_id,
        *tab.subgraph.successors(note.node_id),
        *tab.subgraph.predecessors(note.node_id),
    }
    assert all((item.opacity() == 1.0) == (nid in near) for nid, item in tab.items.items())
    tab.highlight(None)

    # dragging a node carries its edges, and survives a redraw
    edge = note.edges[0]
    start = edge.path().boundingRect()
    note.setPos(note.pos().x() + 200, note.pos().y())
    assert edge.path().boundingRect() != start
    moved = note.pos()
    tab.node_style.setCurrentText("dot")
    assert tab.items[note.node_id].pos() == moved

    for layout in ("tree", "radial", "spring", "page", "ForceAtlas2"):
        tab.layout_choice.setCurrentText(layout)
        assert len(tab.items) == tab.subgraph.number_of_nodes()
    tab.node_style.setCurrentText("crop")
    assert tab.items
    app.processEvents()


def test_layout_changes_and_drags_update_positions_and_scroll_area(dataset: Path):
    from PySide6.QtCore import QPointF

    app = QApplication.instance() or QApplication([])
    window = Viewer(Dataset(dataset, LabelSet.load("lite")))
    window.resize(1000, 700)
    window.show()
    window.tabs.setCurrentIndex(1)
    tab = window.structure
    tab.layout_choice.setCurrentText("ForceAtlas2")
    node_id = next(iter(tab.items))

    def where() -> dict:
        return {n: (round(i.pos().x(), 3), round(i.pos().y(), 3)) for n, i in tab.items.items()}

    # re-running the layout gives a new arrangement and forgets drags
    first = where()
    tab.items[node_id].setPos(5000, 5000)
    tab.relayout()
    assert where() != first and tab.items[node_id].pos() != QPointF(5000, 5000)

    # another layout replaces every position, dragged nodes included
    tab.items[node_id].setPos(-3000, 7000)
    tab.layout_choice.setCurrentText("page")
    page = where()
    expected = {n: (round(p.x(), 3), round(p.y(), 3)) for n, p in tab._layout(tab.subgraph).items()}
    assert page == expected
    tab.layout_choice.setCurrentText("ForceAtlas2")
    assert where() != page

    # a node dragged out stays reachable: the scene grows and the scrollbars with it
    tab.view.zoomed = True
    tab.view.resetTransform()
    tab.view.scale(2.0, 2.0)
    app.processEvents()
    before = tab.view.horizontalScrollBar().maximum()
    item = tab.items[node_id]
    item.setPos(tab.scene.sceneRect().right() + 2000, item.pos().y())
    app.processEvents()
    assert tab.scene.sceneRect().contains(item.sceneBoundingRect())
    assert tab.view.horizontalScrollBar().maximum() > before
    tab.view.centerOn(item)
    app.processEvents()
    assert tab.view.mapToScene(tab.view.viewport().rect()).boundingRect().contains(item.pos())


def test_kern_tab_engraves_ground_truth_and_prediction(dataset: Path, tmp_path: Path):
    import time

    pred = tmp_path / "pred"
    pred.mkdir()
    shutil.copy(dataset / "kern" / "tiny.krn", pred / "tiny.krn")
    app = QApplication.instance() or QApplication([])
    window = Viewer(Dataset(dataset, LabelSet.load("lite"), pred_kern=pred))
    window.tabs.setCurrentIndex(window.tabs.indexOf(window.kern))
    kern = window.kern
    assert kern.score == "tiny"
    assert kern.columns["source"].scene.items()
    assert "100.0%" in kern.status.text() and "1.00x" in kern.status.text()

    deadline = time.monotonic() + 120
    engraved = lambda key: any(  # noqa: E731
        i.type() == 7 for i in kern.columns[key].scene.items()  # QGraphicsPixmapItem
    )
    while not (engraved("gt") and engraved("pred")) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert engraved("gt") and engraved("pred")
    assert kern.texts["gt"].toPlainText() == (dataset / "kern" / "tiny.krn").read_text()

    kern.toggles["gt"].setChecked(False)
    assert kern.columns["gt"].isHidden()
    assert window.order.count() == 3  # page order, worst ground truth, worst predicted
    window.order.setCurrentIndex(2)
    assert sorted(window.data.stems) == ["tiny_page1"]
