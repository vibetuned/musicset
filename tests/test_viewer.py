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
    cfg = BuildConfig(labels=LabelSet.load("lite"), write_graph=True, write_abc=True)
    build_score((root / "src" / "tiny.musicxml", layout, cfg))
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
