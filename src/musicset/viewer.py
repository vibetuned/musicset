"""Dataset viewer (``musicset view``), from the kakigori visualizers.

* **Page**: the page image with its detection boxes (classes grouped by the
  label set, every class its own colour), the graph edges between them
  (toggle per edge type, arrows point from parent to child) and the page's
  text target (ABC, else per-page MusicXML).
* **Structure**: the score's graph as a force-directed layout, filtered by
  connected component, degree and size, exportable as JSON.

Keys: ←/→ or A/D change page, Ctrl+F jumps to a page by name.
Needs the ``viz`` extra (PySide6, networkx).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import networkx as nx
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from musicset.labels import LabelSet
from musicset.score import read_musicxml
from musicset.views.graph import EDGE_TYPES, graph_nodes, page_sort_key

EDGE_COLORS = {1: "#e6194b", 2: "#3cb44b", 3: "#4363d8", 4: "#f58231", 5: "#42d4f4"}
UNKNOWN_EDGE_COLOR = "#9aa0a6"


# -- data -------------------------------------------------------------------


class Dataset:
    """A dataset directory as built by ``musicset build``."""

    def __init__(self, root: Path, labels: LabelSet | None) -> None:
        self.root = root
        self.labels = labels
        imgs = root / "imgs"
        self.stems: list[str] = sorted((p.stem for p in imgs.glob("*.png")), key=page_sort_key)
        self.by_score: dict[str, list[str]] = defaultdict(list)
        for stem in self.stems:
            self.by_score[score_of(stem)].append(stem)

    def image(self, stem: str) -> Path:
        return self.root / "imgs" / f"{stem}.png"

    @cache  # noqa: B019 - one dataset per viewer process
    def annotations(self, stem: str) -> list[dict[str, Any]]:
        path = self.root / "annotations" / f"{stem}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))["annotations"]

    @cache  # noqa: B019
    def graph(self, score: str) -> dict[str, Any] | None:
        path = self.root / "graphs" / f"{score}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @cache  # noqa: B019
    def nodes(self, score: str) -> dict[str, dict[str, Any]]:
        """Graph node id -> annotation (with ``_page`` = index in the score's pages)."""
        _, nodes, _ = graph_nodes([self.annotations(s) for s in self.by_score[score]])
        return nodes

    def edges(self, score: str) -> list[tuple[str, str, int]]:
        graph = self.graph(score)
        if graph is None:
            return []
        ids = graph["node_ids"]
        src, dst = graph["edge_index"]
        return [(ids[u], ids[v], kind) for u, v, kind in zip(src, dst, graph["y"], strict=True)]

    def target(self, stem: str) -> tuple[str, str] | None:
        """(kind, text) of the page's text target, if any."""
        abc = self.root / "abcs" / f"{stem}.abc"
        if abc.exists():
            return "ABC", abc.read_text(encoding="utf-8")
        mxl = self.root / "paginated" / f"{stem}.mxl"
        if mxl.exists():
            return "MusicXML", read_musicxml(mxl).text
        return None

    def classes(self) -> list[str]:
        """Label set classes in index order, then any other class found."""
        known = list(self.labels.classes.target_classes) if self.labels else []
        seen = {a["class"] for s in self.stems[:200] for a in self.annotations(s)}
        return known + sorted(seen - set(known))

    def groups(self) -> dict[str, list[str]]:
        """Class groups from the label set; classes outside every group get their own."""
        classes = self.classes()
        groups = {}
        if self.labels:
            for name, members in self.labels.groups.items():
                kept = [c for c in members if c in classes]
                if kept:
                    groups[name] = kept
        grouped = {c for members in groups.values() for c in members}
        rest = [c for c in classes if c not in grouped]
        if rest:
            groups["Other" if groups else "Classes"] = rest
        return groups


def score_of(stem: str) -> str:
    return stem.rsplit("_page", 1)[0]


def class_colors(classes: list[str]) -> dict[str, QColor]:
    """A distinct, stable colour per class: hues spaced by the golden ratio."""
    return {
        c: QColor.fromHsvF((i * 0.618033988749895) % 1.0, 0.85, 0.9) for i, c in enumerate(classes)
    }


# -- widgets ----------------------------------------------------------------


class ZoomView(QGraphicsView):
    """Wheel to zoom, drag to pan; fits the scene until the user zooms."""

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet("background-color: #2b2b2b; border: none;")
        self.zoomed = False

    def wheelEvent(self, event: Any) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.zoomed = True

    def fit(self) -> None:
        self.zoomed = False
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if not self.zoomed and not self.scene().sceneRect().isEmpty():
            self.fit()


def _arrow(scene: QGraphicsScene, a: QPointF, b: QPointF, pen: QPen, size: float) -> list[Any]:
    line = scene.addLine(QLineF(a, b), pen)
    angle = math.atan2(b.y() - a.y(), b.x() - a.x())
    left = QPointF(b.x() - size * math.cos(angle - 0.4), b.y() - size * math.sin(angle - 0.4))
    right = QPointF(b.x() - size * math.cos(angle + 0.4), b.y() - size * math.sin(angle + 0.4))
    head = scene.addPolygon(QPolygonF([b, left, right]), pen, QBrush(pen.color()))
    return [line, head]


class ClassFilter(QScrollArea):
    """Group checkboxes over class checkboxes; groups follow their classes."""

    def __init__(self, groups: dict[str, list[str]], colors: dict[str, QColor], on_change) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.on_change = on_change
        self.boxes: dict[str, QCheckBox] = {}
        self.group_boxes: dict[str, tuple[QCheckBox, list[str]]] = {}
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for name, members in groups.items():
            group = QCheckBox(name)
            group.setTristate(False)
            group.setChecked(True)
            group.setFont(QFont(group.font().family(), -1, QFont.Weight.Bold))
            group.clicked.connect(lambda checked, n=name: self._set_group(n, checked))
            layout.addWidget(group)
            self.group_boxes[name] = (group, members)
            for cls in members:
                box = QCheckBox(cls)
                box.setChecked(True)
                box.setStyleSheet(f"margin-left: 18px; color: {colors[cls].name()};")
                box.toggled.connect(self._changed)
                layout.addWidget(box)
                self.boxes[cls] = box
        self.setWidget(content)

    def _set_group(self, name: str, checked: bool) -> None:
        for cls in self.group_boxes[name][1]:
            box = self.boxes[cls]
            box.blockSignals(True)
            box.setChecked(checked)
            box.blockSignals(False)
        self._changed()

    def _changed(self) -> None:
        for group, members in self.group_boxes.values():
            states = {self.boxes[c].isChecked() for c in members}
            group.blockSignals(True)
            group.setChecked(states == {True})
            group.blockSignals(False)
        self.on_change()

    def visible(self) -> set[str]:
        return {c for c, box in self.boxes.items() if box.isChecked()}


class PageTab(QWidget):
    def __init__(self, data: Dataset, colors: dict[str, QColor]) -> None:
        super().__init__()
        self.data = data
        self.colors = colors
        self.box_items: list[tuple[str, list[Any]]] = []
        self.edge_items: list[tuple[int, list[Any]]] = []

        sidebar = QTabWidget()
        self.class_filter = ClassFilter(data.groups(), colors, self.refresh)
        sidebar.addTab(self.class_filter, "Classes")

        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.show_boxes = QCheckBox("boxes")
        self.show_boxes.setChecked(True)
        self.show_labels = QCheckBox("class names")
        self.show_labels.setChecked(True)
        for box in (self.show_boxes, self.show_labels):
            box.toggled.connect(self.refresh)
            graph_layout.addWidget(box)
        graph_layout.addWidget(QLabel("Edges"))
        self.edge_boxes: dict[int, QCheckBox] = {}
        for kind, name in EDGE_TYPES.items():
            box = QCheckBox(f"{kind} {name}")
            box.setStyleSheet(f"color: {EDGE_COLORS[kind]};")
            box.toggled.connect(self.refresh)
            graph_layout.addWidget(box)
            self.edge_boxes[kind] = box
        self.edge_status = QLabel()
        self.edge_status.setWordWrap(True)
        graph_layout.addWidget(self.edge_status)
        sidebar.addTab(graph_panel, "Graph")

        self.scene = QGraphicsScene()
        self.view = ZoomView(self.scene)
        self.target_title = QLabel()
        self.target = QPlainTextEdit()
        self.target.setReadOnly(True)
        self.target.setFont(QFont("monospace"))
        target_panel = QWidget()
        target_layout = QVBoxLayout(target_panel)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_title)
        target_layout.addWidget(self.target)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.view)
        splitter.addWidget(target_panel)
        splitter.setSizes([280, 900, 360])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def load(self, stem: str) -> None:
        self.scene.clear()
        self.box_items.clear()
        self.edge_items.clear()
        pixmap = QPixmap(str(self.data.image(stem)))
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.scene.addPixmap(pixmap)
        width = max(1.0, pixmap.width() * 0.0012)
        font = QFont("sans-serif")
        font.setPixelSize(max(8, int(pixmap.width() * 0.006)))

        for ann in self.data.annotations(stem):
            cls = ann.get("class", "?")
            x1, y1, x2, y2 = ann["bbox"]
            color = self.colors.get(cls, QColor("#ffffff"))
            pen = QPen(color)
            pen.setWidthF(width)
            rect = self.scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
            label = self.scene.addSimpleText(cls, font)
            label.setBrush(color)
            label.setPos(x1, y1 - font.pixelSize() * 1.3)
            self.box_items.append((cls, [rect, label]))

        self._load_edges(stem, width * 1.5, font.pixelSize() * 0.8)
        self.view.fit()

        target = self.data.target(stem)
        self.target_title.setText(f"{target[0]} target" if target else "no text target")
        self.target.setPlainText(target[1] if target else "")
        self.refresh()

    def _load_edges(self, stem: str, width: float, arrow: float) -> None:
        score = score_of(stem)
        edges = self.data.edges(score)
        if not edges:
            self.edge_status.setText("no graph for this score")
            return
        page_idx = self.data.by_score[score].index(stem)
        nodes = self.data.nodes(score)
        drawn = off_page = 0
        for u, v, kind in edges:
            a, b = nodes.get(u), nodes.get(v)
            if a is None or b is None or a["_page"] != page_idx or b["_page"] != page_idx:
                off_page += 1
                continue
            pen = QPen(QColor(EDGE_COLORS.get(kind, UNKNOWN_EDGE_COLOR)))
            pen.setWidthF(width)
            ca = QPointF((a["bbox"][0] + a["bbox"][2]) / 2, (a["bbox"][1] + a["bbox"][3]) / 2)
            cb = QPointF((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
            self.edge_items.append((kind, _arrow(self.scene, ca, cb, pen, arrow)))
            drawn += 1
        counts = defaultdict(int)
        for _, _, kind in edges:
            counts[kind] += 1
        self.edge_status.setText(
            f"{drawn} edges on this page, {off_page} touch other pages\n"
            + "\n".join(f"type {k}: {n}" for k, n in sorted(counts.items()))
        )

    def refresh(self) -> None:
        visible = self.class_filter.visible()
        boxes, labels = self.show_boxes.isChecked(), self.show_labels.isChecked()
        for cls, (rect, label) in self.box_items:
            shown = boxes and cls in visible
            rect.setVisible(shown)
            label.setVisible(shown and labels)
        for kind, items in self.edge_items:
            box = self.edge_boxes.get(kind)
            for item in items:
                item.setVisible(box is not None and box.isChecked())


class StructureTab(QWidget):
    def __init__(self, data: Dataset, colors: dict[str, QColor]) -> None:
        super().__init__()
        self.data = data
        self.colors = colors
        self.score: str | None = None
        self.components: list[set[str]] = []
        self.subgraph: nx.DiGraph = nx.DiGraph()

        controls = QWidget()
        form = QFormLayout(controls)
        self.component = QComboBox()
        self.min_degree = QSpinBox()
        self.min_degree.setRange(0, 1000)
        self.max_nodes = QSpinBox()
        self.max_nodes.setRange(10, 20000)
        self.max_nodes.setValue(400)
        self.keep_isolated = QCheckBox("keep isolated nodes")
        self.status = QLabel()
        self.status.setWordWrap(True)
        export = QPushButton("Export JSON…")
        export.clicked.connect(self.export)
        form.addRow("component", self.component)
        form.addRow("min degree", self.min_degree)
        form.addRow("max nodes", self.max_nodes)
        form.addRow(self.keep_isolated)
        form.addRow(export)
        form.addRow(self.status)
        for kind, name in EDGE_TYPES.items():
            form.addRow(QLabel(f"<span style='color:{EDGE_COLORS[kind]}'>━━ {kind} {name}</span>"))
        self.component.currentIndexChanged.connect(self.redraw)
        self.min_degree.valueChanged.connect(self.redraw)
        self.max_nodes.valueChanged.connect(self.redraw)
        self.keep_isolated.toggled.connect(self.redraw)

        self.scene = QGraphicsScene()
        self.view = ZoomView(self.scene)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.view)
        splitter.setSizes([280, 1200])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def load(self, score: str) -> None:
        if score == self.score:
            return
        self.score = score
        nodes = self.data.nodes(score)
        graph = nx.DiGraph()
        for u, v, kind in self.data.edges(score):
            graph.add_edge(u, v, kind=kind)
        for node_id in graph.nodes:
            ann = nodes.get(node_id, {})
            graph.nodes[node_id].update(cls=ann.get("class", "?"), page=ann.get("_page", 0))
        self.graph = graph

        def reading_order(component: set[str]) -> tuple[float, float]:
            # Components in page order, by their systems' position when they have any
            anchors = [n for n in component if nodes.get(n, {}).get("class") == "system"]
            anchors = anchors or list(component)
            boxes = [nodes[n] for n in anchors if n in nodes]
            if not boxes:
                return (math.inf, math.inf)
            page = sum(b["_page"] for b in boxes) / len(boxes)
            y = sum((b["bbox"][1] + b["bbox"][3]) / 2 for b in boxes) / len(boxes)
            return (page, y)

        self.components = sorted(nx.weakly_connected_components(graph), key=reading_order)
        self.component.blockSignals(True)
        self.component.clear()
        self.component.addItem(f"all ({graph.number_of_nodes()} nodes)")
        for i, comp in enumerate(self.components, 1):
            self.component.addItem(f"#{i}: {len(comp)} nodes")
        self.component.blockSignals(False)
        self.redraw()

    def redraw(self) -> None:
        self.scene.clear()
        if self.score is None or self.graph.number_of_nodes() == 0:
            self.status.setText("no graph for this score")
            return
        index = self.component.currentIndex()
        keep = set(self.graph.nodes) if index <= 0 else self.components[index - 1]
        sub = self.graph.subgraph(keep)
        # Degrees are measured inside the shown subgraph
        degree = dict(sub.degree())
        keep = {n for n in keep if degree[n] >= self.min_degree.value()}
        if not self.keep_isolated.isChecked():
            keep = {n for n in keep if degree[n] > 0}
        if len(keep) > self.max_nodes.value():
            keep = set(sorted(keep, key=lambda n: -degree[n])[: self.max_nodes.value()])
        self.subgraph = nx.DiGraph(self.graph.subgraph(keep))

        positions = nx.spring_layout(self.subgraph.to_undirected(), seed=0) if keep else {}
        span = 60.0 * math.sqrt(max(1, len(keep)))
        point = {n: QPointF(x * span, y * span) for n, (x, y) in positions.items()}
        for u, v, attrs in self.subgraph.edges(data=True):
            pen = QPen(QColor(EDGE_COLORS.get(attrs["kind"], UNKNOWN_EDGE_COLOR)))
            pen.setWidthF(1.2)
            _arrow(self.scene, point[u], point[v], pen, 6)
        for n in self.subgraph.nodes:
            cls = self.subgraph.nodes[n]["cls"]
            color = self.colors.get(cls, QColor("#ffffff"))
            dot = self.scene.addEllipse(
                point[n].x() - 5, point[n].y() - 5, 10, 10, QPen(color.darker()), QBrush(color)
            )
            dot.setToolTip(f"{n}\n{cls} (page {self.subgraph.nodes[n]['page'] + 1})")
        self.status.setText(
            f"{self.subgraph.number_of_nodes()} nodes, {self.subgraph.number_of_edges()} edges "
            f"shown; {len(self.components)} components in the score"
        )
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        self.view.fit()

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export subgraph", f"{self.score}.json")
        if not path:
            return
        payload = {
            "nodes": [{"id": n, **attrs} for n, attrs in self.subgraph.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, **a} for u, v, a in self.subgraph.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


class Viewer(QMainWindow):
    def __init__(self, data: Dataset) -> None:
        super().__init__()
        self.data = data
        self.index = 0
        self.setWindowTitle(f"musicset — {data.root}")
        self.resize(1600, 950)

        colors = class_colors(data.classes())
        self.pages = PageTab(data, colors)
        self.structure = StructureTab(data, colors)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.pages, "Page")
        self.tabs.addTab(self.structure, "Structure")
        self.tabs.currentChanged.connect(lambda _: self.show_current())

        self.info = QLabel()
        self.jump = QLineEdit()
        self.jump.setPlaceholderText("jump to page (name contains…)")
        self.jump.returnPressed.connect(self.jump_to)
        prev_btn, next_btn = QPushButton("◀ Prev"), QPushButton("Next ▶")
        prev_btn.clicked.connect(lambda: self.step(-1))
        next_btn.clicked.connect(lambda: self.step(1))
        nav = QHBoxLayout()
        nav.addWidget(self.info, 1)
        nav.addWidget(self.jump)
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(nav)
        self.setCentralWidget(central)

        for key, delta in ((Qt.Key.Key_Right, 1), (Qt.Key.Key_D, 1), (Qt.Key.Key_Left, -1)):
            QShortcut(QKeySequence(key), self, lambda d=delta: self.step(d))
        QShortcut(QKeySequence(Qt.Key.Key_A), self, lambda: self.step(-1))
        QShortcut(QKeySequence.StandardKey.Find, self, self.jump.setFocus)
        self.show_current()

    def step(self, delta: int) -> None:
        if self.data.stems:
            self.index = (self.index + delta) % len(self.data.stems)
            self.show_current()

    def jump_to(self) -> None:
        text = self.jump.text().strip()
        for i, stem in enumerate(self.data.stems):
            if text and text in stem:
                self.index = i
                self.show_current()
                return
        self.info.setText(f"no page matches {text!r}")

    def show_current(self) -> None:
        if not self.data.stems:
            self.info.setText(f"no images in {self.data.root / 'imgs'}")
            return
        stem = self.data.stems[self.index]
        self.info.setText(f"{self.index + 1} / {len(self.data.stems)}   {stem}")
        if self.tabs.currentWidget() is self.pages:
            self.pages.load(stem)
        else:
            self.structure.load(score_of(stem))


def run(root: Path, labels: LabelSet | None) -> None:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Viewer(Dataset(root, labels))
    window.show()
    sys.exit(app.exec())
