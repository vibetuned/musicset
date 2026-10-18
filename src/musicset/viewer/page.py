"""Page tab: the page image with boxes, graph edges and its text target."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from musicset.viewer.data import EDGE_COLORS, UNKNOWN_EDGE_COLOR, Dataset, score_of
from musicset.viewer.widgets import ClassFilter, ZoomView, _arrow
from musicset.views.graph import EDGE_TYPES


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
