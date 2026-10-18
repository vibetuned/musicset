"""Structure tab: a score's graph as a network, nodes drawn as their glyphs.

Layouts: ForceAtlas2 (networkx), spring, or the nodes' positions on the page.
Nodes show their SMuFL glyph (or their crop from the page image, or a dot)
with a label; hovering a node highlights its neighbours, dragging moves it
with its edges. Filters by connected component, degree and size; the shown
subgraph exports as JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QWidget,
)

from musicset.viewer.data import EDGE_COLORS, UNKNOWN_EDGE_COLOR, Dataset
from musicset.viewer.widgets import ZoomView, _qimage
from musicset.views.graph import EDGE_TYPES

RADIUS = 16.0
"""Node disc radius, scene units."""

# SMuFL codepoints for classes the label set's glyph map does not cover:
# structural objects drawn by a representative glyph
CLASS_GLYPHS = {
    "note": 0xECA5,  # metNoteQuarterUp: notehead and stem, the whole note
    "chord": 0xECA5,
    "stem4": 0xE210,  # stem
    "stem8": 0xE210,
    "stem16": 0xE210,
    "stem32": 0xE210,
    "ledgerLines": 0xE022,  # legerLine
    "dots": 0xE1E7,  # augmentationDot
    "staff": 0xE014,  # staff5Lines
    "system-staff": 0xE014,
    "mRest": 0xE4EE,  # restHBar
    "fermata": 0xE4C0,
    "trill": 0xE566,
    "mordent": 0xE56D,
    "turn": 0xE567,
    "arpeg": 0xE63C,
    "pedal": 0xE650,
    "octave": 0xE510,
    "tuplet": 0xE883,  # tuplet3
    "tupletNum": 0xE883,
    "keySig": 0xE262,
    "meterSig": 0xE084,
    "grpSym": 0xE000,  # brace
}
DRAWN = {"beam8", "beam16", "beam32", "beamBroken", "slur", "tie"}
MUSIC_FONTS = ("Bravura", "Leipzig", "Leland", "Petaluma", "Gootville")
LABELS = ("class", "id", "class + id", "none")
STYLES = ("glyph", "crop", "dot")
LAYOUTS = ("ForceAtlas2", "spring", "page")


def _music_font() -> str | None:
    families = set(QFontDatabase.families())
    return next((f for f in MUSIC_FONTS if f in families), None)


def _glyph_path(code: int, family: str, box: float) -> QPainterPath:
    """A glyph's outline, centred on the origin and fitting a ``box`` square."""
    font = QFont(family)
    font.setPixelSize(48)
    path = QPainterPath()
    path.addText(0, 0, font, chr(code))
    rect = path.boundingRect()
    if rect.isEmpty():
        return path
    scale = min(1.0, box / max(rect.width(), rect.height()))
    transform = QTransform()
    transform.scale(scale, scale)
    transform.translate(-rect.center().x(), -rect.center().y())
    return transform.map(path)


def _drawn_path(cls: str) -> QPainterPath:
    """Shapes for classes SMuFL has no glyph for: beams and slurs."""
    path = QPainterPath()
    if cls.startswith("beam"):
        for k in range({"beam8": 1, "beam16": 2, "beam32": 3}.get(cls, 2)):
            dy = 5 * k
            path.addPolygon(
                QPolygonF(
                    [
                        QPointF(-10, -2 + dy),
                        QPointF(10, -6 + dy),
                        QPointF(10, -3 + dy),
                        QPointF(-10, 1 + dy),
                    ]
                )
            )
            path.closeSubpath()
    else:  # slur, tie: a thin crescent
        path.moveTo(-11, 3)
        path.quadTo(0, -9, 11, 3)
        path.quadTo(0, -5, -11, 3)
    return path


class NodeItem(QGraphicsItemGroup):
    """A graph node: disc, glyph and label, draggable, highlighting on hover."""

    def __init__(self, node_id: str, cls: str, page: int, tab: StructureTab) -> None:
        super().__init__()
        self.node_id = node_id
        self.cls = cls
        self.tab = tab
        self.edges: list[EdgeItem] = []
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{node_id}\n{cls} (page {page + 1})")

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self.edges:
                edge.refresh()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: Any) -> None:
        self.tab.highlight(self)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: Any) -> None:
        self.tab.highlight(None)
        super().hoverLeaveEvent(event)


class EdgeItem(QGraphicsPathItem):
    """A directed edge from disc rim to disc rim, following its nodes."""

    def __init__(self, source: NodeItem, target: NodeItem, kind: int) -> None:
        super().__init__()
        self.source, self.target, self.kind = source, target, kind
        color = QColor(EDGE_COLORS.get(kind, UNKNOWN_EDGE_COLOR))
        pen = QPen(color)
        pen.setWidthF(1.4)
        self.setPen(pen)
        self.setBrush(QBrush(color))
        self.setZValue(-1)
        source.edges.append(self)
        target.edges.append(self)
        self.refresh()

    def refresh(self) -> None:
        a, b = self.source.pos(), self.target.pos()
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        path = QPainterPath()
        if length > 2 * RADIUS + 1:
            ux, uy = dx / length, dy / length
            start = QPointF(a.x() + ux * RADIUS, a.y() + uy * RADIUS)
            tip = QPointF(b.x() - ux * (RADIUS + 1), b.y() - uy * (RADIUS + 1))
            path.moveTo(start)
            path.lineTo(tip)
            size = 7.0
            left = QPointF(
                tip.x() - size * (ux * 0.9 - uy * 0.5), tip.y() - size * (uy * 0.9 + ux * 0.5)
            )
            right = QPointF(
                tip.x() - size * (ux * 0.9 + uy * 0.5), tip.y() - size * (uy * 0.9 - ux * 0.5)
            )
            path.addPolygon(QPolygonF([tip, left, right, tip]))
        self.setPath(path)


class StructureTab(QWidget):
    def __init__(self, data: Dataset, colors: dict[str, QColor]) -> None:
        super().__init__()
        self.data = data
        self.colors = colors
        self.music_font = _music_font()
        smufl = data.class_map().smufl
        self.class_codes = {cls: int(code, 16) for code, cls in smufl.items()}
        for cls, code in list(self.class_codes.items()):
            if cls.startswith("accidental"):
                self.class_codes.setdefault(cls.replace("accidental", "keyAccid"), code)
        self.score: str | None = None
        self.graph: nx.DiGraph = nx.DiGraph()
        self.components: list[set[str]] = []
        self.subgraph: nx.DiGraph = nx.DiGraph()
        self.items: dict[str, NodeItem] = {}
        self.edges: list[EdgeItem] = []
        self.positions: dict[str, QPointF] = {}
        self._layout_key: tuple | None = None
        self._crops: dict[str, Image.Image] = {}

        controls = QWidget()
        form = QFormLayout(controls)
        self.component = QComboBox()
        self.min_degree = QSpinBox()
        self.min_degree.setRange(0, 1000)
        self.max_nodes = QSpinBox()
        self.max_nodes.setRange(10, 20000)
        self.max_nodes.setValue(400)
        self.keep_isolated = QCheckBox("keep isolated nodes")
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(LAYOUTS)
        self.node_style = QComboBox()
        self.node_style.addItems(STYLES if self.music_font else STYLES[1:])
        self.labels = QComboBox()
        self.labels.addItems(LABELS)
        relayout = QPushButton("Re-run layout")
        relayout.clicked.connect(self.relayout)
        export = QPushButton("Export JSON…")
        export.clicked.connect(self.export)
        self.status = QLabel()
        self.status.setWordWrap(True)
        form.addRow("component", self.component)
        form.addRow("min degree", self.min_degree)
        form.addRow("max nodes", self.max_nodes)
        form.addRow(self.keep_isolated)
        form.addRow("layout", self.layout_choice)
        form.addRow("nodes", self.node_style)
        form.addRow("labels", self.labels)
        form.addRow(relayout)
        form.addRow(export)
        form.addRow(self.status)
        form.addRow(QLabel("hover: neighbours · drag: move a node"))
        for kind, name in EDGE_TYPES.items():
            form.addRow(QLabel(f"<span style='color:{EDGE_COLORS[kind]}'>━━ {kind} {name}</span>"))
        for widget in (self.component, self.layout_choice):
            widget.currentIndexChanged.connect(self.redraw)
        for spin in (self.min_degree, self.max_nodes):
            spin.valueChanged.connect(self.redraw)
        self.keep_isolated.toggled.connect(self.redraw)
        self.node_style.currentIndexChanged.connect(self.draw)
        self.labels.currentIndexChanged.connect(self.draw)

        self.scene = QGraphicsScene()
        self.view = ZoomView(self.scene)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.view)
        splitter.setSizes([280, 1200])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # -- data -----------------------------------------------------------------

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
            graph.nodes[node_id].update(
                cls=ann.get("class", "?"), page=ann.get("_page", 0), bbox=ann.get("bbox")
            )
        self.graph = graph

        def reading_order(component: set[str]) -> tuple[float, float]:
            # Components in page order, by their systems' position when they have any
            anchors = [n for n in component if nodes.get(n, {}).get("class") == "system"]
            boxes = [nodes[n] for n in (anchors or list(component)) if n in nodes]
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

    def _filtered(self) -> nx.DiGraph:
        index = self.component.currentIndex()
        keep = set(self.graph.nodes) if index <= 0 else self.components[index - 1]
        # Degrees are measured inside the shown component
        degree = dict(self.graph.subgraph(keep).degree())
        keep = {n for n in keep if degree[n] >= self.min_degree.value()}
        if not self.keep_isolated.isChecked():
            keep = {n for n in keep if degree[n] > 0}
        if len(keep) > self.max_nodes.value():
            keep = set(sorted(keep, key=lambda n: -degree[n])[: self.max_nodes.value()])
        return nx.DiGraph(self.graph.subgraph(keep))

    # -- layout ---------------------------------------------------------------

    def _layout(self, sub: nx.DiGraph) -> dict[str, QPointF]:
        nodes = list(sub.nodes)
        if not nodes:
            return {}
        kind = self.layout_choice.currentText()
        if kind == "page":
            # where the objects are on the page, pages stacked downwards
            heights = {}
            for n in nodes:
                b = sub.nodes[n]["bbox"] or [0, 0, 0, 0]
                heights[sub.nodes[n]["page"]] = max(heights.get(sub.nodes[n]["page"], 0), b[3])
            offset, top = {}, 0.0
            for page in sorted(heights):
                offset[page] = top
                top += heights[page] * 1.1
            pos = {}
            for n in nodes:
                b = sub.nodes[n]["bbox"] or [0, 0, 0, 0]
                pos[n] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 + offset[sub.nodes[n]["page"]])
            scale = 1.5
        else:
            undirected = sub.to_undirected()
            if kind == "ForceAtlas2":
                # Distributed attraction spreads the leaves of hubs (measures,
                # staves) and strong gravity keeps components together: on a
                # 160-node score graph the extent drops from 128 to 10 edge
                # lengths with fewer collisions than the defaults
                raw = nx.forceatlas2_layout(
                    undirected, max_iter=300, seed=0, distributed_action=True, strong_gravity=True
                )
            else:
                raw = nx.spring_layout(undirected, seed=0)
            pos = {n: (float(p[0]), float(p[1])) for n, p in raw.items()}
            scale = self._edge_scale(sub, pos)
            pos = self._declutter({n: (x * scale, y * scale) for n, (x, y) in pos.items()})
            return {n: QPointF(x, y) for n, (x, y) in pos.items()}
        return {n: QPointF(x * scale, y * scale) for n, (x, y) in pos.items()}

    @staticmethod
    def _edge_scale(sub: nx.DiGraph, pos: dict[str, tuple[float, float]]) -> float:
        """Scale making the median edge about four node radii long."""
        lengths = [math.dist(pos[u], pos[v]) for u, v in sub.edges if u in pos and v in pos]
        lengths = [x for x in lengths if x > 0]
        return 4 * RADIUS / float(np.median(lengths)) if lengths else 1.0

    @staticmethod
    def _declutter(
        pos: dict[str, tuple[float, float]], rounds: int = 40
    ) -> dict[str, tuple[float, float]]:
        """Push apart nodes whose discs overlap."""
        if len(pos) < 2:
            return pos
        from scipy.spatial import KDTree

        names = list(pos)
        pts = np.array([pos[n] for n in names], dtype=np.float64)
        rng = np.random.default_rng(0)
        gap = 2.4 * RADIUS
        for _ in range(rounds):
            pairs = KDTree(pts).query_pairs(gap, output_type="ndarray")
            if not len(pairs):
                break
            d = pts[pairs[:, 1]] - pts[pairs[:, 0]]
            dist = np.linalg.norm(d, axis=1)
            same = dist < 1e-6
            d[same] = rng.normal(size=(int(same.sum()), 2))
            dist[same] = np.linalg.norm(d[same], axis=1)
            push = (d / dist[:, None]) * ((gap - dist) / 2)[:, None]
            np.add.at(pts, pairs[:, 0], -push)
            np.add.at(pts, pairs[:, 1], push)
        return {n: (x, y) for n, (x, y) in zip(names, pts.tolist(), strict=True)}

    def relayout(self) -> None:
        self._layout_key = None
        self.redraw()

    def redraw(self) -> None:
        """Filter, lay out (unless only the drawing changed) and draw."""
        if self.score is None:
            return
        self.subgraph = self._filtered()
        key = (self.score, frozenset(self.subgraph.nodes), self.layout_choice.currentText())
        if key != self._layout_key:
            self.positions = self._layout(self.subgraph)
            self._layout_key = key
        self.draw()

    # -- drawing --------------------------------------------------------------

    def _crop(self, node: dict) -> QPixmap | None:
        score = self.score
        if score is None or node.get("bbox") is None:
            return None
        stems = self.data.by_score.get(score, [])
        if node["page"] >= len(stems):
            return None
        stem = stems[node["page"]]
        if stem not in self._crops:
            with Image.open(self.data.image(stem)) as im:
                self._crops[stem] = im.convert("RGB")
        x1, y1, x2, y2 = node["bbox"]
        pad = 3
        crop = self._crops[stem].crop((int(x1) - pad, int(y1) - pad, int(x2) + pad, int(y2) + pad))
        crop.thumbnail((int(2 * RADIUS), int(2 * RADIUS)))
        return QPixmap.fromImage(_qimage(np.asarray(crop)))

    def _visual(self, item: NodeItem, node: dict, style: str) -> None:
        cls = node["cls"]
        color = self.colors.get(cls, QColor("#ffffff"))
        if style == "dot":
            disc = QGraphicsEllipseItem(-RADIUS / 2, -RADIUS / 2, RADIUS, RADIUS)
            disc.setBrush(QBrush(color))
            disc.setPen(QPen(color.darker()))
            item.addToGroup(disc)
            return
        disc = QGraphicsEllipseItem(-RADIUS, -RADIUS, 2 * RADIUS, 2 * RADIUS)
        disc.setBrush(QBrush(QColor("#fbfaf6")))
        ring = QPen(color)
        ring.setWidthF(2.5)
        disc.setPen(ring)
        item.addToGroup(disc)
        if style == "crop":
            pixmap = self._crop(node)
            if pixmap is not None:
                pic = QGraphicsPixmapItem(pixmap)
                pic.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
                item.addToGroup(pic)
            return
        code = self.class_codes.get(cls, CLASS_GLYPHS.get(cls))
        if code is not None and self.music_font:
            shape = QGraphicsPathItem(_glyph_path(code, self.music_font, 1.4 * RADIUS))
        elif cls in DRAWN:
            shape = QGraphicsPathItem(_drawn_path(cls))
        else:
            badge = QGraphicsSimpleTextItem(cls[:4])
            font = QFont("sans-serif")
            font.setPixelSize(9)
            badge.setFont(font)
            rect = badge.boundingRect()
            badge.setPos(-rect.width() / 2, -rect.height() / 2)
            item.addToGroup(badge)
            return
        shape.setBrush(QBrush(QColor("#111111")))
        shape.setPen(QPen(Qt.PenStyle.NoPen))
        item.addToGroup(shape)

    def _label(self, item: NodeItem, node_id: str, cls: str) -> None:
        mode = self.labels.currentText()
        if mode == "none":
            return
        text = {"class": cls, "id": node_id, "class + id": f"{cls}\n{node_id}"}[mode]
        label = QGraphicsSimpleTextItem(text)
        font = QFont("sans-serif")
        font.setPixelSize(8)
        label.setFont(font)
        label.setBrush(QBrush(QColor("#e8e8e8")))
        rect = label.boundingRect()
        label.setPos(-rect.width() / 2, RADIUS + 1)
        item.addToGroup(label)

    def draw(self) -> None:
        # keep where the user dragged nodes
        for node_id, item in self.items.items():
            self.positions[node_id] = item.pos()
        self.scene.clear()
        self.items, self.edges = {}, []
        sub = self.subgraph
        if self.score is None or sub.number_of_nodes() == 0:
            self.status.setText("no graph for this score")
            return
        style = self.node_style.currentText()
        for node_id in sub.nodes:
            node = sub.nodes[node_id]
            item = NodeItem(node_id, node["cls"], node["page"], self)
            self._visual(item, node, style)
            self._label(item, node_id, node["cls"])
            item.setPos(self.positions.get(node_id, QPointF(0, 0)))
            self.scene.addItem(item)
            self.items[node_id] = item
        for u, v, attrs in sub.edges(data=True):
            edge = EdgeItem(self.items[u], self.items[v], attrs["kind"])
            self.scene.addItem(edge)
            self.edges.append(edge)
        self.status.setText(
            f"{sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges shown; "
            f"{len(self.components)} components in the score"
            + ("" if self.music_font else "; no SMuFL font installed (scripts/install-fonts.sh)")
        )
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40))
        if not self.view.zoomed:
            self.view.fit()

    def highlight(self, focus: NodeItem | None) -> None:
        """Dim everything but a node, its neighbours and their edges."""
        if focus is None:
            for item in self.items.values():
                item.setOpacity(1.0)
            for edge in self.edges:
                edge.setOpacity(1.0)
            return
        near = {focus.node_id, *self.subgraph.successors(focus.node_id)}
        near |= set(self.subgraph.predecessors(focus.node_id))
        for node_id, item in self.items.items():
            item.setOpacity(1.0 if node_id in near else 0.12)
        for edge in self.edges:
            touches = focus in (edge.source, edge.target)
            edge.setOpacity(1.0 if touches else 0.05)

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export subgraph", f"{self.score}.json")
        if not path:
            return
        payload = {
            "nodes": [
                {"id": n, "class": a["cls"], "page": a["page"], "bbox": a["bbox"]}
                for n, a in self.subgraph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "kind": a["kind"]}
                for u, v, a in self.subgraph.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["CLASS_GLYPHS", "EdgeItem", "NodeItem", "StructureTab"]
