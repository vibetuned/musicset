"""Widgets shared by the viewer's tabs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsScene,
    QGraphicsView,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

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


def _qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr[..., :3] if arr.ndim == 3 else np.stack([arr] * 3, axis=-1))
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
