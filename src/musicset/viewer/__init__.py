"""Dataset viewer (``musicset view``), from the kakigori visualizers.

* **Page**: the page image with its detection boxes (classes grouped by the
  label set, every class its own colour), the graph edges between them
  (toggle per edge type, arrows point from parent to child) and the page's
  text target (ABC, else per-page MusicXML).
* **Structure**: the score's graph as a network: ForceAtlas2, spring or
  page-position layout; nodes drawn as their SMuFL glyph (or page crop, or a
  dot) with class/id labels; hover highlights neighbours, drag moves nodes;
  filters by component, degree and size; exportable as JSON.
* **Augment**: the page clean, augmented (boxes moved with the ink) and next
  to a real scan scaled to the same staff size; preset, strength and seed
  controls, optional re-engraving with a sampled render profile, and the
  staff line / staff space ratio of all three.

Keys: ←/→ or A/D change page, Ctrl+F jumps to a page by name.
Needs the ``viz`` extra (PySide6, networkx).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from musicset.labels import LabelSet
from musicset.viewer.augment import AugmentTab
from musicset.viewer.data import Dataset, class_colors, score_of
from musicset.viewer.page import PageTab
from musicset.viewer.structure import StructureTab

__all__ = ["Dataset", "Viewer", "run"]


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
        self.augment = AugmentTab(data, colors, self.pages.class_filter.visible)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.pages, "Page")
        self.tabs.addTab(self.structure, "Structure")
        self.tabs.addTab(self.augment, "Augment")
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
        current = self.tabs.currentWidget()
        if current is self.pages:
            self.pages.load(stem)
        elif current is self.augment:
            self.augment.load(stem)
        else:
            self.structure.load(score_of(stem))


def run(root: Path, labels: LabelSet | None, scans: list[Path] | None = None) -> None:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Viewer(Dataset(root, labels, scans))
    window.show()
    sys.exit(app.exec())
