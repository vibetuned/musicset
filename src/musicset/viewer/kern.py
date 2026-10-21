"""Kern tab: a score's source pages, its ground-truth **kern and a predicted
**kern side by side — the two kerns engraved by Verovio at the same width —
with their scores against the MEI (``musicset.evaluate``) and both kern texts.

Predictions come from ``musicset view --pred-kern DIR`` (``<score>.krn``, the
layout of ``kern/``), e.g. a recognition pipeline's output on the dataset.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from musicset.viewer.data import Dataset
from musicset.viewer.widgets import ZoomView

PAGE_GAP = 24.0

# Verovio's C++ core aborts the process on some inconsistent **kern (a
# predicted one, typically): engrave in a subprocess, never in the viewer
ENGRAVE = r"""
import json, sys, verovio
from pathlib import Path
from musicset.render.engrave import rasterize
krn, prefix = Path(sys.argv[1]), sys.argv[2]
tk = verovio.toolkit()
tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 35, "adjustPageHeight": True,
               "header": "none", "footer": "none"})
if not tk.loadData(krn.read_text(encoding="utf-8")):
    sys.exit(1)
files = []
for p in range(1, tk.getPageCount() + 1):
    path = Path(f"{prefix}_{p}.png")
    path.write_bytes(rasterize(tk.renderToSVG(p)).png)
    files.append(str(path))
print(json.dumps(files))
"""


class Engraver(QObject):
    """Engraves **kern files to PNG pages in the background, cached per file
    version; ``done(key, which, pngs)`` arrives on the GUI thread (no pages:
    Verovio could not engrave it)."""

    done = Signal(str, str, list)

    def __init__(self) -> None:
        super().__init__()
        self._dir = tempfile.TemporaryDirectory(prefix="musicset-kern-")
        self._cache: dict[tuple[str, int], list[str]] = {}
        self._lock = threading.Lock()

    def request(self, key: str, which: str, krn: Path) -> None:
        stamp = (str(krn), krn.stat().st_mtime_ns)
        with self._lock:
            cached = self._cache.get(stamp)
        if cached is not None:
            self.done.emit(key, which, cached)
            return
        prefix = Path(self._dir.name) / f"{len(self._cache)}_{which}"

        def work() -> None:
            files: list[str] = []
            try:
                run = subprocess.run(
                    [sys.executable, "-c", ENGRAVE, str(krn), str(prefix)],
                    capture_output=True, text=True, timeout=300,
                )
                if run.returncode == 0:
                    files = json.loads(run.stdout.strip().splitlines()[-1])
            except (subprocess.TimeoutExpired, ValueError, IndexError):
                files = []
            with self._lock:
                self._cache[stamp] = files
            self.done.emit(key, which, files)

        threading.Thread(target=work, daemon=True).start()


class WidthView(ZoomView):
    """A column of pages fitted to the view's width, scrolled from the top."""

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

    def fit(self) -> None:
        self.zoomed = False
        rect = self.scene().sceneRect()
        if rect.width() <= 0:
            return
        self.resetTransform()
        factor = max(0.01, (self.viewport().width() - 4) / rect.width())
        self.scale(factor, factor)
        self.verticalScrollBar().setValue(self.verticalScrollBar().minimum())


class Column(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = QLabel(f"<b>{title}</b>")
        self.scene = QGraphicsScene()
        self.view = WidthView(self.scene)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.view, 1)

    def show_pages(self, pngs: list[Any]) -> None:
        self.scene.clear()
        y, width = 0.0, 1.0
        for png in pngs:
            pixmap = QPixmap(str(png))
            item = self.scene.addPixmap(pixmap)
            item.setPos(0, y)
            y += pixmap.height() + PAGE_GAP
            width = max(width, pixmap.width())
        self.scene.setSceneRect(QRectF(0, 0, width, max(1.0, y - PAGE_GAP)))
        self.view.fit()

    def show_message(self, text: str) -> None:
        self.scene.clear()
        label = self.scene.addSimpleText(text, QFont("sans-serif", 14))
        label.setBrush(Qt.GlobalColor.white)
        self.scene.setSceneRect(QRectF(0, 0, 900, 200))
        self.view.fit()


class KernTab(QWidget):
    def __init__(self, data: Dataset) -> None:
        super().__init__()
        self.data = data
        self.score: str | None = None
        self.engraver = Engraver()
        self.engraver.done.connect(self._engraved)

        self.columns = {
            "source": Column("Source pages (the engraved dataset pages)"),
            "gt": Column("Ground truth (kern/)"),
            "pred": Column(
                f"Predicted ({data.pred_kern})" if data.pred_kern else "Predicted (--pred-kern DIR)"
            ),
        }
        self.toggles: dict[str, QCheckBox] = {}
        bar = QHBoxLayout()
        names = (("source", "source pages"), ("gt", "ground truth"), ("pred", "predicted"))
        for key, name in names:
            box = QCheckBox(name)
            box.setChecked(key != "pred" or data.pred_kern is not None)
            box.toggled.connect(lambda on, k=key: self._toggle(k, on))
            bar.addWidget(box)
            self.toggles[key] = box
        self.show_text = QCheckBox("kern text")
        bar.addWidget(self.show_text)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.RichText)
        bar.addWidget(self.status, 1)

        self.pages = pages = QSplitter(Qt.Orientation.Horizontal)
        for key, column in self.columns.items():
            pages.addWidget(column)
            pages.setStretchFactor(pages.indexOf(column), 1)
            column.setVisible(self.toggles[key].isChecked())
        self.texts = {key: QPlainTextEdit() for key in ("gt", "pred")}
        texts = QSplitter(Qt.Orientation.Horizontal)
        for text in self.texts.values():
            text.setReadOnly(True)
            text.setFont(QFont("monospace"))
            texts.addWidget(text)
        texts.setVisible(False)
        self.show_text.toggled.connect(texts.setVisible)
        body = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(pages)
        body.addWidget(texts)
        body.setSizes([700, 250])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(body, 1)

    def _toggle(self, key: str, on: bool) -> None:
        self.columns[key].setVisible(on)
        self._equalize()

    def _equalize(self) -> None:
        """Shown columns share the width equally (else the splitter sizes
        them by content and an engraving can end up a sliver)."""
        self.pages.setSizes([1000 if c.isVisible() else 0 for c in self.columns.values()])

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._equalize()

    def load(self, score: str) -> None:
        if score == self.score:
            return
        self.score = score
        self.columns["source"].show_pages([self.data.image(s) for s in self.data.by_score[score]])
        for which, path in (("gt", self.data.kern(score)), ("pred", self.data.pred(score))):
            text = self.texts[which]
            if path is None:
                missing = (
                    "no ground-truth kern (build with --kern)"
                    if which == "gt"
                    else "no predicted kern for this score"
                    if self.data.pred_kern
                    else "no predictions: musicset view --pred-kern DIR"
                )
                self.columns[which].show_message(missing)
                text.setPlainText("")
                continue
            self.columns[which].show_message("engraving…")
            text.setPlainText(path.read_text(encoding="utf-8"))
            self.engraver.request(score, which, path)
        self.status.setText(self._scores_text(score))

    def _engraved(self, key: str, which: str, pngs: list) -> None:
        if key != self.score:
            return
        if pngs:
            self.columns[which].show_pages(pngs)
        else:
            self.columns[which].show_message("Verovio could not engrave this kern (see its text)")

    def _scores_text(self, score: str) -> str:
        scores = self.data.kern_scores(score)
        parts = []
        gt, pred = scores.get("gt"), scores.get("pred")
        if gt is not None:
            parts.append(f"ground truth vs MEI {_pct(gt['pitch'], gt['pitch'] < 95)}")
        if pred is not None:
            parts.append(
                f"predicted: pitch {_pct(pred['pitch'], pred['pitch'] < 80)}, "
                f"rhythm {_pct(pred['rhythm'], pred['rhythm'] < 80)}, "
                f"precision {_pct(pred['precision'], pred['precision'] < 70)}, "
                f"notes out {pred['ratio']:.2f}x of {pred['notes']}"
            )
        return " &nbsp;·&nbsp; ".join(parts) if parts else "no MEI to score against"


def _pct(value: float, bad: bool) -> str:
    text = f"{value:.1f}%"
    return f'<span style="color:#d32f2f;font-weight:600">{text}</span>' if bad else text
