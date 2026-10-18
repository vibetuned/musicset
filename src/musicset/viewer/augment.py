"""Augment tab: clean, augmented and real-scan pages side by side."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from musicset.annotate.extract import annotate_svg
from musicset.augment import PRESETS, Augmenter
from musicset.render.engrave import Engraving, rasterize
from musicset.render.profiles import has_encoded_breaks, sample_render_options
from musicset.scans import load_scan, pdf_page_count, staff_profile
from musicset.score import read_musicxml
from musicset.viewer.data import Dataset, score_of
from musicset.viewer.widgets import ZoomView, _qimage
from musicset.views.detection import detection_annotations


def _ratio(image) -> str:
    profile = staff_profile(image)
    return f"{profile.ratio:.3f}" if profile else "n/a"


class AugmentTab(QWidget):
    def __init__(self, data: Dataset, colors: dict[str, QColor], visible_classes) -> None:
        super().__init__()
        self.data = data
        self.colors = colors
        self.visible_classes = visible_classes
        self.stem: str | None = None
        self._scan_cache: dict[tuple[str, int], Image.Image] = {}
        self._redraw = QTimer(self, singleShot=True, interval=250)
        self._redraw.timeout.connect(self.refresh)

        controls = QWidget()
        form = QFormLayout(controls)
        self.preset = QComboBox()
        self.preset.addItems(["mixed", *[p for p in PRESETS if p != "clean"], "clean"])
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(0, 200)
        self.strength.setValue(100)
        self.strength_label = QLabel("1.00")
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        new_seed = QPushButton("New seed")
        new_seed.clicked.connect(lambda: self.seed.setValue(random.randint(0, 1_000_000)))
        self.vary = QCheckBox("vary engraving (sampled render profile)")
        self.show_boxes = QCheckBox("boxes (classes shown in the Page tab)")
        self.show_boxes.setChecked(True)
        self.scan = QComboBox()
        self.scan.addItem("no reference scan", None)
        for path in data.scans:
            pages = pdf_page_count(path) if path.suffix.lower() == ".pdf" else 1
            self.scan.addItem(path.name[:40], (path, pages))
        self.scan_page = QSpinBox()
        self.scan_page.setRange(1, 1)
        open_scan = QPushButton("Open scan…")
        open_scan.clicked.connect(self.open_scan)
        self.steps = QLabel()
        self.steps.setWordWrap(True)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        form.addRow("preset", self.preset)
        form.addRow("strength", self.strength)
        form.addRow("", self.strength_label)
        form.addRow("seed", self.seed)
        form.addRow(new_seed)
        form.addRow(self.vary)
        form.addRow(self.show_boxes)
        form.addRow("reference", self.scan)
        form.addRow("scan page", self.scan_page)
        form.addRow(open_scan)
        form.addRow(QLabel("<b>applied</b>"))
        form.addRow(self.steps)
        form.addRow(QLabel("<b>staff line / staff space</b>"))
        form.addRow(self.stats)

        for signal in (
            self.preset.currentIndexChanged,
            self.strength.valueChanged,
            self.seed.valueChanged,
            self.vary.toggled,
            self.show_boxes.toggled,
            self.scan_page.valueChanged,
        ):
            signal.connect(self._schedule)
        self.scan.currentIndexChanged.connect(self._scan_changed)

        self.scenes = [QGraphicsScene() for _ in range(3)]
        self.views = [ZoomView(scene) for scene in self.scenes]
        self.titles = [QLabel(t) for t in ("clean", "augmented", "reference scan")]
        panels = QSplitter(Qt.Orientation.Horizontal)
        for title, view in zip(self.titles, self.views, strict=True):
            box = QWidget()
            lay = QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(title)
            lay.addWidget(view)
            panels.addWidget(box)
        # pan and zoom together
        for view in self.views:
            for bar in (view.horizontalScrollBar(), view.verticalScrollBar()):
                bar.valueChanged.connect(lambda _=0, v=view: self._sync(v))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(panels)
        splitter.setSizes([300, 1300])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._syncing = False

    # -- events ---------------------------------------------------------------

    def _schedule(self) -> None:
        self.strength_label.setText(f"{self.strength.value() / 100:.2f}")
        self._redraw.start()

    def _scan_changed(self) -> None:
        choice = self.scan.currentData()
        self.scan_page.setRange(1, choice[1] if choice else 1)
        self.scan_page.setValue(min(3, choice[1]) if choice else 1)
        self._schedule()

    def open_scan(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference scan", "", "Scans (*.pdf *.png *.jpg)"
        )
        if path:
            p = Path(path)
            pages = pdf_page_count(p) if p.suffix.lower() == ".pdf" else 1
            self.scan.addItem(p.name[:40], (p, pages))
            self.scan.setCurrentIndex(self.scan.count() - 1)

    def _sync(self, source: ZoomView) -> None:
        if self._syncing:
            return
        self._syncing = True
        for view in self.views:
            if view is not source:
                view.setTransform(source.transform())
                view.zoomed = source.zoomed
                view.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
                view.verticalScrollBar().setValue(source.verticalScrollBar().value())
        self._syncing = False

    def load(self, stem: str) -> None:
        self.stem = stem
        self.vary.setEnabled(self.data.source(score_of(stem)) is not None)
        self.refresh()

    # -- rendering ------------------------------------------------------------

    def _clean_page(self, stem: str) -> tuple[np.ndarray, list[dict]]:
        """The page as built, or re-engraved with a sampled render profile."""
        source = self.data.source(score_of(stem))
        if not (self.vary.isChecked() and source is not None):
            with Image.open(self.data.image(stem)) as im:
                return np.asarray(im.convert("RGB")), self.data.annotations(stem)
        xml = read_musicxml(source).text
        score = score_of(stem)
        options = sample_render_options(
            random.Random(f"{self.seed.value()}:{score}"), encoded_breaks=has_encoded_breaks(xml)
        )
        engraving = Engraving.load(xml, options)
        number = min(int(stem.rsplit("_page", 1)[1]), engraving.page_count)
        svg = engraving.svg(number)
        raster = rasterize(svg)
        page = annotate_svg(svg, text_ink=raster.text_ink)
        with Image.open(io.BytesIO(raster.png)) as im:
            image = np.asarray(im.convert("RGB"))
        self.titles[0].setText(
            f"clean: {options.font}, scale {options.scale}, {options.breaks} breaks"
        )
        return image, detection_annotations(page, self.data.class_map(), raster.size)

    def _reference(self, clean: np.ndarray) -> np.ndarray | None:
        choice = self.scan.currentData()
        if not choice:
            return None
        path, _ = choice
        key = (str(path), self.scan_page.value() - 1)
        if key not in self._scan_cache:
            self._scan_cache[key] = load_scan(path, key[1])
        scan = self._scan_cache[key]
        ours, theirs = staff_profile(clean), staff_profile(scan)
        if ours and theirs:
            # same staff space as the render: engraving weights compare directly
            f = ours.space / theirs.space
            scan = scan.resize((max(1, round(scan.width * f)), max(1, round(scan.height * f))))
        return np.asarray(scan)

    def refresh(self) -> None:
        if self.stem is None:
            return
        self.titles[0].setText("clean")
        clean, anns = self._clean_page(self.stem)
        boxes = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
        out = Augmenter(self.preset.currentText(), self.strength.value() / 100)(
            clean, boxes, rng=self.seed.value()
        )
        reference = self._reference(clean)
        classes = [a["class"] for a in anns]
        visible = self.visible_classes() if self.show_boxes.isChecked() else set()
        self._show(0, clean, boxes, classes, None, visible)
        self._show(1, out.image, out.boxes, classes, out.keep, visible)
        self._show(2, reference, None, [], None, set())
        self.titles[1].setText(f"augmented: {out.preset}")
        self.steps.setText(", ".join(out.log) or "nothing (clean)")
        self.stats.setText(
            f"clean {_ratio(clean)} · augmented {_ratio(out.image)}"
            + (f" · scan {_ratio(reference)}" if reference is not None else "")
            + "\n(IMSLP plate engravings: 0.14-0.33, median 0.25)"
        )

    def _show(self, i, image, boxes, classes, keep, visible) -> None:
        scene, view = self.scenes[i], self.views[i]
        transform, zoomed = view.transform(), view.zoomed
        scene.clear()
        if image is None:
            scene.setSceneRect(0, 0, 1, 1)
            return
        h, w = image.shape[:2]
        scene.addPixmap(QPixmap.fromImage(_qimage(image)))
        scene.setSceneRect(0, 0, w, h)
        width = max(1.0, w * 0.0012)
        if boxes is not None:
            for j, (cls, box) in enumerate(zip(classes, boxes, strict=True)):
                if cls not in visible or (keep is not None and not keep[j]):
                    continue
                pen = QPen(self.colors.get(cls, QColor("#ffffff")))
                pen.setWidthF(width)
                x1, y1, x2, y2 = (float(v) for v in box)
                scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
        if zoomed:
            view.setTransform(transform)
        else:
            view.fit()
