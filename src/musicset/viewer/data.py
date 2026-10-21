"""Dataset access for the viewer: pages, boxes, graphs, targets."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from PySide6.QtGui import (
    QColor,
)

from musicset.labels import LabelSet
from musicset.score import read_musicxml
from musicset.views.detection import ClassMap
from musicset.views.graph import graph_nodes, page_sort_key

EDGE_COLORS = {1: "#e6194b", 2: "#3cb44b", 3: "#4363d8", 4: "#f58231", 5: "#42d4f4"}
UNKNOWN_EDGE_COLOR = "#9aa0a6"


# -- data -------------------------------------------------------------------


class Dataset:
    """A dataset directory as built by ``musicset build``."""

    def __init__(
        self,
        root: Path,
        labels: LabelSet | None,
        scans: list[Path] | None = None,
        pred_kern: Path | None = None,
    ) -> None:
        self.root = root
        self.labels = labels
        self.scans = scans or []
        self.pred_kern = pred_kern
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

    def class_map(self) -> ClassMap:
        """Label set for re-engraved pages: the dataset's, else its classes with
        the full set's glyph map."""
        if self.labels:
            return self.labels.classes
        return ClassMap(tuple(self.classes()), LabelSet.load("full").classes.smufl)

    def kern(self, score: str) -> Path | None:
        """The score's ground-truth **kern (``build --kern``)."""
        path = self.root / "kern" / f"{score}.krn"
        return path if path.exists() else None

    def pred(self, score: str) -> Path | None:
        """The score's predicted **kern from ``--pred-kern``."""
        if self.pred_kern is None:
            return None
        path = self.pred_kern / f"{score}.krn"
        return path if path.exists() else None

    @cache  # noqa: B019
    def kern_scores(self, score: str) -> dict[str, dict[str, Any] | None]:
        """``musicset.evaluate.score_kern`` of the ground-truth and the
        predicted kern against the score's MEI (None where one is missing)."""
        from musicset.evaluate import score_kern

        mei = self.root / "mei" / f"{score}.mei"
        out: dict[str, dict[str, Any] | None] = {}
        for which, path in (("gt", self.kern(score)), ("pred", self.pred(score))):
            out[which] = score_kern(mei, path) if mei.exists() and path is not None else None
        return out

    def source(self, score: str) -> Path | None:
        path = self.root / "injected" / f"{score}.mxl"
        return path if path.exists() else None

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
