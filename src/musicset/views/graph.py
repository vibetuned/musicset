"""Graph view: relations between a score's annotated objects, from MEI.

Nodes are the detection-view annotations of every page (identified by their
id); edges come from the MEI encoding plus page geometry, typed:

    1 structural    measure -> staff -> layer -> note, notehead -> stem ...
    2 modifier      note -> accidental / dot / articulation / slur / tie ...
    3 temporal      consecutive events within a layer
    4 text sync     note -> syllable
    5 simultaneity  events sounding together across staves / layers

Ported from kakigori's ``GroundTruthGraphBuilder`` with the same output, so
graphs built here match kakigori's. Annotations that share an id (glyphs of one
object) get a positional pseudo-id ``{id}_{class}_{n}``, so the annotation
order of every page, and the page order, are part of the contract.
"""

from __future__ import annotations

import copy
import itertools
import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

MEI_NS = {"mei": "http://www.music-encoding.org/ns/mei"}
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

EDGE_TYPES = {
    1: "structural",
    2: "modifier",
    3: "temporal",
    4: "text sync",
    5: "simultaneity",
}

type Edge = tuple[str, str, int]


def page_sort_key(name: str | Path) -> tuple[str, int]:
    """Natural order of per-page files: ``x_page10`` after ``x_page2``."""
    stem = Path(name).name.removesuffix(".gz").rsplit(".", 1)[0]
    m = re.search(r"_page(\d+)$", stem)
    return (stem[: m.start()], int(m.group(1))) if m else (stem, 0)


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def graph_nodes(
    pages: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Edge]]:
    """Graph nodes from per-page annotations: (all annotations, id -> node, glyph links).

    Annotations come back copied, tagged with ``_page``. When several share an
    id (glyphs of one object), each later one gets the pseudo-id
    ``{id}_{class}_{n}`` and a structural edge from the first.
    """
    annotations: list[dict[str, Any]] = []
    for page_idx, page_annotations in enumerate(pages):
        for ann in copy.deepcopy(page_annotations):
            ann["_page"] = page_idx
            annotations.append(ann)

    nodes: dict[str, dict[str, Any]] = {}
    links: list[Edge] = []
    for ann in annotations:
        if "id" not in ann:
            continue
        base_id = ann["id"]
        if base_id in nodes:
            pseudo_id = f"{base_id}_{ann['class']}_{len(nodes)}"
            ann["id"] = pseudo_id
            nodes[pseudo_id] = ann
            links.append((base_id, pseudo_id, 1))
        else:
            nodes[base_id] = ann
    return annotations, nodes, links


class _GraphBuilder:
    def __init__(
        self,
        mei_root: ET.Element,
        pages: list[list[dict[str, Any]]],
        node_roles: dict[str, list[str]],
    ) -> None:
        self.root = mei_root
        self.roles = node_roles
        self.spatial_nodes, self.node_map, self.edges = graph_nodes(pages)

    def _role(self, name: str) -> set[str]:
        return set(self.roles.get(name, []))

    def _class_of(self, node_id: str | None) -> str | None:
        node = self.node_map.get(node_id) if node_id else None
        return node["class"] if node else None

    @staticmethod
    def _event_duration(element: ET.Element, tuplet_factor: Fraction) -> Fraction | None:
        """Duration in whole notes, or None when not numeric (breve...)."""
        dur = element.get("dur")
        if dur is None:
            return None
        try:
            base = Fraction(1, int(dur))
        except (ValueError, ZeroDivisionError):
            return None
        dots = int(element.get("dots", 0))
        return base * (2 - Fraction(1, 2**dots)) * tuplet_factor

    def _collect_onsets(
        self,
        parent: ET.Element,
        tuplet_factor: Fraction,
        onset: Fraction,
        events: list[tuple[Fraction, str | None]],
    ) -> Fraction:
        for child in parent:
            tag = child.tag.split("}")[-1]
            if tag in ("beam", "graceGrp", "bTrem", "fTrem"):
                onset = self._collect_onsets(child, tuplet_factor, onset, events)
            elif tag == "tuplet":
                num = int(child.get("num", 3))
                numbase = int(child.get("numbase", 2))
                onset = self._collect_onsets(
                    child, tuplet_factor * Fraction(numbase, num), onset, events
                )
            elif tag in ("note", "chord", "rest", "mRest", "space"):
                if child.get("grace"):
                    continue  # grace notes take no time
                events.append((onset, child.get(_XML_ID)))
                dur = self._event_duration(child, tuplet_factor)
                if dur is not None:
                    onset += dur
        return onset

    def build(self) -> list[Edge]:
        self._temporal_edges()
        self._simultaneity_edges()
        self._hierarchy_edges()
        self._spatial_fallbacks()
        seen: set[tuple[str, str]] = set()
        deduped = []
        for u, v, kind in self.edges:
            if (u, v) not in seen:
                seen.add((u, v))
                deduped.append((u, v, kind))
        return deduped

    def _temporal_edges(self) -> None:
        """Type 3: left-to-right event sequence within each layer."""
        keep = self._role("temporal_anchors") | self._role("context_globals")
        for layer in self.root.findall(".//mei:layer", MEI_NS):
            sequence: list[str] = []
            self._layer_events(layer, keep, sequence)
            self.edges.extend((a, b, 3) for a, b in itertools.pairwise(sequence))

    def _layer_events(self, el: ET.Element, keep: set[str], out: list[str]) -> None:
        """Event ids of a layer in order, looking inside beams and tuplets."""
        for child in el:
            if child.tag.split("}")[-1] in ("beam", "tuplet"):
                self._layer_events(child, keep, out)
                continue
            ev_id = child.get(_XML_ID)
            if ev_id and self._class_of(ev_id) in keep:
                out.append(ev_id)

    def _simultaneity_edges(self) -> None:
        """Type 5: events sharing an onset across staves/layers, chained top to bottom."""
        temporal = self._role("temporal_anchors")
        for measure in self.root.findall(".//mei:measure", MEI_NS):
            groups: dict[Fraction, list[tuple[int, int, str]]] = {}
            for staff in measure.findall("mei:staff", MEI_NS):
                staff_n = int(staff.get("n", 0))
                for layer_idx, layer in enumerate(staff.findall("mei:layer", MEI_NS)):
                    layer_n = int(layer.get("n", layer_idx + 1))
                    events: list[tuple[Fraction, str | None]] = []
                    self._collect_onsets(layer, Fraction(1), Fraction(0), events)
                    for onset, ev_id in events:
                        cls = self._class_of(ev_id)
                        # A measure rest is centred, never on the shared beat line
                        if ev_id and cls in temporal and cls != "mRest":
                            groups.setdefault(onset, []).append((staff_n, layer_n, ev_id))
            for group in groups.values():
                group.sort()
                for (s1, l1, id1), (s2, l2, id2) in itertools.pairwise(group):
                    if (s1, l1) != (s2, l2):
                        self.edges.append((id1, id2, 5))

    def _hierarchy_edges(self) -> None:
        """Types 1/2/4 from MEI nesting; control events link to their anchors."""
        modifier = self._role("modifiers")
        sync = self._role("synchronization_text")
        parent_map = {c: p for p in self.root.iter() for c in p}

        for el in self.root.iter():
            child_id = el.get(_XML_ID)
            child_class = self._class_of(child_id)
            if child_id is None or child_class is None:
                continue

            if child_class in modifier:
                # Control events (slur, trill, fermata...) anchor via @startid/@endid
                # (arpeggios via @plist) rather than through nesting
                start = el.get("startid", "").lstrip("#")
                if not start:
                    plist = el.get("plist", "").split()
                    start = plist[0].lstrip("#") if plist else ""
                if start and start in self.node_map:
                    self.edges.append((start, child_id, 2))
                    end = el.get("endid", "").lstrip("#")
                    if end and end != start and end in self.node_map:
                        self.edges.append((end, child_id, 2))
                    continue

            parent_id = None
            node = el
            while node in parent_map:
                node = parent_map[node]
                pid = node.get(_XML_ID)
                if pid and pid in self.node_map:
                    parent_id = pid
                    break
            if parent_id:
                kind = 2 if child_class in modifier else 4 if child_class in sync else 1
                self.edges.append((parent_id, child_id, kind))

    def _spatial_fallbacks(self) -> None:
        """Relations MEI does not encode, recovered from page geometry."""
        temporal = self._role("temporal_anchors")
        modifier = self._role("modifiers")
        context = self._role("context_globals")
        nodes = self.spatial_nodes

        # System -> measure
        systems = [n for n in nodes if n.get("class") == "system"]
        for measure in (n for n in nodes if n.get("class") == "measure"):
            m_cy = (measure["bbox"][1] + measure["bbox"][3]) / 2
            for system in systems:
                same_page = system.get("_page") == measure.get("_page")
                if same_page and system["bbox"][1] <= m_cy <= system["bbox"][3]:
                    self.edges.append((system["id"], measure["id"], 1))
                    break

        # Staff -> clefs / key and time signatures not reached through MEI
        staves = [n for n in nodes if n.get("class") == "staff"]
        children = {child for _, child, _ in self.edges}
        for node in nodes:
            if node["class"] in context and node.get("id") and node["id"] not in children:
                _, n_cy = _center(node["bbox"])
                for staff in staves:
                    same_page = staff.get("_page") == node.get("_page")
                    if same_page and staff["bbox"][1] <= n_cy <= staff["bbox"][3]:
                        self.edges.append((staff["id"], node["id"], 1))
                        break

        # Note -> dots and other attribute-encoded modifiers: nearest event
        children = {child for _, child, _ in self.edges}
        events = [n for n in nodes if n.get("class") in temporal and "id" in n]
        for node in nodes:
            if not (node["class"] in modifier and node.get("id") and node["id"] not in children):
                continue
            n_cx, n_cy = _center(node["bbox"])
            best, best_dist = None, float("inf")
            for note in events:
                if note.get("_page") != node.get("_page"):
                    continue
                if not note["bbox"][1] - 30 <= n_cy <= note["bbox"][3] + 30:
                    continue
                dist = abs(n_cx - (note["bbox"][0] + note["bbox"][2]) / 2.0)
                if dist < best_dist:
                    best, best_dist = note, dist
            if best is not None:
                self.edges.append((best["id"], node["id"], 2))

        # Notehead -> stem: stems exist only in the engraving
        heads = [
            n
            for n in nodes
            if n.get("class") in ("noteheadWhole", "noteheadHalf", "noteheadBlack") and "id" in n
        ]
        children = {child for _, child, _ in self.edges}
        for node in nodes:
            if not (
                node["class"] in ("stem4", "stem8", "stem16", "stem32")
                and node.get("id")
                and node["id"] not in children
            ):
                continue
            s = node["bbox"]
            touching = [
                h
                for h in heads
                if h.get("_page") == node.get("_page")
                and s[0] <= h["bbox"][2]
                and s[2] >= h["bbox"][0]
                and s[1] <= h["bbox"][3]
                and s[3] >= h["bbox"][1]
            ]
            # Heads at the same height are unisons of different layers
            unisons: dict[int, list[dict[str, Any]]] = {}
            for head in touching:
                unisons.setdefault(round(_center(head["bbox"])[1]), []).append(head)
            stem_cy = (s[1] + s[3]) / 2.0
            for cy, group in unisons.items():
                if len(group) == 1:
                    self.edges.append((group[0]["id"], node["id"], 1))
                    continue
                # An up stem belongs to the upper layer (lower id), a down stem
                # to the lower one
                group.sort(key=lambda h: h["id"])
                target = group[0] if stem_cy < cy else group[-1]
                self.edges.append((target["id"], node["id"], 1))


def build_edges(
    mei: str | Path,
    pages: list[list[dict[str, Any]]],
    node_roles: dict[str, list[str]],
) -> list[Edge]:
    """Typed edges of one score.

    ``pages`` holds each page's detection annotations (``class``, ``bbox``,
    ``id``) in page order; the input is not modified.
    """
    root = ET.parse(mei).getroot() if isinstance(mei, Path) else ET.fromstring(mei)
    return _GraphBuilder(root, pages, node_roles).build()


def graph_record(
    edges: list[Edge], *, mei_file: str = "", json_files: list[str] | None = None
) -> dict[str, Any]:
    """Index-based graph (the layout of kakigori's .pt graphs, as plain JSON)."""
    node_ids = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    return {
        "node_ids": node_ids,
        "num_nodes": len(node_ids),
        "edge_index": [[index[u] for u, _, _ in edges], [index[v] for _, v, _ in edges]],
        "y": [kind for _, _, kind in edges],
        "mei_file": mei_file,
        "json_files": json_files or [],
    }
