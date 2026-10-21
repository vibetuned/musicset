import json
import shutil
from pathlib import Path

from conftest import FIXTURES

from musicset.cli import main
from musicset.labels import LabelSet
from musicset.pipeline import BuildConfig, Layout, build_score
from musicset.views.kern import MinimalHumdrumSerializer, Spine

TINY_KERN = """**kern
*part1
*staff1
*clefG2
*k[f#]
*M4/4
=1
8gL
8aJ
16bL
16cc
16dd
16eeJ
32f#
32r
16r
8r
4g
==
*-
"""


def _build(tmp_path: Path, **cfg) -> Layout:
    (tmp_path / "src").mkdir(exist_ok=True)
    shutil.copy(FIXTURES / "tiny.musicxml", tmp_path / "src" / "tiny.musicxml")
    layout = Layout(tmp_path / "out")
    layout.create()
    build_score((tmp_path / "src" / "tiny.musicxml", layout, BuildConfig(**cfg), 0))
    return layout


def test_tiny_score_kern(tmp_path: Path):
    layout = _build(tmp_path, labels=LabelSet.load("lite"), write_kern=True)
    assert (layout.kern / "tiny.krn").read_text() == TINY_KERN
    manifest = json.loads((layout.manifest / "tiny.json").read_text())
    assert manifest["kern"] is True
    assert (layout.mei / "tiny.mei").exists()  # kept for re-projection


def test_cli_kern_matches_build(tmp_path: Path):
    layout = _build(tmp_path, labels=LabelSet.load("lite"), write_kern=True)
    out = tmp_path / "kern"
    main(["kern", str(layout.mei), str(layout.annotations), str(out), "--labels", "lite"])
    assert (out / "tiny.krn").read_text() == TINY_KERN


def test_from_pyg_reads_tensor_like_edges():
    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def tolist(self):
            return self.data

    serializer = MinimalHumdrumSerializer.from_pyg(
        FakeTensor([[0, 1], [1, 2]]), FakeTensor([1, 5]), {}, ["a", "b", "c"]
    )
    assert serializer.edges == [("a", "b", 1), ("b", "c", 5)]


def test_opening_signature_is_the_leftmost_on_the_staff():
    staff = {"bbox": [0, 100, 1000, 200]}
    nodes = {
        "change": {"class": "meterSig", "cx": 600, "cy": 150},  # mid-system change
        "opening": {"class": "meterSig", "cx": 80, "cy": 150},
        "elsewhere": {"class": "meterSig", "cx": 10, "cy": 900},  # another staff
    }
    order = ["change", "elsewhere", "opening"]
    assert Spine._leftmost_on_staff("metersig", staff, order, nodes) == "opening"


def test_geometric_slices_never_merge_one_layer():
    """Closely set notes of one layer are a run, not a chord; other layers at
    the same x still share the row."""
    from musicset.views.kern import HumdrumContext, Measure

    m = Measure("m")
    for i, token in enumerate(("12cc", "12cc", "12cc")):
        m.add(token, 1 / 12, 100.0 + 12 * i, f"n{i}", "layer1")
    m.add("4e", 0.25, 101.0, "v2", "layer2")
    slices, _ = HumdrumContext._slices_from_geometry([m])
    assert [row[0] for row in slices] == [["12cc", "4e"], ["12cc"], ["12cc"]]


def test_tuplet_ratio_from_written_total():
    def ratio(durations, dots=None):
        events = [{"id": str(i)} for i in range(len(durations))]
        infos = [
            {"duration": d, "dots": (dots or [0] * len(durations))[i]}
            for i, d in enumerate(durations)
        ]
        tuplet_of = {e["id"]: "t" for e in events}
        return MinimalHumdrumSerializer._tuplet_ratios(events, infos, tuplet_of).get("0")

    assert ratio([8, 16, 16, 8]) == (3, 2)  # 3/8, four events
    assert ratio([4, 8]) == (3, 2)  # quarter-eighth triplet, two events
    assert ratio([8, 8, 16]) == (3, 2)  # one misread notehead
    assert ratio([16] * 5) == (5, 4)
    assert ratio([16] * 6) == (3, 2)  # sextuplet: same durations as 3:2
    assert ratio([8, 8]) is None  # nothing to read


def test_reference_measure_has_the_system_staves():
    """Spines come from a measure with a staff cell per row: on a predicted
    graph the leftmost measure can miss its measure->staff edges."""
    nodes = {f"s{i}": {"class": "staff"} for i in range(5)}
    children = {
        "m1": [],  # lost its edges
        "m2": [("s0", 1), ("s1", 1)],
        "m3": [("s2", 1), ("s3", 1), ("s4", 1)],  # a wrong edge into another system
        "m4": [("s0", 1), ("note", 2)],
    }
    ref = MinimalHumdrumSerializer._reference_measure
    assert ref(["m1", "m2", "m3"], 2, children, nodes) == "m2"
    assert ref(["m1", "m4"], 2, children, nodes) == "m4"  # none complete: the most staves
    children["m1"] = [("s0", 1), ("s1", 1)]
    assert ref(["m1", "m2", "m3"], 2, children, nodes) == "m1"  # ground truth: the first


def test_signature_glyph_belongs_to_the_row_whose_band_holds_it():
    """A grand staff's bass clef linked to the treble cell by a predicted edge
    is not the treble staff's clef: a glyph is the staff's whose band (rows
    split at their midpoint) holds it."""
    from musicset.views.kern import _on_staff

    rows = [{"bbox": [0, 100, 1000, 140]}, {"bbox": [0, 200, 1000, 240]}]
    treble_cell = {"bbox": [0, 85, 500, 160]}  # content box, wider than the lines
    assert _on_staff({"cy": 120}, treble_cell, rows)
    assert _on_staff({"cy": 92}, treble_cell, rows)  # a sharp above the top line
    assert not _on_staff({"cy": 220}, treble_cell, rows)  # the bass staff's clef
    assert not _on_staff({"cy": 185}, treble_cell, rows)  # past the midpoint (170)
    assert _on_staff({"cy": 220}, treble_cell, [])  # no rows: no geometry to judge by


def test_event_reached_from_two_measures_is_emitted_by_the_one_holding_it():
    """A wrong edge across the barline (measure 2's layer under measure 1's
    staff cell) makes measure 1 reach measure 2's notes: each event is
    emitted once, by the reaching measure whose box holds it — the narrowest
    one, so a phantom measure spanning both never takes them."""
    nodes = {
        "m1": {"class": "measure", "bbox": [0, 0, 100, 50]},
        "m2": {"class": "measure", "bbox": [100, 0, 200, 50]},
        "m0": {"class": "measure", "bbox": [0, 0, 200, 50]},  # a phantom over both
        "s1": {"class": "staff", "bbox": [0, 10, 100, 40]},
        "s2": {"class": "staff", "bbox": [100, 10, 200, 40]},
        "s0": {"class": "staff", "bbox": [0, 10, 200, 40]},
        "l1": {"class": "layer"},
        "l2": {"class": "layer"},
        "n1": {"class": "note", "cx": 50.0, "cy": 25.0},
        "n2": {"class": "note", "cx": 150.0, "cy": 25.0},
        "n3": {"class": "note", "cx": 170.0, "cy": 25.0},
    }
    for node_id, node in nodes.items():
        node["id"] = node_id
    children = {
        "m1": [("s1", 1)],
        "m2": [("s2", 1)],
        "m0": [("s0", 1)],
        "s1": [("l1", 1), ("l2", 1)],  # l2: the wrong edge
        "s2": [("l2", 1)],
        "s0": [("l1", 1)],
        "l1": [("n1", 1)],
        "l2": [("n2", 1), ("n3", 1)],
    }
    serializer = MinimalHumdrumSerializer([], {})
    owner = serializer._measure_owners(["m1", "m2", "m0"], children, nodes)
    assert owner == {"n1": "m1", "n2": "m2", "n3": "m2"}

    serializer._measure_owner = owner
    rows = [[0, 10, 200, 40]]
    events = serializer._events_by_staff_band(["s1"], rows, children, nodes, measure_id="m1")
    assert [e["id"] for e in events["s1"][0]] == ["n1"]
    events = serializer._events_by_staff_band(["s2"], rows, children, nodes, measure_id="m2")
    assert sorted(e["id"] for e in events["s2"][0]) == ["n2", "n3"]
