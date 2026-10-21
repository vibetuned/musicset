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
