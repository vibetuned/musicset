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
